"""
Optimized JIT inference for compressed BgGPT using lax.scan.
Based on model/jit_inference.py approach.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple, List

import jax
import jax.numpy as jnp

from bggpt_compressed_kv_model_jax import CompressedBgGPTForCausalLM

Array = jnp.ndarray
PyTree = Dict[str, Any]


def init_inference_cache(
    model: CompressedBgGPTForCausalLM,
    batch_size: int,
    max_length: int,
    dtype: jnp.dtype = jnp.float16,  # Changed default to float16 to match params
) -> List[Tuple[Array, Array]]:
    """
    Pre-allocate KV cache for inference.
    Returns list of (k_comp, v_comp) tuples, one per layer.
    """
    latent_dim = max(1, int(model.head_dim * model.kv_compression_ratio))
    
    # Use float16 if bfloat16 is requested but not supported, or just match model dtype
    # But here we just use what's passed in.
    
    cache = []
    for _ in range(model.num_layers):
        k_comp = jnp.zeros(
            (batch_size, model.num_kv_heads, max_length, latent_dim),
            dtype=dtype
        )
        v_comp = jnp.zeros(
            (batch_size, model.num_kv_heads, max_length, latent_dim),
            dtype=dtype
        )
        cache.append((k_comp, v_comp))
    
    return cache


def prefill_prompt(
    apply_fn,
    params: PyTree,
    prompt_tokens: Array,  # [B, L]
    past_key_values: Optional[List[Tuple[Array, Array]]] = None,
) -> Tuple[Array, List[Tuple[Array, Array]], Array]:
    """
    Process the full prompt and return logits + filled cache + last position.
    
    Returns:
        logits: [B, L, vocab]
        past_key_values: List of (k_comp, v_comp) per layer
        last_pos: scalar indicating last token position
    """
    # If past_key_values is provided (fixed cache), we use cache_position=0
    # to update it in place.
    cache_position = jnp.array(0, dtype=jnp.int32) if past_key_values is not None else None
    
    logits, past_kv = apply_fn(
        {'params': params},
        prompt_tokens,
        past_key_values=past_key_values,
        use_cache=True,
        deterministic=True,
        cache_position=cache_position,
    )
    
    last_pos = jnp.array(prompt_tokens.shape[1] - 1, dtype=jnp.int32)
    return logits, past_kv, last_pos


def _top_k_logits(logits: Array, k: int) -> Array:
    """Mask everything below the kth largest logit."""
    if k <= 0:
        return logits
    top_values, _ = jax.lax.top_k(logits, k)
    kth = top_values[..., -1, None]
    return jnp.where(logits < kth, float('-inf'), logits)


@partial(
    jax.jit,
    static_argnames=("apply_fn", "steps", "do_sample", "top_k"),
)
def decode_tokens(
    apply_fn,
    params: PyTree,
    past_key_values: List[Tuple[Array, Array]],
    last_token: Array,  # [B, 1]
    start_pos: Array,   # Scalar or [1]
    *,
    steps: int,
    do_sample: bool = False,
    top_k: int = 0,
    temperature: float = 1.0,
    rng_key: Optional[Array] = None,
) -> Tuple[Array, List[Tuple[Array, Array]]]:
    """
    Decode `steps` tokens autoregressively using lax.scan.
    
    Args:
        apply_fn: The model's apply function
        params: Model parameters
        past_key_values: Initial KV cache state
        last_token: Last token from prompt [B, 1]
        steps: Number of tokens to generate
        do_sample: Whether to sample or use greedy decoding
        top_k: Top-k filtering (0 disables)
        temperature: Sampling temperature
        rng_key: RNG key for sampling (required if do_sample=True)
    
    Returns:
        tokens: Generated tokens [B, steps]
        past_key_values: Updated KV cache
    """
    B = last_token.shape[0]
    out_tokens = jnp.zeros((B, steps), dtype=jnp.int32)
    
    def body(carry, i):
        past_kv, tok_prev, rng, out, current_pos = carry
        
        # Forward pass with single token
        logits, new_past_kv = apply_fn(
            {'params': params},
            tok_prev,
            past_key_values=past_kv,
            use_cache=True,
            deterministic=True,
            cache_position=current_pos,
        )
        
        # Get logits for next token
        step_logits = logits[:, -1, :]
        
        # Apply temperature and top-k
        if do_sample:
            scaled = step_logits / jnp.maximum(temperature, 1e-6)
            scaled = _top_k_logits(scaled, top_k) if top_k > 0 else scaled
            rng, sub = jax.random.split(rng)
            next_tok = jax.random.categorical(sub, scaled, axis=-1)
        else:
            next_tok = jnp.argmax(step_logits, axis=-1)
        
        # Update output array
        out = jax.lax.dynamic_update_slice(out, next_tok[:, None], (0, i))
        next_tok_2d = next_tok[:, None]
        
        # Explicitly cast to match input dtype if needed
        def cast_to_input(new, old):
            return new.astype(old.dtype)
            
        new_past_kv = jax.tree_util.tree_map(cast_to_input, new_past_kv, past_kv)
        
        # Increment position
        next_pos = current_pos + 1
        
        return (new_past_kv, next_tok_2d, rng, out, next_pos), None
    
    # Initialize RNG
    init_rng = rng_key if do_sample else jnp.zeros((2,), jnp.uint32)
    
    # Ensure start_pos is correct shape/type
    start_pos = jnp.asarray(start_pos, dtype=jnp.int32)
    
    # Run scan loop
    (past_kv, _, _, out_tokens, _), _ = jax.lax.scan(
        body,
        init=(past_key_values, last_token, init_rng, out_tokens, start_pos),
        xs=jnp.arange(steps, dtype=jnp.int32),
    )
    
    return out_tokens, past_kv


def make_generate_fn(
    model: CompressedBgGPTForCausalLM,
    params: PyTree,
):
    """
    Create a compiled generation function.
    
    Returns a function that takes (prompt_ids, max_new_tokens, temperature, top_k, seed)
    and returns generated tokens.
    """
    
    apply_fn = model.apply
    
    def generate(
        prompt_ids: Array,  # [B, L]
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        seed: int = 0,
    ) -> Array:
        """Generate tokens from prompt."""
        
        B, L = prompt_ids.shape
        max_length = L + max_new_tokens
        
        # Initialize fixed-size cache
        # We use float16 to match model params/computation
        past_key_values = init_inference_cache(model, B, max_length, dtype=jnp.float16)
        
        # Prefill (updates cache in-place starting at 0)
        logits, past_kv, last_pos = prefill_prompt(apply_fn, params, prompt_ids, past_key_values)
        last_token = prompt_ids[:, -1:]
        
        # Decode
        do_sample = temperature > 0.0
        rng_key = jax.random.PRNGKey(seed) if do_sample else None
        
        # Start position for decoding is last_pos + 1
        start_pos = last_pos + 1
        
        new_tokens, _ = decode_tokens(
            apply_fn=apply_fn,
            params=params,
            past_key_values=past_kv,
            last_token=last_token,
            start_pos=start_pos,
            steps=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            rng_key=rng_key,
        )
        
        # Concatenate prompt and generated tokens
        full_sequence = jnp.concatenate([prompt_ids, new_tokens], axis=1)
        return full_sequence
    
    return generate


# Simplified interface functions

def create_compiled_generate_fn(
    model: CompressedBgGPTForCausalLM,
    params: PyTree,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
):
    """
    Create a fully compiled generation function with fixed generation parameters.
    This is faster than the flexible version but requires recompilation if params change.
    """
    
    @partial(
        jax.jit,
        static_argnames=("steps", "do_sample", "top_k")
    )
    def compiled_generate(
        prompt_ids: Array,
        seed: int = 0,
        steps: int = max_new_tokens,
        temp: float = temperature,
        topk: int = top_k,
        do_sample: bool = True,
    ) -> Array:
        B, L = prompt_ids.shape
        max_length = L + steps
        
        # Initialize fixed-size cache
        past_key_values = init_inference_cache(model, B, max_length, dtype=jnp.float16)
        
        # Prefill
        logits, past_kv, last_pos = prefill_prompt(model.apply, params, prompt_ids, past_key_values)
        last_token = prompt_ids[:, -1:]
        
        # Decode
        rng_key = jax.random.PRNGKey(seed) if do_sample else None
        start_pos = last_pos + 1
        
        new_tokens, _ = decode_tokens(
            apply_fn=model.apply,
            params=params,
            past_key_values=past_kv,
            last_token=last_token,
            start_pos=start_pos,
            steps=steps,
            do_sample=do_sample,
            top_k=topk,
            temperature=temp,
            rng_key=rng_key,
        )
        
        return jnp.concatenate([prompt_ids, new_tokens], axis=1)
    
    return compiled_generate


def benchmark_generate(
    model: CompressedBgGPTForCausalLM,
    params: PyTree,
    prompt_ids: Array,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    seed: int = 0,
    num_warmup: int = 2,
    num_runs: int = 5,
):
    """
    Benchmark generation speed.
    
    Returns average tokens/second over multiple runs.
    """
    import time
    
    generate_fn = make_generate_fn(model, params)
    
    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        output = generate_fn(prompt_ids, max_new_tokens, temperature, top_k, seed)
        output.block_until_ready()
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        output = generate_fn(prompt_ids, max_new_tokens, temperature, top_k, seed)
        output.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s ({max_new_tokens/elapsed:.2f} tok/s)")
    
    avg_time = sum(times) / len(times)
    avg_toks_per_sec = max_new_tokens / avg_time
    
    print(f"\nAverage: {avg_time:.4f}s ({avg_toks_per_sec:.2f} tok/s)")
    
    return avg_toks_per_sec
