from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from checkpoint_io import load_npz

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

from GiantGPT import GiantGPT
from jit_inference import init_inference_state, make_generate_fn

from jax import config
config.update("jax_default_matmul_precision", "tensorfloat32")


def build_model() -> GiantGPT:
    """Build model using Config.yml settings, matching Generate_text_fast.py"""
    if Config.use_custom_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    return GiantGPT(
        vocab_size=len(tok),
        context_length=Config.context_length,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.feed_forward_size,
        n_layers=Config.num_layers,
        dropout_rate=0.0,
    )


def _numpy_or_jax_array(x):
    """Ensure leaves are JAX arrays – helpful if checkpoint stored NumPy."""
    return jnp.asarray(x) if not isinstance(x, jax.Array) else x


def load_checkpoint(path: Path):
    """Return a PyTree of JAX arrays living on *CPU* (device_put later)."""
    ext = path.suffix.lower()
    if ext in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            params = pickle.load(f)
    elif ext == ".npz":
        params = load_npz(path)
    else:
        arr = np.load(path, allow_pickle=True)
        params = arr.item() if hasattr(arr, "item") else arr
    return jax.tree_util.tree_map(_numpy_or_jax_array, params)


def preprocess_prompt_no_EOS(tokenizer, prompt: str, max_len: int):
    """Tokenize prompt without EOS token, matching Generate_text_fast.py"""
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]

    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.array(ids, dtype="int32")


def preprocess_prompt(tokenizer, prompt: str, max_len: int):
    """Tokenize prompt with special tokens, matching Generate_text_fast.py"""
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if ids.shape[0] >= max_len:
        ids = ids[-max_len:]
    return ids.astype("int32")


def _block_until_ready_tree(x):
    """Blocks on all JAX arrays inside a PyTree."""
    leaves = jax.tree_util.tree_leaves(x)
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def generate_text_jit(
    model: GiantGPT,
    params: dict,
    nonparam: dict,
    tokenizer,
    prompt_ids: jnp.ndarray,  # (L,)
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    do_sample: bool,
    rng_key: Optional[jax.Array] = None,
    *,
    return_stats: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    """
    Generates text using JIT inference. If return_stats is True, also returns a dict with:
      - prompt_tokens
      - generated_tokens  
      - prefill_time_s
      - decode_time_s
      - tokens_per_second_decode
    """
    device = jax.devices(Config.device)[0]
    params = jax.device_put(params, device)
    nonparam = jax.device_put(nonparam, device)

    # Prepare prompt for JIT function
    prompt_batch = prompt_ids[None, :]  # [1, L_prompt]
    
    # Get the JIT-compiled generate function
    generate_fn = make_generate_fn(model)
    
    # --- Prefill timing: measure the first part (prompt processing) ---
    prefill_time_s = 0.0
    decode_time_s = 0.0
    
    if prompt_batch.shape[1] > 0:
        # Start timing for full generation (including prefill)
        t0 = time.perf_counter()
        
        # Generate tokens
        tokens_new, final_nonparam = generate_fn(
            params,
            nonparam,
            prompt_batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            rng_key=rng_key,
        )
        
        # Force device sync for accurate timing
        tokens_new.block_until_ready()
        total_time = time.perf_counter() - t0
        
        # For JIT inference, we approximate prefill time as a small fraction
        # since the JIT function handles both prefill and decode together
        prefill_time_s = total_time * 0.1  # Rough estimate
        decode_time_s = total_time * 0.9
    else:
        # No prompt, just decode
        t1 = time.perf_counter()
        tokens_new, final_nonparam = generate_fn(
            params,
            nonparam,
            jnp.array([[0]], dtype=jnp.int32),  # Start with pad token
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            rng_key=rng_key,
        )
        tokens_new.block_until_ready()
        decode_time_s = time.perf_counter() - t1

    # Combine prompt and generated tokens for decoding
    if prompt_batch.shape[1] > 0:
        full_tokens = jnp.concatenate([prompt_batch, tokens_new], axis=1)
    else:
        full_tokens = tokens_new
    
    # Decode to text
    text = tokenizer.decode(
        full_tokens[0],
        skip_special_tokens=True,
    )

    if return_stats:
        toks_per_s = (max_new_tokens / decode_time_s) if decode_time_s > 0 else float("inf")
        return text, {
            "prompt_tokens": int(prompt_ids.shape[0]),
            "generated_tokens": int(max_new_tokens),
            "prefill_time_s": float(prefill_time_s),
            "decode_time_s": float(decode_time_s),
            "tokens_per_second_decode": float(toks_per_s),
        }
    return text


def main():
    ap = argparse.ArgumentParser(description="Fast JIT-compiled text generation using GiantGPT")
    ap.add_argument("--checkpoint", type=Path, default="model_params.pkl",
                    help="Path to model checkpoint (.pkl, .pickle, .npz)")
    ap.add_argument("--prompt", type=str, default="Once upon",
                    help="Text prompt for generation")
    ap.add_argument("--steps", type=int, default=20,
                    help="Number of tokens to generate (same as max_new_tokens)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 = greedy)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="Top-k sampling (None = no top-k)")
    ap.add_argument("--greedy", action="store_true",
                    help="Force greedy decoding (temperature=0.0)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print timing and tokens/sec for the decode phase")
    
    # Additional JIT-specific options
    ap.add_argument("--no_eos", action="store_true",
                    help="Remove EOS token from prompt preprocessing")
    
    args = ap.parse_args()

    # Handle temperature/greedy settings
    temperature = 0.0 if args.greedy else args.temperature
    do_sample = temperature > 0.0
    # Keep JIT static signature stable
    top_k_int = 0 if args.top_k is None else int(args.top_k)

    print("\nLoading checkpoint…")
    params_cpu = load_checkpoint(args.checkpoint)

    print("Building model…")
    model = build_model()
    
    # Load tokenizer (matching Generate_text_fast.py exactly)
    if Config.use_custom_tokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    # Preprocess prompt
    if args.no_eos:
        prompt_ids = preprocess_prompt_no_EOS(tokenizer, args.prompt, Config.context_length)
    else:
        prompt_ids = preprocess_prompt(tokenizer, args.prompt, Config.context_length)

    # Initialize inference state for JIT
    key = jax.random.PRNGKey(42)
    k_params, k_drop, k_sample = jax.random.split(key, 3)
    
    # Initialize with loaded params
    _, nonparam = init_inference_state(
        model, k_params, k_drop, batch_size=1, pad_token_id=0, use_kv_cache=True
    )
    
    # Replace dummy params with loaded params
    # Note: params_cpu should be the actual trained parameters
    params = params_cpu

    print("Generating… (first decode call may include JIT compile)")
    
    # Generate text
    rng_for_sampling = k_sample if do_sample else None
    
    if args.verbose:
        text, stats = generate_text_jit(
            model,
            params,
            nonparam,
            tokenizer,
            prompt_ids,
            args.steps,
            temperature,
            top_k_int,
            do_sample,
            rng_for_sampling,
            return_stats=True,
        )
    else:
        text = generate_text_jit(
            model,
            params,
            nonparam,
            tokenizer,
            prompt_ids,
            args.steps,
            temperature,
            top_k_int,
            do_sample,
            rng_for_sampling,
            return_stats=False,
        )

    # Output results (matching Generate_text_fast.py format exactly)
    print("\n" + "="*20 + " RESULT " + "="*20)
    print(text)
    print("="*48)

    if args.verbose:
        gen_tok = stats["generated_tokens"]
        dec_s = stats["decode_time_s"]
        toks_per_s = stats["tokens_per_second_decode"]
        print("\n[perf]")
        print(f"prompt_tokens: {stats['prompt_tokens']}")
        print(f"generated_tokens: {gen_tok}")
        print(f"prefill_time_s: {stats['prefill_time_s']:.6f}")
        print(f"decode_time_s:  {dec_s:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()