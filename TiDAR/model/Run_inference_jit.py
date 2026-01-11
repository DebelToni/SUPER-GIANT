"""
TiDAR inference with full JIT compilation.

Key optimizations:
1. All sampling happens on device using JAX (no numpy)
2. Decode loop uses lax.while_loop (no Python loops)
3. Single JITed function for the entire decode step
4. Minimal host-device transfers
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TiDAR.model.TiDAR import TiDAR
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from v2.model.checkpoint_manager import load_npz, latest as latest_ckpt

DEFAULT_CACHE_BUCKETS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)


def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(model_dir / "Config.yml"),
    )

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str((base_prefix / path).resolve())

    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)

    for key in ("checkpoints_root", "hf_cache_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "tokenizer" in cfg:
        cache_dir = cfg.tokenizer.get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cfg.tokenizer.cache_dir = str(Path(cfg.paths.data_root) / cache_path)
        custom_path = cfg.tokenizer.get("custom_path")
        if custom_path:
            custom_path = Path(str(custom_path))
            if not custom_path.is_absolute():
                cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path)

    return cfg


def resolve_checkpoint_path(cfg: OmegaConf, checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    base_root = Path(cfg.paths.data_root)
    ckpt_dir = Path(checkpoint_dir or cfg.paths.checkpoints_root)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()

    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        if path.is_dir():
            latest = latest_ckpt(str(path))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {path}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

    latest = latest_ckpt(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return Path(latest)


def load_tokenizer(cfg: OmegaConf):
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def build_model(cfg: OmegaConf, vocab_size: int, context_length: int) -> TiDAR:
    model_cfg = cfg.model
    return TiDAR(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
    )


def tokenize_prompt(tokenizer, prompt: str, max_len: int, *, strip_eos: bool) -> np.ndarray:
    if strip_eos:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
    else:
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()

    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.asarray(ids, dtype=np.int32)


def load_params(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


# ==============================================================================
# JAX-native sampling functions (all on device, no numpy)
# ==============================================================================

def sample_from_logits_jax(
    logits: jnp.ndarray,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
) -> jnp.ndarray:
    """Sample from logits using JAX (on device).
    
    Args:
        logits: shape (..., vocab_size)
        rng_key: JAX random key
        temperature: sampling temperature
        top_k: if > 0, only sample from top_k tokens
        
    Returns:
        Sampled token ids with shape (...)
    """
    # Handle greedy decoding
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)
    
    # Scale by temperature
    scaled = logits / jnp.maximum(temperature, 1e-6)
    
    # Apply top-k masking
    if top_k > 0:
        # Get top-k values and create mask
        top_k_values = jax.lax.top_k(scaled, top_k)[0]
        threshold = top_k_values[..., -1:]  # minimum value in top-k
        scaled = jnp.where(scaled >= threshold, scaled, -jnp.inf)
    
    # Sample using Gumbel-max trick (more efficient than explicit softmax + categorical)
    # This is equivalent to: sample from softmax(scaled)
    gumbel_noise = jax.random.gumbel(rng_key, scaled.shape, dtype=scaled.dtype)
    return jnp.argmax(scaled + gumbel_noise, axis=-1)


def sample_batch_from_logits_jax(
    logits: jnp.ndarray,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
) -> jnp.ndarray:
    """Sample from logits for a batch of positions.
    
    Args:
        logits: shape (batch, seq, vocab_size) or (seq, vocab_size)
        rng_key: JAX random key
        
    Returns:
        Sampled tokens with shape (batch, seq) or (seq,)
    """
    if logits.ndim == 2:
        # (seq, vocab)
        keys = jax.random.split(rng_key, logits.shape[0])
        return jax.vmap(
            lambda l, k: sample_from_logits_jax(l, k, temperature, top_k)
        )(logits, keys)
    else:
        # (batch, seq, vocab)
        batch_size, seq_len = logits.shape[:2]
        keys = jax.random.split(rng_key, batch_size * seq_len).reshape(batch_size, seq_len, 2)
        return jax.vmap(
            jax.vmap(lambda l, k: sample_from_logits_jax(l, k, temperature, top_k))
        )(logits, keys)


# ==============================================================================
# Attention bias builders (same as original but optimized)
# ==============================================================================

@lru_cache(maxsize=None)
def build_prefill_bias_template(max_seq_len: int, draft_len: int, bias_value: float) -> jnp.ndarray:
    """TiDAR-specific: Fig.7 prefill mask, cached and sliced per request."""
    total = draft_len + max_seq_len
    idx = jnp.arange(total)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    is_mask_q = q_idx < draft_len
    is_mask_k = k_idx < draft_len
    is_prompt_k = k_idx >= draft_len

    allow_mask_to_mask = is_mask_q & is_mask_k
    allow_mask_to_prompt = is_mask_q & is_prompt_k

    prompt_q_pos = q_idx - draft_len
    prompt_k_pos = k_idx - draft_len
    is_prompt_q = q_idx >= draft_len
    allow_prompt_to_prompt = is_prompt_q & is_prompt_k & (prompt_k_pos <= prompt_q_pos)

    allow = allow_mask_to_mask | allow_mask_to_prompt | allow_prompt_to_prompt
    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


@lru_cache(maxsize=None)
def build_decode_bias_template(cache_len: int, draft_len: int, bias_value: float) -> jnp.ndarray:
    """TiDAR-specific: constant draft->draft + draft->prefix template."""
    step_len = draft_len + (draft_len * draft_len)
    key_len = cache_len + step_len

    q_idx = jnp.arange(step_len)[:, None]
    k_idx = jnp.arange(key_len)[None, :]

    is_verify_q = q_idx < draft_len
    is_cand_q = q_idx >= draft_len
    is_prefix_k = k_idx < cache_len
    is_step_k = k_idx >= cache_len

    step_k_idx = k_idx - cache_len
    is_verify_k = is_step_k & (step_k_idx < draft_len)
    is_cand_k = is_step_k & (step_k_idx >= draft_len)

    allow_verify_to_prefix = is_verify_q & is_prefix_k
    allow_verify_to_verify = is_verify_q & is_verify_k & (step_k_idx <= q_idx)

    cand_q_offset = q_idx - draft_len
    cand_k_offset = step_k_idx - draft_len
    cand_q_block = cand_q_offset // draft_len
    cand_k_block = cand_k_offset // draft_len
    cand_r = cand_q_block + 1

    allow_cand_to_prefix = is_cand_q & is_prefix_k
    allow_cand_to_verify = is_cand_q & is_verify_k & (step_k_idx < cand_r)
    allow_cand_to_cand = is_cand_q & is_cand_k & (cand_q_block == cand_k_block)

    allow = (
        allow_verify_to_prefix
        | allow_verify_to_verify
        | allow_cand_to_prefix
        | allow_cand_to_verify
        | allow_cand_to_cand
    )

    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


def parse_cache_buckets(raw: Optional[str], *, context_length: int) -> Tuple[int, ...]:
    if raw is None:
        return tuple(b for b in DEFAULT_CACHE_BUCKETS if b <= context_length)
    buckets = tuple(sorted({int(v) for v in raw.split(",") if v.strip()}))
    return tuple(b for b in buckets if b <= context_length)


def select_cache_bucket(required_len: int, buckets) -> int:
    for bucket in buckets:
        if bucket >= required_len:
            return bucket
    raise ValueError(f"No cache bucket >= {required_len} (available: {buckets})")


# ==============================================================================
# Position ID builders
# ==============================================================================

def build_prefill_position_ids(prefix_len: int, draft_len: int) -> jnp.ndarray:
    """Positions for the prefill K mask tokens."""
    return jnp.arange(prefix_len, prefix_len + draft_len, dtype=jnp.int32)


def build_decode_position_ids(prefix_len: int, draft_len: int) -> jnp.ndarray:
    """Positions for TiDAR decode layout [VERIFY | PREDRAFT].
    
    Note: For use inside JIT, use build_decode_position_ids_template + offset instead.
    """
    pos_verify = jnp.arange(prefix_len, prefix_len + draft_len, dtype=jnp.int32)
    # Candidate positions: for r in [1..K], positions [prefix_len+r, prefix_len+r+K-1]
    offsets = (jnp.arange(1, draft_len + 1, dtype=jnp.int32)[:, None] + 
               jnp.arange(draft_len, dtype=jnp.int32)[None, :])
    pos_predraft = prefix_len + offsets.reshape(-1)
    return jnp.concatenate([pos_verify, pos_predraft], axis=0)


def build_decode_position_ids_template(draft_len: int) -> jnp.ndarray:
    """Build position offsets template for decode step (0-indexed).
    
    Returns offsets that can be added to cache_idx to get actual positions.
    This allows the template to be precomputed and used with traced cache_idx.
    """
    # Verify positions: [0, 1, ..., K-1] (relative to cache_idx)
    pos_verify = jnp.arange(draft_len, dtype=jnp.int32)
    # Candidate positions: for r in [1..K], offsets [r, r+1, ..., r+K-1]
    offsets = (jnp.arange(1, draft_len + 1, dtype=jnp.int32)[:, None] + 
               jnp.arange(draft_len, dtype=jnp.int32)[None, :])
    pos_predraft = offsets.reshape(-1)
    return jnp.concatenate([pos_verify, pos_predraft], axis=0)


# ==============================================================================
# KV Cache initialization and management
# ==============================================================================

def init_kv_cache(model: TiDAR, *, batch_size: int, pad_token_id: int) -> object:
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
    )
    return variables["cache"]


def prefill_prompt_cache(
    model: TiDAR,
    params,
    cache_vars,
    prompt_ids: jnp.ndarray,
    *,
    kv_cache_len: int,
) -> Tuple[object, int]:
    """Prefill the KV cache with prompt tokens."""
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
    batch_size, prompt_len = prompt_ids.shape
    if prompt_len == 0:
        return cache_vars, 0

    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, prompt_len))

    _, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        prompt_ids,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=0,
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        mutable=["cache"],
    )
    return mutated["cache"], prompt_len


# ==============================================================================
# JIT-compiled model application functions
# ==============================================================================

def make_prefill_draft_fn(
    model: TiDAR,
    *,
    cache_len: int,
    prefill_bias: jnp.ndarray,
):
    """Create JITed prefill function."""
    prefill_bias = jax.device_put(prefill_bias)

    @jax.jit
    def prefill_draft(params, cache_vars, tokens, position_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=prefill_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )
        return logits

    return prefill_draft


def make_decode_step_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    bias_value: float,
):
    """Create JITed decode step function."""
    decode_bias = build_decode_bias_template(cache_len, draft_len, bias_value)
    decode_bias = jax.device_put(decode_bias)

    @jax.jit
    def decode_step(params, cache_vars, tokens, position_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )
        return logits

    return decode_step


def make_commit_fn(
    model: TiDAR,
    *,
    kv_cache_len: int,
    max_commit_len: int,
):
    """Create JITed cache commit function.
    
    Uses static shapes with padding to avoid recompilation.
    """
    @jax.jit
    def commit_tokens(params, cache_vars, tokens, prefix_len, actual_len):
        """Commit tokens to cache.
        
        Args:
            tokens: shape (1, max_commit_len) - padded to max length
            prefix_len: current cache index
            actual_len: actual number of tokens to commit
        """
        # Build position ids
        position_ids = prefix_len + jnp.arange(max_commit_len, dtype=jnp.int32)
        position_ids = position_ids[None, :]
        
        # Apply model to update cache
        _, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=prefix_len,
            position_ids=position_ids,
            kv_cache_len=kv_cache_len,
            cache_write_len=actual_len,  # Only write actual_len tokens
            mutable=["cache"],
        )
        return mutated["cache"]

    return commit_tokens


# ==============================================================================
# Full JIT-compiled decode loop
# ==============================================================================

def make_full_decode_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    max_steps: int,
    mask_id: int,
    bias_value: float,
    temperature: float,
    top_k: int,
    always_accept: bool,
):
    """Create a fully JIT-compiled decode function using lax.while_loop.
    
    This function runs the entire decode loop on device without returning to Python.
    """
    decode_bias = build_decode_bias_template(cache_len, draft_len, bias_value)
    decode_bias = jax.device_put(decode_bias)
    step_len = draft_len + (draft_len * draft_len)
    
    # Precompute position offsets template (can be added to cache_idx)
    step_position_offsets = build_decode_position_ids_template(draft_len)
    step_position_offsets = jax.device_put(step_position_offsets)
    
    # Precompute commit position offsets
    commit_position_offsets = jnp.arange(draft_len, dtype=jnp.int32)
    commit_position_offsets = jax.device_put(commit_position_offsets)
    
    # Precompute predraft masks (constant)
    predraft_masks = jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32)
    predraft_masks = jax.device_put(predraft_masks)

    def decode_step_inner(params, cache_vars, tokens, position_ids, prefix_len):
        """Single decode step - returns logits."""
        return model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )

    def commit_step_inner(params, cache_vars, tokens, position_ids, prefix_len, write_len):
        """Commit tokens to cache."""
        _, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=prefix_len,
            position_ids=position_ids,
            kv_cache_len=cache_len,
            cache_write_len=write_len,
            mutable=["cache"],
        )
        return mutated["cache"]

    @partial(jax.jit, static_argnames=['eos_id'])
    def full_decode(
        params,
        cache_vars,
        initial_draft_tokens: jnp.ndarray,  # (draft_len,)
        initial_draft_logits: jnp.ndarray,  # (draft_len, vocab_size)
        initial_prefix_ids: jnp.ndarray,    # (prompt_len,)
        cache_index: int,
        rng_key: jax.Array,
        eos_id: Optional[int],
    ):
        """Run the full decode loop on device.
        
        Returns:
            generated_tokens: (buffer_size,) - generated token ids (padded)
            num_generated: scalar - actual number of generated tokens
        """
        vocab_size = initial_draft_logits.shape[-1]
        
        # Round up buffer size to multiple of draft_len for safe slicing
        buffer_size = ((max_steps + draft_len - 1) // draft_len) * draft_len
        
        # State for lax.while_loop
        # - generated_tokens: buffer for generated tokens
        # - num_generated: count of generated tokens
        # - draft_tokens: current draft (K tokens)
        # - draft_logits: logits for current draft (for rejection sampling)
        # - cache_vars: KV cache
        # - cache_index: current position in cache
        # - rng_key: random key
        # - done: whether to stop
        
        initial_state = (
            jnp.zeros((buffer_size,), dtype=jnp.int32),  # generated_tokens
            jnp.int32(0),                               # num_generated
            initial_draft_tokens,                       # draft_tokens
            initial_draft_logits,                       # draft_logits  
            cache_vars,                                 # cache
            jnp.int32(cache_index),                    # cache_index
            rng_key,                                    # rng_key
            jnp.bool_(False),                          # done
        )
        
        def cond_fn(state):
            (generated_tokens, num_generated, draft_tokens, draft_logits,
             cache, cache_idx, key, done) = state
            return ~done & (num_generated < max_steps)
        
        def body_fn(state):
            (generated_tokens, num_generated, draft_tokens, draft_logits,
             cache, cache_idx, key, done) = state
            
            key, sample_key, cand_key = jax.random.split(key, 3)
            
            # Build step tokens: [verify (K) | predraft masks (K*K)]
            step_tokens = jnp.concatenate([draft_tokens, predraft_masks], axis=0)
            step_tokens = step_tokens[None, :]  # (1, step_len)
            
            # Build position ids by adding cache_idx to precomputed offsets
            step_position_ids = (cache_idx + step_position_offsets)[None, :]
            
            # Forward pass
            step_logits = decode_step_inner(
                params, cache, step_tokens, step_position_ids, cache_idx
            )
            step_logits = step_logits[0]  # (step_len, vocab_size)
            
            # Split logits
            verify_logits = step_logits[:draft_len]  # (K, vocab)
            cand_logits = step_logits[draft_len:].reshape(draft_len, draft_len, vocab_size)  # (K, K, vocab)
            
            # Sample candidates for each block
            cand_keys = jax.random.split(cand_key, draft_len)
            candidate_tokens = jax.vmap(
                lambda logits, k: sample_batch_from_logits_jax(logits, k, temperature, top_k)
            )(cand_logits, cand_keys)  # (K, K)
            
            # Determine committed tokens (always accept all draft tokens)
            # Note: We always commit draft_len tokens and use masking for EOS
            committed = draft_tokens
            
            # Write all draft_len generated tokens to buffer
            # Use lax.dynamic_update_slice for in-place update
            generated_tokens = lax.dynamic_update_slice(
                generated_tokens,
                committed,
                (num_generated,)
            )
            num_generated = num_generated + draft_len
            
            # Commit to cache (always commit all draft_len tokens)
            commit_position_ids = (cache_idx + commit_position_offsets)[None, :]
            cache = commit_step_inner(
                params, cache, committed[None, :], commit_position_ids, 
                cache_idx, draft_len
            )
            cache_idx = cache_idx + draft_len
            
            # Check for EOS if enabled
            if eos_id is not None:
                has_eos = jnp.any(committed == eos_id)
                done = done | has_eos
            
            # Check if we've generated enough
            done = done | (num_generated >= max_steps)
            
            # Select next draft based on r (always draft_len for always_accept)
            block_idx = draft_len - 1
            next_draft_tokens = candidate_tokens[block_idx]
            next_draft_logits = cand_logits[block_idx]
            
            return (generated_tokens, num_generated, next_draft_tokens, next_draft_logits,
                    cache, cache_idx, key, done)
        
        final_state = lax.while_loop(cond_fn, body_fn, initial_state)
        generated_tokens, num_generated = final_state[0], final_state[1]
        
        return generated_tokens, num_generated

    return full_decode


# ==============================================================================
# Simpler version: JITed single step with fori_loop for fixed iterations
# ==============================================================================

def make_decode_loop_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    max_steps: int,
    mask_id: int,
    bias_value: float,
    temperature: float,
    top_k: int,
):
    """Create a decode function using lax.fori_loop for fixed number of iterations.
    
    This is simpler than while_loop and works well when we want exactly max_steps tokens.
    """
    decode_bias = build_decode_bias_template(cache_len, draft_len, bias_value)
    decode_bias = jax.device_put(decode_bias)
    
    # Precompute position offsets template (can be added to cache_idx)
    step_position_offsets = build_decode_position_ids_template(draft_len)
    step_position_offsets = jax.device_put(step_position_offsets)
    
    # Precompute commit position offsets
    commit_position_offsets = jnp.arange(draft_len, dtype=jnp.int32)
    commit_position_offsets = jax.device_put(commit_position_offsets)
    
    # Precompute predraft masks (constant)
    predraft_masks = jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32)
    predraft_masks = jax.device_put(predraft_masks)
    
    # Number of iterations = ceil(max_steps / draft_len)
    num_iterations = (max_steps + draft_len - 1) // draft_len

    def decode_step_inner(params, cache_vars, tokens, position_ids, prefix_len):
        return model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )

    def commit_step_inner(params, cache_vars, tokens, position_ids, prefix_len, write_len):
        _, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=prefix_len,
            position_ids=position_ids,
            kv_cache_len=cache_len,
            cache_write_len=write_len,
            mutable=["cache"],
        )
        return mutated["cache"]

    @jax.jit
    def decode_loop(
        params,
        cache_vars,
        initial_draft_tokens: jnp.ndarray,
        cache_index: int,
        rng_key: jax.Array,
    ):
        """Run decode loop for fixed number of iterations.
        
        Always accepts all draft tokens (--always_accept mode).
        """
        
        def body_fn(i, state):
            (generated_tokens, cache, cache_idx, draft_tokens, key) = state
            
            key, sample_key = jax.random.split(key)
            
            # Build step tokens (draft + predraft masks)
            step_tokens = jnp.concatenate([draft_tokens, predraft_masks], axis=0)
            step_tokens = step_tokens[None, :]
            
            # Build position ids by adding cache_idx to precomputed offsets
            step_position_ids = (cache_idx + step_position_offsets)[None, :]
            
            # Forward pass
            step_logits = decode_step_inner(
                params, cache, step_tokens, step_position_ids, cache_idx
            )
            step_logits = step_logits[0]
            
            # Get candidate logits
            cand_logits = step_logits[draft_len:].reshape(draft_len, draft_len, -1)
            
            # Sample from last candidate block (r=draft_len in always_accept mode)
            next_draft_logits = cand_logits[-1]  # (draft_len, vocab)
            next_draft_tokens = sample_batch_from_logits_jax(
                next_draft_logits, sample_key, temperature, top_k
            )
            
            # Commit current draft tokens to output buffer
            write_start = i * draft_len
            generated_tokens = lax.dynamic_update_slice(
                generated_tokens, draft_tokens, (write_start,)
            )
            
            # Update cache with current draft tokens
            commit_position_ids = (cache_idx + commit_position_offsets)[None, :]
            cache = commit_step_inner(
                params, cache, draft_tokens[None, :], commit_position_ids,
                cache_idx, draft_len
            )
            cache_idx = cache_idx + draft_len
            
            return (generated_tokens, cache, cache_idx, next_draft_tokens, key)
        
        initial_state = (
            jnp.zeros((num_iterations * draft_len,), dtype=jnp.int32),
            cache_vars,
            jnp.int32(cache_index),
            initial_draft_tokens,
            rng_key,
        )
        
        final_state = lax.fori_loop(0, num_iterations, body_fn, initial_state)
        generated_tokens = final_state[0]
        
        # Trim to max_steps
        return generated_tokens[:max_steps]

    return decode_loop


# ==============================================================================
# Main
# ==============================================================================

def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TiDAR inference (JIT optimized)")
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=_parse_bool, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    parser.add_argument("--always_accept", action="store_true")
    parser.add_argument("--cache_buckets", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_while_loop", action="store_true", 
                       help="Use while_loop instead of fori_loop (for variable length)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)

    stop_on_eos = args.stop_on_eos if args.stop_on_eos is not None else bool(cfg.inference.stop_on_eos)

    model_context_length = int(cfg.model.context_length)
    context_length = args.context_length if args.context_length is not None else model_context_length

    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if draft_len <= 0:
        raise ValueError("draft_len must be > 0")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")
    if context_length > model_context_length:
        raise ValueError(
            f"context_length {context_length} exceeds model context_length {model_context_length}."
        )

    # Ensure max_steps is divisible by draft_len for fori_loop
    if not args.use_while_loop:
        # Round up to nearest multiple of draft_len
        original_steps = max_steps
        max_steps = ((max_steps + draft_len - 1) // draft_len) * draft_len
        if max_steps != original_steps and args.verbose:
            print(f"[info] Rounded max_steps from {original_steps} to {max_steps} for fori_loop")

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens. Provide non-empty text.")

    prompt_len = int(prompt_ids.shape[0])
    cache_buckets = parse_cache_buckets(args.cache_buckets, context_length=context_length)
    if not cache_buckets:
        raise ValueError("No valid cache buckets available within context length.")
    required_cache_len = prompt_len + max_steps
    if required_cache_len > context_length:
        raise ValueError(
            f"prompt_len + max_steps ({required_cache_len}) exceeds context_length {context_length}."
        )
    cache_len = select_cache_bucket(required_cache_len, cache_buckets)
    print(f"Using cache bucket: {cache_len}")

    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"[mask] added token '{mask_token}' (id={mask_id})")
    else:
        print(f"[mask] using existing token '{mask_token}' (id={mask_id})")

    model = build_model(cfg, len(tokenizer), context_length)
    params = load_params(checkpoint_path)

    rng = jax.random.PRNGKey(args.seed)
    rng, resize_key = jax.random.split(rng)
    params, added_rows = resize_embedding_params(params, len(tokenizer), key=resize_key)
    if added_rows:
        print(f"[checkpoint] expanded embeddings by {added_rows} rows for TiDAR mask token")

    params = jax.device_put(params)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)

    # Prefill cache with prompt
    prompt_ids_jax = jnp.asarray(prompt_ids, dtype=jnp.int32)
    cache_vars, cache_index = prefill_prompt_cache(
        model,
        params,
        cache_vars,
        prompt_ids_jax,
        kv_cache_len=cache_len,
    )

    # Build prefill bias for initial draft
    prefill_template = build_prefill_bias_template(context_length, draft_len, bias_value)
    prefill_slice = prefill_template[:, :, :draft_len, : (draft_len + prompt_len)]
    if cache_len > prompt_len:
        pad_width = cache_len - prompt_len
        pad = jnp.zeros((1, 1, draft_len, pad_width), dtype=prefill_slice.dtype)
        prefill_bias = jnp.concatenate([prefill_slice, pad], axis=-1)
    else:
        prefill_bias = prefill_slice

    prefill_draft_fn = make_prefill_draft_fn(
        model,
        cache_len=cache_len,
        prefill_bias=prefill_bias,
    )

    # Initial draft via prefill
    rng, prefill_key = jax.random.split(rng)
    prefill_masks = jnp.full((1, draft_len), mask_id, dtype=jnp.int32)
    prefill_position_ids = build_prefill_position_ids(cache_index, draft_len)

    prefill_start = time.perf_counter()
    prefill_logits = prefill_draft_fn(
        params,
        cache_vars,
        prefill_masks,
        prefill_position_ids[None, :],
        jnp.int32(cache_index),
    )
    prefill_logits.block_until_ready()
    prefill_logits = prefill_logits[0]  # (draft_len, vocab_size)
    
    # Sample initial draft tokens on device
    draft_tokens = sample_batch_from_logits_jax(
        prefill_logits, prefill_key, temperature, top_k
    )
    prefill_time = time.perf_counter() - prefill_start

    eos_id = tokenizer.eos_token_id if stop_on_eos else None

    # Choose decode function based on mode
    if args.use_while_loop:
        # Use while_loop for variable-length generation with early stopping
        decode_fn = make_full_decode_fn(
            model,
            cache_len=cache_len,
            draft_len=draft_len,
            max_steps=max_steps,
            mask_id=mask_id,
            bias_value=bias_value,
            temperature=temperature,
            top_k=top_k,
            always_accept=args.always_accept,
        )
        
        rng, decode_key = jax.random.split(rng)
        decode_start = time.perf_counter()
        
        generated_tokens, num_generated = decode_fn(
            params,
            cache_vars,
            draft_tokens,
            prefill_logits,
            prompt_ids_jax,
            cache_index,
            decode_key,
            eos_id,
        )
        generated_tokens.block_until_ready()
        decode_time = time.perf_counter() - decode_start
        
        # Clip to max_steps (buffer may be larger due to rounding)
        actual_generated = min(int(num_generated), max_steps)
        generated_tokens = np.asarray(generated_tokens[:actual_generated])
        generated = actual_generated
    else:
        # Use fori_loop for fixed iterations (simpler, faster compilation)
        if not args.always_accept:
            print("[warning] fori_loop mode requires --always_accept, enabling it")
        
        decode_fn = make_decode_loop_fn(
            model,
            cache_len=cache_len,
            draft_len=draft_len,
            max_steps=max_steps,
            mask_id=mask_id,
            bias_value=bias_value,
            temperature=temperature,
            top_k=top_k,
        )
        
        rng, decode_key = jax.random.split(rng)
        decode_start = time.perf_counter()
        
        generated_tokens = decode_fn(
            params,
            cache_vars,
            draft_tokens,
            cache_index,
            decode_key,
        )
        generated_tokens.block_until_ready()
        decode_time = time.perf_counter() - decode_start
        
        generated_tokens = np.asarray(generated_tokens)
        generated = min(len(generated_tokens), max_steps)
        generated_tokens = generated_tokens[:generated]

    # Combine prompt and generated tokens for decoding
    all_ids = np.concatenate([prompt_ids, generated_tokens], axis=0)
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    
    if stop_on_eos and eos_id is not None:
        eos_hits = np.where(all_ids == eos_id)[0]
        if eos_hits.size > 0:
            cut = int(eos_hits[0])
            text = tokenizer.decode(all_ids[:cut], skip_special_tokens=True) + "<EOS>"

    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")

    if args.verbose:
        toks_per_s = (generated / decode_time) if decode_time > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {prompt_len}")
        print(f"generated_tokens: {generated}")
        print(f"prefill_time_s: {prefill_time:.6f}")
        print(f"decode_time_s:  {decode_time:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
