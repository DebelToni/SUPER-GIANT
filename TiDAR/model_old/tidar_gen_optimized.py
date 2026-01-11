"""Fully JIT-compiled TiDAR generation."""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from model.GiantGPT import GiantGPT
from model.tidar_masks import build_tidar_decode_bias_cached
from model.tidar_utils import jax_sample, jax_rejection_sample_vectorized


@partial(jax.jit, static_argnames=('draft_len',))
def _build_position_ids_decode(prefix_len: jax.Array, draft_len: int) -> jnp.ndarray:
    """Build position_ids for decode step."""
    pos_verify = prefix_len + jnp.arange(draft_len, dtype=jnp.int32)
    r_offsets = jnp.arange(1, draft_len + 1, dtype=jnp.int32)
    local_offsets = jnp.arange(draft_len, dtype=jnp.int32)
    pos_predraft = prefix_len + r_offsets[:, None] + local_offsets[None, :]
    pos_predraft = pos_predraft.ravel()
    return jnp.concatenate([pos_verify, pos_predraft])


@partial(jax.jit, static_argnames=('model', 'draft_len', 'mask_id', 'kv_cache_len', 'temperature', 'top_k'))
def tidar_decode_step_jit(
    model: GiantGPT,
    params,
    cache_vars,
    prefix_len: jax.Array,
    verify_ids: jnp.ndarray,
    mask_id: int,
    draft_len: int,
    kv_cache_len: int,
    key: jax.Array,
    temperature: float,
    top_k: int,
    draft_logits: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, object, jax.Array, jax.Array, jax.Array]:
    """Fully JIT-compiled TiDAR decode step."""
    batch_size = verify_ids.shape[0]
    
    # Build inputs
    predraft = jnp.full((batch_size, draft_len * draft_len), mask_id, dtype=jnp.int32)
    step_tokens = jnp.concatenate([verify_ids, predraft], axis=1)
    position_ids_1d = _build_position_ids_decode(prefix_len, draft_len)
    position_ids = jnp.broadcast_to(position_ids_1d[None, :], (batch_size, position_ids_1d.shape[0]))
    
    # Get attention bias (this is JIT-compiled too)
    attn_bias = build_tidar_decode_bias_cached(context_len=kv_cache_len, draft_len=draft_len)
    
    # Forward pass
    hidden, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        step_tokens,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        cache_write_len=draft_len,
        prefix_len=prefix_len,
        cur_index=prefix_len,
        attn_bias=attn_bias,
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        return_hidden=True,
        mutable=["cache"],
    )
    cache_vars = mutated["cache"]
    
    # Get embeddings for unembedding
    embed = jnp.asarray(params["Embed_0"]["embedding"])
    
    # Verify logits
    hidden_verify = hidden[:, :draft_len, :]
    logits_verify = jnp.einsum("bld,vd->blv", hidden_verify.astype(jnp.float32), embed)
    
    # Rejection sampling (vectorized)
    verify_ids_1d = verify_ids[0]
    key, sub = jax.random.split(key)
    r, committed_full = jax_rejection_sample_vectorized(
        verify_ids_1d,
        logits_verify[0],
        key=sub,
        draft_logits=draft_logits,
        temperature=temperature,
        top_k=top_k,
    )
    
    # Select candidate logits based on r (device-side selection)
    # We'll compute all K candidates and select based on r
    def get_cand_logits(r_val):
        cand_start = draft_len + (r_val - 1) * draft_len
        hidden_cand = jax.lax.dynamic_slice(hidden, (0, cand_start, 0), (1, draft_len, hidden.shape[-1]))
        return jnp.einsum("bld,vd->blv", hidden_cand.astype(jnp.float32), embed)[0]
    
    # Use switch to select the right candidate without host sync
    logits_cand = jax.lax.switch(r - 1, [lambda i=i: get_cand_logits(i+1) for i in range(draft_len)])
    
    # Sample next verify
    key, sub = jax.random.split(key)
    next_verify = jax_sample(logits_cand, key=sub, temperature=temperature, top_k=top_k)
    
    # Update prefix_len
    new_prefix_len = prefix_len + r
    
    # Return committed tokens, next verify, new state
    return committed_full, next_verify, logits_cand, cache_vars, new_prefix_len, r, key


@partial(jax.jit, static_argnames=('model', 'draft_len', 'mask_id', 'max_steps', 'kv_cache_len', 'temperature', 'top_k'))
def tidar_generate_loop_jit(
    model: GiantGPT,
    params,
    cache_vars,
    prefix_len: jax.Array,
    verify_ids: jnp.ndarray,
    mask_id: int,
    draft_len: int,
    kv_cache_len: int,
    max_steps: int,
    key: jax.Array,
    temperature: float,
    top_k: int,
) -> Tuple[object, jax.Array, jax.Array]:
    """Fully JIT-compiled generation loop using while_loop."""
    
    def cond_fn(state):
        _, _, generated, _, _, _ = state
        return generated < max_steps
    
    def body_fn(state):
        cache_vars, prefix_len, generated, verify_ids, verify_logits, key = state
        
        # Decode step
        committed, next_verify, next_logits, new_cache, new_prefix_len, r, new_key = tidar_decode_step_jit(
            model,
            params,
            cache_vars,
            prefix_len,
            verify_ids[None, :],  # Add batch dim
            mask_id,
            draft_len,
            kv_cache_len,
            key,
            temperature,
            top_k,
            verify_logits,
        )
        
        return (new_cache, new_prefix_len, generated + r, next_verify, next_logits, new_key)
    
    # Initial state
    init_state = (cache_vars, prefix_len, jnp.array(0, jnp.int32), verify_ids, None, key)
    
    # Run loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    cache_vars, prefix_len, generated, _, _, key = final_state
    
    return cache_vars, prefix_len, generated
