from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from model.GiantGPT import GiantGPT
from model.tidar_masks import build_tidar_prefill_bias_cached, build_tidar_decode_bias_cached
from model.tidar_utils import jax_sample, jax_rejection_sample_vectorized as jax_rejection_sample


@partial(jax.jit, static_argnames=('draft_len', 'mask_id'))
def _build_decode_inputs_jit(
    verify_ids: jnp.ndarray,
    prefix_len: jax.Array,
    mask_id: int,
    draft_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build decode step inputs in JIT-compiled fashion."""
    batch_size = verify_ids.shape[0]
    
    # Predraft masks
    predraft = jnp.full((batch_size, draft_len * draft_len), mask_id, dtype=jnp.int32)
    step_tokens = jnp.concatenate([verify_ids, predraft], axis=1)
    
    # Position IDs using JIT helper
    position_ids_1d = _build_position_ids_jit(prefix_len, draft_len)
    position_ids = jnp.broadcast_to(position_ids_1d[None, :], (batch_size, position_ids_1d.shape[0]))
    
    return step_tokens, position_ids


@partial(jax.jit, static_argnames=('draft_len',))
def _build_position_ids_jit(prefix_len: jax.Array, draft_len: int) -> jnp.ndarray:
    """Build position_ids for decode step without Python loops."""
    # Verify positions: prefix_len + [0, 1, ..., K-1]
    pos_verify = prefix_len + jnp.arange(draft_len, dtype=jnp.int32)
    
    # Predraft positions: for candidate r, positions are prefix_len + r + [0..K-1]
    r_offsets = jnp.arange(1, draft_len + 1, dtype=jnp.int32)  # [1, 2, ..., K]
    local_offsets = jnp.arange(draft_len, dtype=jnp.int32)     # [0, 1, ..., K-1]
    # Outer product + broadcast
    pos_predraft = prefix_len + r_offsets[:, None] + local_offsets[None, :]
    pos_predraft = pos_predraft.ravel()  # [K*K]
    
    return jnp.concatenate([pos_verify, pos_predraft])


def _as_batch(tokens: jnp.ndarray) -> jnp.ndarray:
    if tokens.ndim == 1:
        return tokens[None, :]
    return tokens


def _get_embed_matrix(params) -> jnp.ndarray:
    embed = params.get("Embed_0") if isinstance(params, dict) else None
    if embed is None:
        raise KeyError("Embed_0 not found in params; cannot unembed logits")
    return jnp.asarray(embed["embedding"])


def _logits_from_hidden(hidden: jnp.ndarray, embed: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("bld,vd->blv", hidden.astype(jnp.float32), embed)


def init_kv_cache(model: GiantGPT, *, batch_size: int, pad_token_id: int = 0):
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(0),
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
    )
    return variables["cache"]


def prefill_prompt_cache(
    model: GiantGPT,
    params,
    cache_vars,
    prompt_ids: np.ndarray,
    *,
    kv_cache_len: Optional[int] = None,
) -> Tuple[object, int]:
    prompt_ids = np.asarray(prompt_ids, dtype=np.int32)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
    batch_size, prompt_len = prompt_ids.shape
    if prompt_len == 0:
        return cache_vars, 0
    position_ids = np.arange(prompt_len, dtype=np.int32)
    position_ids = np.broadcast_to(position_ids[None, :], (batch_size, prompt_len))

    tokens = jnp.asarray(prompt_ids)
    if kv_cache_len is None:
        kv_cache_len = prompt_len

    @jax.jit
    def _scan_prefill(params, cache_vars, tokens):
        batch_size, seq_len = tokens.shape
        idx0 = jnp.array(0, jnp.int32)

        def step(carry, tok_t):
            cache_vars, idx = carry
            tok_t = tok_t[:, None]
            pos_ids = jnp.full((batch_size, 1), idx, dtype=jnp.int32)
            _, mutated = model.apply(
                {"params": params, "cache": cache_vars},
                tok_t,
                deterministic=True,
                use_kv_cache=True,
                write_to_cache=True,
                cur_index=idx,
                position_ids=pos_ids,
                kv_cache_len=kv_cache_len,
                mutable=["cache"],
            )
            return (mutated["cache"], idx + 1), None

        tokens_t = jnp.swapaxes(tokens, 0, 1)
        (cache_vars, idx), _ = jax.lax.scan(step, (cache_vars, idx0), tokens_t, length=seq_len)
        return cache_vars, idx

    cache_vars, idx = _scan_prefill(params, cache_vars, tokens)
    return cache_vars, int(jax.device_get(idx))


def tidar_prefill_draft_cached(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    prefix_len: int,
    context_len: int,
    batch_size: int,
    mask_id: int,
    draft_len: int,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
    kv_cache_len: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    masks = jnp.full((batch_size, draft_len), mask_id, dtype=jnp.int32)
    position_ids = jnp.arange(prefix_len, prefix_len + draft_len, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, draft_len))

    if kv_cache_len is None:
        kv_cache_len = context_len
    attn_bias = build_tidar_prefill_bias_cached(context_len=kv_cache_len, draft_len=draft_len)
    hidden = model.apply(
        {"params": params, "cache": cache_vars},
        masks,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        prefix_len=prefix_len,
        attn_bias=jnp.asarray(attn_bias),
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        return_hidden=True,
    )

    embed = _get_embed_matrix(params)
    logits = _logits_from_hidden(hidden, embed)
    logits = logits[:, :draft_len, :]
    key, sub = jax.random.split(key)
    draft_tokens = jax_sample(logits[0], key=sub, temperature=temperature, top_k=top_k)
    return draft_tokens, logits[0], key


def _build_decode_step_inputs_cached(
    verify_ids: jnp.ndarray,
    *,
    mask_id: int,
    draft_len: int,
    prefix_len: int,
    context_len: int,
    kv_cache_len: Optional[int] = None,
):
    verify_ids = _as_batch(verify_ids)
    batch_size = verify_ids.shape[0]

    predraft = jnp.full((batch_size, draft_len * draft_len), mask_id, dtype=jnp.int32)
    step_tokens = jnp.concatenate([verify_ids, predraft], axis=1)

    # Use JIT-compiled position builder
    prefix_len_jax = jnp.asarray(prefix_len, dtype=jnp.int32)
    position_ids_1d = _build_position_ids_jit(prefix_len_jax, draft_len)
    position_ids = jnp.broadcast_to(position_ids_1d[None, :], (batch_size, position_ids_1d.shape[0]))

    if kv_cache_len is None:
        kv_cache_len = context_len
    attn_bias = build_tidar_decode_bias_cached(context_len=kv_cache_len, draft_len=draft_len)
    return step_tokens, position_ids, attn_bias


def commit_tokens_to_cache(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    committed: jnp.ndarray,
    prefix_len: int,
) -> Tuple[object, int]:
    committed = jnp.asarray(committed, dtype=jnp.int32)
    if committed.size == 0:
        return cache_vars, prefix_len
    tokens = _as_batch(committed)
    batch_size, step_len = tokens.shape
    position_ids = jnp.arange(prefix_len, prefix_len + step_len, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, step_len))

    _, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        tokens,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=prefix_len,
        position_ids=position_ids,
        mutable=["cache"],
    )
    return mutated["cache"], prefix_len + step_len


def tidar_decode_step_cached(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    prefix_len: int,
    context_len: int,
    prefix_ids: np.ndarray,
    verify_ids: jnp.ndarray,
    mask_id: int,
    draft_len: int,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
    draft_logits: Optional[jnp.ndarray] = None,
    always_accept: bool = False,
    max_commit: Optional[int] = None,
    return_prefix: bool = True,
    kv_cache_len: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], jnp.ndarray, jnp.ndarray, int, object, int, jax.Array]:
    step_tokens, position_ids, attn_bias = _build_decode_step_inputs_cached(
        verify_ids,
        mask_id=mask_id,
        draft_len=draft_len,
        prefix_len=prefix_len,
        context_len=context_len,
        kv_cache_len=kv_cache_len,
    )

    hidden, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        step_tokens,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        cache_write_len=draft_len,
        prefix_len=prefix_len,
        cur_index=prefix_len,
        attn_bias=jnp.asarray(attn_bias),
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        return_hidden=True,
        mutable=["cache"],
    )
    cache_vars = mutated["cache"]

    embed = _get_embed_matrix(params)
    hidden_verify = hidden[:, :draft_len, :]
    logits_verify = _logits_from_hidden(hidden_verify, embed)

    verify_ids_1d = verify_ids[0] if verify_ids.ndim == 2 else verify_ids
    if always_accept:
        if max_commit is None:
            r = draft_len
        else:
            r = int(min(draft_len, max_commit))
            if r < 1:
                raise ValueError("max_commit must be >= 1 when always_accept=True")
        committed = verify_ids_1d[:r]
    else:
        key, sub = jax.random.split(key)
        r_dev, committed_full = jax_rejection_sample(
            verify_ids_1d,
            logits_verify[0],
            key=sub,
            draft_logits=draft_logits,
            temperature=temperature,
            top_k=top_k,
        )
        r = int(jax.device_get(r_dev))
        committed = committed_full[:r]

    cand_start = draft_len + (r - 1) * draft_len
    hidden_cand = jax.lax.dynamic_slice(hidden, (0, cand_start, 0), (1, draft_len, hidden.shape[-1]))
    logits_cand = _logits_from_hidden(hidden_cand, embed)

    key, sub = jax.random.split(key)
    next_verify = jax_sample(logits_cand[0], key=sub, temperature=temperature, top_k=top_k)

    prefix_len = prefix_len + r

    if return_prefix:
        committed_host = np.array(jax.device_get(committed))
        new_prefix = np.concatenate([np.asarray(prefix_ids, dtype=np.int32), committed_host], axis=0)
    else:
        new_prefix = None
    return (
        None if new_prefix is None else new_prefix.astype(np.int32),
        next_verify,
        logits_cand[0],
        r,
        cache_vars,
        prefix_len,
        key,
    )
