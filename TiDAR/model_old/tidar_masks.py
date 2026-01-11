from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp


def _ensure_token_types(token_types: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    if token_types.ndim == 1:
        token_types = jnp.broadcast_to(token_types[None, :], (batch_size, token_types.shape[0]))
    return token_types


def build_tidar_train_bias(
    position_ids: jnp.ndarray,
    token_types: jnp.ndarray,
    *,
    block_len: int,
    key_padding_mask: Optional[jnp.ndarray] = None,
    bias_value: float = -1.0e10,
) -> jnp.ndarray:
    """Build TiDAR training attention bias (clean+diffusion)."""
    batch_size, seq_len = position_ids.shape
    token_types = _ensure_token_types(token_types, batch_size)

    pos_q = position_ids[:, :, None]
    pos_k = position_ids[:, None, :]
    t_q = token_types[:, :, None]
    t_k = token_types[:, None, :]

    block_q = pos_q // block_len
    block_k = pos_k // block_len
    block_start = block_q * block_len

    is_clean_q = t_q == 0
    is_clean_k = t_k == 0
    is_diff_q = t_q == 1
    is_diff_k = t_k == 1

    allow_clean_to_clean = is_clean_q & is_clean_k & (pos_k <= pos_q)
    allow_diff_to_diff = is_diff_q & is_diff_k & (block_k <= block_q)
    allow_diff_to_clean = is_diff_q & is_clean_k & (pos_k < block_start)

    allow = allow_clean_to_clean | allow_diff_to_diff | allow_diff_to_clean

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.astype(bool)
        if key_padding_mask.ndim == 1:
            key_padding_mask = jnp.broadcast_to(key_padding_mask[None, :], (batch_size, key_padding_mask.shape[0]))
        allow = allow & key_padding_mask[:, None, :]
    bias = jnp.where(allow, 0.0, bias_value)
    bias = bias[:, None, :, :]
    return bias


def build_tidar_prefill_bias(
    *,
    prompt_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
) -> jnp.ndarray:
    """Bias for prefill layout: [mask*K, prompt*L]."""
    total = draft_len + prompt_len
    idx = jnp.arange(total)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    is_mask_q = q_idx < draft_len
    is_mask_k = k_idx < draft_len
    is_prompt_q = q_idx >= draft_len
    is_prompt_k = k_idx >= draft_len

    allow_mask_to_mask = is_mask_q & is_mask_k
    allow_mask_to_prompt = is_mask_q & is_prompt_k

    prompt_q_pos = q_idx - draft_len
    prompt_k_pos = k_idx - draft_len
    allow_prompt_to_prompt = is_prompt_q & is_prompt_k & (prompt_k_pos <= prompt_q_pos)

    allow = allow_mask_to_mask | allow_mask_to_prompt | allow_prompt_to_prompt
    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


@partial(jax.jit, static_argnames=('context_len', 'draft_len', 'dtype'))
def build_tidar_prefill_bias_cached(
    *,
    context_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Bias for cached prefill: queries are masks, keys are prefix + masks."""
    total = context_len + draft_len
    _ = bias_value
    return jnp.zeros((1, 1, draft_len, total), dtype=dtype)


def build_tidar_decode_bias(
    *,
    prefix_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
) -> jnp.ndarray:
    """Bias for decode layout: [prefix, verify, cand_blocks]."""
    total = prefix_len + draft_len + (draft_len * draft_len)
    idx = jnp.arange(total)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    prefix_end = prefix_len
    verify_start = prefix_end
    verify_end = verify_start + draft_len
    cand_start = verify_end

    is_prefix_q = q_idx < prefix_end
    is_prefix_k = k_idx < prefix_end
    is_verify_q = (q_idx >= verify_start) & (q_idx < verify_end)
    is_verify_k = (k_idx >= verify_start) & (k_idx < verify_end)
    is_cand_q = q_idx >= cand_start
    is_cand_k = k_idx >= cand_start

    prefix_q_pos = q_idx
    prefix_k_pos = k_idx
    allow_prefix = is_prefix_q & is_prefix_k & (prefix_k_pos <= prefix_q_pos)

    verify_q_pos = q_idx - verify_start
    verify_k_pos = k_idx - verify_start
    allow_verify_to_prefix = is_verify_q & is_prefix_k
    allow_verify_to_verify = is_verify_q & is_verify_k & (verify_k_pos <= verify_q_pos)

    cand_q_offset = q_idx - cand_start
    cand_k_offset = k_idx - cand_start
    cand_q_block = cand_q_offset // draft_len
    cand_k_block = cand_k_offset // draft_len

    cand_r = cand_q_block + 1
    allow_cand_to_prefix = is_cand_q & is_prefix_k
    allow_cand_to_verify = is_cand_q & is_verify_k & (verify_k_pos < cand_r)
    allow_cand_to_cand = is_cand_q & is_cand_k & (cand_q_block == cand_k_block)

    allow = (
        allow_prefix
        | allow_verify_to_prefix
        | allow_verify_to_verify
        | allow_cand_to_prefix
        | allow_cand_to_verify
        | allow_cand_to_cand
    )

    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


@partial(jax.jit, static_argnames=('context_len', 'draft_len', 'dtype'))
def build_tidar_decode_bias_cached(
    *,
    context_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Bias for cached decode: queries are verify+candidate, keys include prefix."""
    step_len = draft_len + (draft_len * draft_len)
    key_len = context_len + step_len

    q_idx = jnp.arange(step_len)[:, None]
    k_idx = jnp.arange(key_len)[None, :]

    is_verify_q = q_idx < draft_len
    is_cand_q = q_idx >= draft_len
    is_prefix_k = k_idx < context_len
    is_step_k = k_idx >= context_len

    step_k_idx = k_idx - context_len
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

    bias = jnp.where(allow, jnp.zeros((), dtype=dtype), jnp.full((), bias_value, dtype=dtype))
    return bias[None, None, :, :]
