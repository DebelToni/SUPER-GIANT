from __future__ import annotations

from typing import Optional

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
    """Build TiDAR training attention bias (clean + diffusion)."""
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
    return bias[:, None, :, :]


__all__ = ["build_tidar_train_bias"]
