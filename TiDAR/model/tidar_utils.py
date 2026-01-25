from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from TiDAR.model.tidar_masks import build_tidar_train_bias

IGNORE_INDEX = -100


def build_train_batch(
    tokens: jnp.ndarray,
    lengths: Optional[jnp.ndarray],
    *,
    mask_id: int,
    block_len: int,
    bias_value: float = -1.0e10,
    ignore_index: int = IGNORE_INDEX,
    token_mask: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Build TiDAR training inputs for a batch of token sequences."""
    batch_size, seq_len = tokens.shape
    clean = tokens
    diff = jnp.full_like(clean, mask_id)

    input_ids = jnp.concatenate([clean, diff], axis=1)

    pos = jnp.arange(seq_len, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(pos[None, :], (batch_size, seq_len))
    position_ids = jnp.concatenate([position_ids, position_ids], axis=1)

    labels = jnp.full((batch_size, seq_len * 2), ignore_index, dtype=jnp.int32)
    labels = labels.at[:, : seq_len - 1].set(clean[:, 1:])
    labels = labels.at[:, seq_len:].set(clean)

    loss_mask_ntp = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)
    loss_mask_diff = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)

    if lengths is None:
        valid = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    else:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        valid = (positions < lengths[:, None]).astype(jnp.float32)
    if token_mask is not None:
        valid = valid * token_mask.astype(jnp.float32)

    # Ensure no batch row has all-zero mask (prevents NaN in attention)
    # If a row has zero valid tokens, set mask to all 1.0s as fallback
    row_has_tokens = valid.sum(axis=1, keepdims=True) > 0  # (batch_size, 1)
    valid = jnp.where(row_has_tokens, valid, jnp.ones_like(valid))

    if seq_len > 1:
        loss_mask_ntp = loss_mask_ntp.at[:, : seq_len - 1].set(valid[:, 1:])
    loss_mask_diff = loss_mask_diff.at[:, seq_len:].set(valid)

    token_types = jnp.concatenate(
        [jnp.zeros(seq_len, dtype=jnp.int32), jnp.ones(seq_len, dtype=jnp.int32)]
    )
    key_padding_mask = jnp.concatenate([valid, valid], axis=1) > 0
    attn_bias = build_tidar_train_bias(
        position_ids,
        token_types,
        block_len=block_len,
        key_padding_mask=key_padding_mask,
        bias_value=bias_value,
    )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask_ntp": loss_mask_ntp,
        "loss_mask_diff": loss_mask_diff,
        "attn_bias": attn_bias,
    }


__all__ = ["build_train_batch", "IGNORE_INDEX"]
