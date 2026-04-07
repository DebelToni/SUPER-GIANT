from __future__ import annotations

import jax.numpy as jnp


def build_answer_hidden_bias(
    loss_mask,
    *,
    enabled: bool,
    causal: bool,
    dtype,
):
    if not enabled or causal:
        return None

    mask = jnp.asarray(loss_mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected loss_mask to have shape [batch, seq], got {mask.shape}")

    seq_len = mask.shape[1]
    mask_int = mask.astype(jnp.int32)
    has_single_target = jnp.sum(mask_int, axis=1) == 1
    answer_key = jnp.clip(jnp.argmax(mask_int, axis=1) + 1, 0, seq_len - 1)
    key_positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    blocked = (key_positions == answer_key[:, None]) & has_single_target[:, None]

    zero = jnp.asarray(0.0, dtype=dtype)
    neg = jnp.asarray(-1e10, dtype=dtype)
    blocked = blocked[:, None, None, :]
    blocked = jnp.broadcast_to(blocked, (mask.shape[0], 1, seq_len, seq_len))
    return jnp.where(blocked, neg, zero)


__all__ = ["build_answer_hidden_bias"]
