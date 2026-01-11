from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.tidar_utils import build_train_batch


def _attention_weights(q, k, bias):
    head_dim = q.shape[-1]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
    logits = logits + bias
    return jax.nn.softmax(logits, axis=-1)


def smoke_train_padding_mask() -> None:
    tokens = jnp.array([[10, 11, 12, 13]], dtype=jnp.int32)
    lengths = jnp.array([2], dtype=jnp.int32)
    batch = build_train_batch(tokens, lengths, mask_id=0, block_len=2)

    bias = batch["attn_bias"]
    q_len = bias.shape[-2]
    k_len = bias.shape[-1]

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, q_len, 1, 4))
    k = jax.random.normal(key, (1, k_len, 1, 4))
    v = jax.random.normal(key, (1, k_len, 1, 4))

    _ = jax.nn.dot_product_attention(q, k, v, bias=bias, is_causal=False)
    weights = _attention_weights(q, k, bias)

    pos_ids = batch["position_ids"][0]
    valid = pos_ids < lengths[0]
    invalid_idx = jnp.where(~valid)[0]
    max_invalid = weights[0, 0, :, invalid_idx].max()
    print(f"[smoke] max attention on padded keys: {float(max_invalid):.6f}")
    if max_invalid > 1e-5:
        raise AssertionError("Padding keys received non-trivial attention weight.")


if __name__ == "__main__":
    smoke_train_padding_mask()
    print("[smoke] OK")
