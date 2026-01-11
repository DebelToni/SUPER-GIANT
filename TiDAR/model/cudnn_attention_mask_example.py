from __future__ import annotations

import jax
import jax.numpy as jnp


def cudnn_bias_from_mask(mask: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """Convert a boolean mask to a cuDNN attention bias.

    `mask` expects True for allowed positions and False for masked positions.
    The output bias should be passed to jax.nn.dot_product_attention.
    """
    bias_value = jnp.finfo(dtype).min
    return jnp.where(mask, jnp.array(0.0, dtype=dtype), jnp.array(bias_value, dtype=dtype))


def key_valid_bias(prefix_len: jnp.ndarray, key_len: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Bias for KV-cache style masking over keys only.

    Returns shape (1, 1, 1, key_len) so it can broadcast across batch/heads/queries.
    """
    key_idx = jnp.arange(key_len)
    valid = key_idx < prefix_len
    bias = jnp.where(valid, 0.0, jnp.finfo(dtype).min).astype(dtype)
    return bias[None, None, None, :]


def causal_bias(seq_len: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Causal bias with shape (1, 1, seq_len, seq_len)."""
    row = jnp.arange(seq_len)[:, None]
    col = jnp.arange(seq_len)[None, :]
    allow = row >= col
    return cudnn_bias_from_mask(allow[None, None, :, :], dtype)


def demo_full_sequence():
    key = jax.random.PRNGKey(0)
    batch, seq_len, heads, head_dim = 1, 128, 8, 64
    dtype = jnp.bfloat16

    q = jax.random.normal(key, (batch, seq_len, heads, head_dim), dtype=dtype)
    k = jax.random.normal(key, (batch, seq_len, heads, head_dim), dtype=dtype)
    v = jax.random.normal(key, (batch, seq_len, heads, head_dim), dtype=dtype)

    bias = causal_bias(seq_len, dtype)
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        is_causal=False,
        implementation="cudnn",
    )


def demo_kv_cache():
    key = jax.random.PRNGKey(1)
    batch, query_len, key_len, heads, head_dim = 1, 128, 256, 8, 64
    dtype = jnp.bfloat16

    q = jax.random.normal(key, (batch, query_len, heads, head_dim), dtype=dtype)
    k = jax.random.normal(key, (batch, key_len, heads, head_dim), dtype=dtype)
    v = jax.random.normal(key, (batch, key_len, heads, head_dim), dtype=dtype)

    prefix_len = jnp.array(192, dtype=jnp.int32)
    bias = key_valid_bias(prefix_len, key_len, dtype)
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        is_causal=False,
        implementation="cudnn",
    )


if __name__ == "__main__":
    demo_full_sequence()
    demo_kv_cache()
