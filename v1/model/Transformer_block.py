
"""
Self‑contained TinyTransformerBlock **with KV cache support**.

We use Flax’s `cache` collection: during the first forward pass the cache is
created with length 0; every subsequent token concatenates to the existing
keys/values and writes them back.

Nothing outside the module needs to know about the implementation details.
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax, jax.numpy as jnp
from flax import linen as nn
import Config
import functools

def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    b, s, d = x.shape
    h        = num_heads
    head_dim = d // h
    return x.reshape(b, s, h, head_dim).transpose(0, 2, 1, 3)   # (b, h, s, d/h)

def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    b, h, s, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, s, h * d)

class NativeJaxSelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype    = Config.compute_dtype
    use_cache: bool     = False

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 deterministic: bool = True
                 ) -> jnp.ndarray:
        b, s, _ = x.shape
        head_dim = self.qkv_features // self.num_heads

        # dense = functools.partial(
        #     nn.Dense,
        #     features=self.qkv_features,
        #     use_bias=False,
        #     dtype=self.dtype,
        #     param_dtype=Config.param_dtype,
        #     kernel_init=nn.initializers.normal(stddev=0.02),
        # )
        #
        # q = dense(name="q_proj")(x)
        # k = dense(name="k_proj")(x)
        # v = dense(name="v_proj")(x)
        q = nn.Dense(self.qkv_features,
                    use_bias=False,
                    dtype=self.dtype,
                    param_dtype=Config.param_dtype,
                    name="q_proj",
                    kernel_init=nn.initializers.normal(stddev=0.02))
        k = nn.Dense(self.qkv_features,
                    use_bias=False,
                    dtype=self.dtype,
                    param_dtype=Config.param_dtype,
                    name="k_proj",
                    kernel_init=nn.initializers.normal(stddev=0.02))(x)
        v = nn.Dense(self.qkv_features,
                    use_bias=False,
                    dtype=self.dtype,
                    param_dtype=Config.param_dtype,
                    name="v_proj",
                    kernel_init=nn.initializers.normal(stddev=0.02))(x)

        q = _split_heads(q, self.num_heads)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)

        # -------- KV cache ----------------------------------------------------
        if self.use_cache:
            # create or fetch the cache variables
            cache_k = self.variable("cache", "k",
                                    lambda: jnp.zeros((b, self.num_heads, 0, head_dim),
                                                      self.dtype))
            cache_v = self.variable("cache", "v",
                                    lambda: jnp.zeros((b, self.num_heads, 0, head_dim),
                                                      self.dtype))

            k = jnp.concatenate([cache_k.value, k], axis=2)
            v = jnp.concatenate([cache_v.value, v], axis=2)

            # write updated tensors back
            cache_k.value = k
            cache_v.value = v

        # -------- Flash Attention via cuDNN -----------------------------------
        # NB: `jax.nn.attention` transparently calls the cuDNN flash‑Attn kernel
        # when the shapes are supported (sm80+, bf16/fp16).
        attn_out = jax.nn.attention.dot_product_attention(
            query=q, key=k, value=v,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        attn_out = _merge_heads(attn_out)                                  # (b, s, d)
        attn_out = nn.Dense(self.qkv_features,
                            use_bias=False,
                            dtype=self.dtype,
                            param_dtype=Config.param_dtype,
                            name="o_proj")(attn_out)
        return attn_out.astype(self.dtype)

class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype    = Config.compute_dtype
    use_cache: bool     = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        # LayerNorm (f32)
        residual = x
        h = nn.LayerNorm(dtype=jnp.float32, name="ln1")(x)
        h = NativeJaxSelfAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                use_cache=self.use_cache,
                name="mha")(h, deterministic=deterministic)
        h = residual + h

        # Feed‑forward
        residual = h
        h = nn.LayerNorm(dtype=jnp.float32, name="ln2")(h)
        h = nn.Dense(self.d_ff,
                     dtype=self.dtype,
                     param_dtype=Config.param_dtype,
                     name="fc1")(h)
        h = nn.gelu(h, approximate=False)
        h = nn.Dense(self.d_model,
                     dtype=self.dtype,
                     param_dtype=Config.param_dtype,
                     name="fc2")(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return residual + h
