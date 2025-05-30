from __future__ import annotations
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

import Config                                

cfg = Config                                 

def sinusoid_position_encoding(length: int, d_model: int, dtype=jnp.float32):
    pos = jnp.arange(length)[:, None]
    i   = jnp.arange(d_model)[None, :]
    angle_rates = 1.0 / jnp.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = jnp.where(i % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return pe.astype(dtype)

class FeedForward(nn.Module):
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.gelu(nn.Dense(self.d_ff, dtype=cfg.compute_dtype)(x))
        h = nn.Dense(x.shape[-1], dtype=cfg.compute_dtype)(h)
        return h

class TransformerBlock(nn.Module):
    """
    One decoder block that **updates** the mutable KV cache in-place.

    Args
    ----
    d_model : hidden width
    n_heads : number of attention heads
    d_ff    : hidden size of the feed-forward sub-layer
    dropout : unused in inference, retained for training compatibility
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0

    def _split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """[B, T, D] → [B, H, T, Dh]"""
        B, T, _ = x.shape
        Dh = self.d_model // self.n_heads
        x = x.reshape(B, T, self.n_heads, Dh)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """[B, H, T, Dh] → [B, T, D]"""
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    @staticmethod
    def _flash_attention(q, k, v) -> jnp.ndarray:
        """
        Wrapper that **prefers cuDNN Flash-Attention v9** (≥ CUDA 12.1)
        and silently falls back to the dense softmax path when unavailable.
        """
        try:
            import jax.experimental.cuda
            return jax.experimental.cuda.flash_attention(q, k, v)
        except (ImportError, AttributeError):

            scale = 1.0 / jnp.sqrt(q.shape[-1]).astype(q.dtype)
            attn = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            return jnp.einsum("...hqk,...khd->...qhd", attn, v)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                      
        *,
        cache: Dict[str, jnp.ndarray],       
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        B, T_new, _ = x.shape               

        h = nn.LayerNorm(dtype=cfg.compute_dtype)(x)

        qkv = nn.Dense(3 * self.d_model,
                       use_bias=False,
                       dtype=cfg.compute_dtype,
                       name="qkv")(h)

        q, k, v = jnp.split(qkv, 3, axis=-1)      
        q = self._split_heads(q)                  
        k = self._split_heads(k)
        v = self._split_heads(v)

        idx = cache["idx"]
        cache["k"] = cache["k"].at[:, :, idx, :].set(k.squeeze(2))
        cache["v"] = cache["v"].at[:, :, idx, :].set(v.squeeze(2))
        cache["idx"] = idx + 1

        k_full = cache["k"][:, :, : idx + 1, :]
        v_full = cache["v"][:, :, : idx + 1, :]

        q = q.astype(cfg.compute_dtype)
        k_full = k_full.astype(cfg.compute_dtype)
        v_full = v_full.astype(cfg.compute_dtype)

        attn_out = self._flash_attention(q, k_full, v_full)     
        attn_out = self._merge_heads(attn_out)                  

        out = nn.Dense(self.d_model, dtype=cfg.compute_dtype,
                       name="proj")(attn_out)
        if not deterministic and self.dropout > 0.0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        x = x + out
        h = nn.LayerNorm(dtype=cfg.compute_dtype)(x)
        h_ff = FeedForward(self.d_ff, name="ffn")(h)
        if not deterministic and self.dropout > 0.0:
            h_ff = nn.Dropout(self.dropout)(h_ff,
                                            deterministic=deterministic)
        y = x + h_ff
        return y, cache
