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
        h = nn.gelu(nn.Dense(self.d_ff, dtype=cfg.compute_dtype,
                                param_dtype=cfg.param_dtype,#name="ffn_dense"   
                             )(x))
        h = nn.Dense(x.shape[-1], dtype=cfg.compute_dtype
                                , param_dtype=cfg.param_dtype, #name="ffn_proj" 
                     )(h)
        return h

class TransformerBlock(nn.Module):
    d_model : int
    n_heads : int
    d_ff    : int
    dropout : float = 0.0          # kept for training only

    # ---------- helpers --------------------------------------------------
    def _split_heads(self, x):
        B, T, _ = x.shape
        H , Dh  = self.n_heads, self.d_model // self.n_heads
        return x.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)


    @staticmethod
    def _flash_attention(q, k, v) -> jnp.ndarray:
        """
        Wrapper that **prefers cuDNN Flash-Attention v9** (≥ CUDA 12.1)
        and silently falls back to the dense softmax path when unavailable.
        """
        try:
            # import jax.experimental.cuda
            # return jax.experimental.cuda.flash_attention(q, k, v)
            from jax.lax import flash_attention
            return flash_attention(q, k, v, dropout_rate=0.0)
        except (ImportError):
            from jax.experimental.cuda import flash_attention
        # except (ImportError, AttributeError):
        #
        #     scale = 1.0 / jnp.sqrt(q.shape[-1]).astype(q.dtype)
        #     attn = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale
        #     attn = jax.nn.softmax(attn, axis=-1)
        #     return jnp.einsum("...hqk,...khd->...qhd", attn, v)

    @nn.compact
    def __call__(
        self,
        x,                      # [B, T, D]  (T=1 in inference)
        *,
        cache = None,
        deterministic = True
        ):
        B, T, _ = x.shape
        assert (cache is None) or (T == 1), \
            "While using a cache the block must be called with a single token"

        h   = nn.LayerNorm(dtype=cfg.compute_dtype)(x)
        qkv = nn.Dense(
            3 * self.d_model,
            dtype       = cfg.compute_dtype,
            param_dtype = cfg.param_dtype,
            name="qkv",
            use_bias=False,
        )(h)

        q, k, v = jnp.split(qkv, 3, axis=-1)        # [B,T,D] x3
        q, k, v = map(self._split_heads, (q, k, v)) # [B,H,T,Dh] x3

        # ── branch 1: *training*  (no cache, full causal Flash-Attn) ─────
        if cache is None:
            attn_out = jax.lax.flash_attention(
                q, k, v,
                causal       = True,                # <<< IMPORTANT
                dropout_rate = self.dropout if not deterministic else 0.0,
            )

        # ── branch 2: *decoding*  (cache is updated in-place) ────────────
        else:
            idx     = cache["idx"]                                   # scalar
            # write the new K,V at position `idx`
            cache["k"]  = cache["k"].at[:, :, idx, :].set(k.squeeze(2))
            cache["v"]  = cache["v"].at[:, :, idx, :].set(v.squeeze(2))
            cache["idx"]= idx + 1

            # slice [0 : idx+1] – everything seen so far
            k_full = cache["k"][:, :, : idx + 1, :]
            v_full = cache["v"][:, :, : idx + 1, :]

            # q-length == 1 ⇒ causal mask unnecessary
            attn_out = jax.lax.flash_attention(q, k_full, v_full, causal=False)

        # ── output projection + residual ─────────────────────────────────
        attn_out = self._merge_heads(attn_out)                       # [B,T,D]
        o        = nn.Dense(
            self.d_model,
            dtype       = cfg.compute_dtype,
            param_dtype = cfg.param_dtype,
            name="proj",
            use_bias=False,
        )(attn_out)

        o = nn.Dropout(self.dropout)(o, deterministic=deterministic)
        x = x + o

        # ── MLP ──────────────────────────────────────────────────────────
        h    = nn.LayerNorm(dtype=cfg.compute_dtype)(x)
        h_ff = nn.Dense(self.d_ff,  name="fc1")(h)
        h_ff = nn.gelu(h_ff, approximate=False)
        h_ff = nn.Dense(self.d_model, name="fc2")(h_ff)
        h_ff = nn.Dropout(self.dropout)(h_ff, deterministic=deterministic)

        y = x + h_ff
        return y, cache
    # ) -> Tuple[jnp.ndarray, Dict[str, Any] | None]:
    #     """
    #     • Training (`cache is None`):  full-sequence flash-attention.
    #     • Inference (dict):            KV-cached, one-token step.
    #     """
    #     B, T_new, _ = x.shape
    #
    #     h = nn.LayerNorm(dtype=cfg.compute_dtype)(x)
    #
    #     qkv = nn.Dense(3 * self.d_model,
    #                    use_bias=False,
    #                    dtype=cfg.compute_dtype,
    #                    param_dtype=cfg.param_dtype,
    #                    name="qkv")(h)
    #     q, k, v = jnp.split(qkv, 3, axis=-1)      # each [B, T, D]
    #     q = self._split_heads(q)                  # [B, H, T, Dh]
    #     k = self._split_heads(k)
    #     v = self._split_heads(v)
    #
    #     if cache is None:                       # ─── training path ───
    #         k_full, v_full = k, v               # whole sequence
    #     else:                                   # ─── inference path ─
    #         idx = cache["idx"]
    #         cache["k"] = cache["k"].at[:, :, idx, :].set(k.squeeze(2))
    #         cache["v"] = cache["v"].at[:, :, idx, :].set(v.squeeze(2))
    #         cache["idx"] = idx + 1
    #
    #         k_full = cache["k"][:, :, : idx + 1, :]
    #         v_full = cache["v"][:, :, : idx + 1, :]
    #
    #     q = q.astype(cfg.compute_dtype)
    #     k_full = k_full.astype(cfg.compute_dtype)
    #     v_full = v_full.astype(cfg.compute_dtype)
    #
    #     attn_out = self._flash_attention(q, k_full, v_full)     
    #     attn_out = self._merge_heads(attn_out)                  
    #
    #     out = nn.Dense(self.d_model, dtype=cfg.compute_dtype, 
    #                     param_dtype=cfg.param_dtype,
    #                    name="proj")(attn_out)
    #     if not deterministic and self.dropout > 0.0:
    #         out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
    #
    #     x = x + out
    #     h = nn.LayerNorm(dtype=cfg.compute_dtype)(x)
    #     h_ff = FeedForward(self.d_ff, name="ffn")(h)
    #     if not deterministic and self.dropout > 0.0:
    #         h_ff = nn.Dropout(self.dropout)(h_ff,
    #                                         deterministic=deterministic)
    #     y = x + h_ff
    #     return y, cache
    #
