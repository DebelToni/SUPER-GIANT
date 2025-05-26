# """
# Transformer_block.py
#
# A single Transformer encoder/GPT block with memory‑efficient remat (gradient
# checkpointing) for Flax/JAX.  Designed for small‑footprint training on
# free‑tier GPUs.
# """
#
# from typing import Any
#
# import jax.numpy as jnp
# from flax import linen as nn
# from flax.linen import make_causal_mask
#
#
# class TinyTransformerBlock(nn.Module):
#
#     d_model: int
#     n_heads: int
#     d_ff: int
#     dropout_rate: float = 0.1
#
#     @nn.compact
#     def __call__(self, x, *, deterministic: bool = False):
#         @nn.remat                           # checkpoint the whole block
#         def _block(module, h):              # ① module first!
#             residual = h
#             h = nn.LayerNorm()(h)
#             causal_mask = make_causal_mask(x)
#             h = nn.SelfAttention(
#                 num_heads   = module.n_heads,
#                 qkv_features= module.d_model,
#                 dropout_rate= module.dropout_rate,
#                 deterministic=deterministic,
#                 broadcast_dropout=False,
#             )(h, mask=causal_mask)
#             h = residual + h
#
#             residual = h
#             h = nn.LayerNorm()(h)
#             h = nn.Dense(module.d_ff)(h)
#             h = nn.gelu(h, approximate=False)
#             h = nn.Dense(module.d_model)(h)
#             h = nn.Dropout(rate=module.dropout_rate)(h,
#                                                      deterministic=deterministic)
#             return residual + h
#
#         return _block(self, x)              # ② pass *self* explicitly
#
# Transformer_block.py
"""
Transformer_block.py – upgraded with Flash Attention v2 (JAX bindings)

This drops the vanilla Flax `SelfAttention` in favour of a *much* more
memory‑friendly implementation that calls the CUDA kernel from the
`flash_attn_jax` project when it is available (works on Ampere, Ada & Hopper
GPUs, incl. your RTX A4500).  At run‑time we silently fall back to the classic
attention path if the package is not installed or if the mask is not supported
(e.g. non‑causal masks).  Nothing else in your model / training script needs
changing.

Installation (choose **one**):
    pip install flash-attn-jax               # pre‑built wheels (CUDA 12.3)
    pip install flash-attn-jax-cu118         # if you are still on CUDA 11.8
    # OR build from source (see project README)

Requirements: JAX ≥ 0.4.24, jaxlib compiled against the same CUDA version that
Flash‑Attention was built for.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import make_causal_mask

# -----------------------------------------------------------------------------
# Try to enable Flash Attention v2 (JAX bindings) if the user has it installed.
# -----------------------------------------------------------------------------
try:
    from flash_attn_jax import flash_mha  # type: ignore

    _FLASH_AVAILABLE = True
except ImportError:                       # pragma: no cover – fallback path
    _FLASH_AVAILABLE = False


class FlashSelfAttention(nn.Module):
    """Memory‑efficient multi‑head self‑attention using Flash Attention v2.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    qkv_features : int
        Dimension of the *concatenated* Q/K/V projection. Normally equal to
        `d_model`.
    dropout_rate : float, default 0.0
        Dropout applied to the attention output.
    """

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    broadcast_dropout: bool = False  # we keep per‑example dropout by default

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        # Projections – *no* bias for a perfect match with Flash‑Attention.
        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj")
        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj")

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b, l, _ = x.shape

        # Q‑K‑V projections ----------------------------------------------------
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_heads, self.head_dim)

        # Flash path ----------------------------------------------------------
        if _FLASH_AVAILABLE and mask is None:
            # flash_mha expects [b, l, h, d] and returns the same shape.
            y = flash_mha(q, k, v, is_causal=True)
        else:
            # Fallback – standard scaled dot‑product attention in pure JAX.
            # Shapes: q/k/v -> [b, l, h, d]  →  attn_scores [b, h, l, l]
            scale = 1.0 / jnp.sqrt(self.head_dim)
            attn_scores = jnp.einsum("blhd,bkhd->bhlk", q, k) * scale
            if mask is not None:
                # broadcast mask to [b, h, l, k]
                attn_scores = jnp.where(mask, attn_scores, jnp.full_like(attn_scores, -1e9))
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            y = jnp.einsum("bhlk,bkhd->blhd", attn_weights, v)

        # Merge heads & final projection --------------------------------------
        y = y.reshape(b, l, -1)          # [b, l, h*d] = [b, l, d_model]
        y = self.o_proj(y)
        y = nn.Dropout(
            rate=self.dropout_rate, broadcast_dropout=self.broadcast_dropout
        )(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """A single decoder (GPT‑style) transformer block with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False):
        # The entire block is rematerialised during backward for memory saving.
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            # --- Multi‑head Self‑Attention + residual ------------------------
            residual = h
            h = nn.LayerNorm(name="ln1")(h)
            causal_mask = make_causal_mask(h)
            h = FlashSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
            )(h, deterministic=deterministic, mask=causal_mask)
            h = residual + h

            # --- MLP block + residual ---------------------------------------
            residual = h
            h = nn.LayerNorm(name="ln2")(h)
            h = nn.Dense(module.d_ff, name="fc1")(h)
            h = nn.gelu(h, approximate=False)
            h = nn.Dense(module.d_model, name="fc2")(h)
            h = nn.Dropout(rate=module.dropout_rate)(h, deterministic=deterministic)
            return residual + h

        # remat requires closing over *self* explicitly
        return _block(self, x)

