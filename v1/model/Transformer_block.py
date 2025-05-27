# Transformer_block.py – Native JAX cuDNN Flash‑Attention (Corrected)

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
# We no longer need make_causal_mask if we rely solely on is_causal=True
# from flax.linen import make_causal_mask

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj")
        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj")
        
        # Dropout applied *after* attention + output projection
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        # We remove the 'mask' parameter here, as we won't be passing it
        # mask: Optional[jnp.ndarray] = None, 
    ) -> jnp.ndarray:
        b, l, _ = x.shape

        # Project to Q‑K‑V and reshape to (B, T, H, D)
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_heads, self.head_dim)

        # Call JAX’s attention fn.
        # --- FIX: REMOVED mask=mask ---
        y = jax.nn.dot_product_attention(
            q,
            k,
            v,
            bias=None,        # No explicit bias unless handling padding
            # mask=mask,      # <-- REMOVED! This was likely causing the error.
            is_causal=True,   # <-- Rely on this for causality.
            implementation="cudnn",
        )

        # Merge heads and apply output projection
        y = y.reshape(b, l, -1)
        y = self.o_proj(y)
        # Apply dropout *after* output projection
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False):
        # Rematerialise whole block to reduce activation memory.
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            # --- Self‑Attention --------------------------------------------------
            residual = h
            h = nn.LayerNorm(name="ln1")(h)
            
            # --- FIX: REMOVED make_causal_mask ---
            # causal_mask = make_causal_mask(h) # <-- REMOVED! No longer needed.
            
            # --- FIX: REMOVED mask=causal_mask ---
            h = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
            )(h, deterministic=deterministic) # <-- REMOVED 'mask=' argument
            h = residual + h

            # --- Feed‑forward ----------------------------------------------------
            residual = h
            h = nn.LayerNorm(name="ln2")(h)
            h = nn.Dense(module.d_ff, name="fc1")(h)
            h = nn.gelu(h, approximate=False)
            h = nn.Dense(module.d_model, name="fc2")(h)
            h = nn.Dropout(rate=module.dropout_rate)(h, deterministic=deterministic)
            return residual + h

        return _block(self, x)
