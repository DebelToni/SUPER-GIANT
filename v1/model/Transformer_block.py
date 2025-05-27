# Transformer_block.py – Native JAX cuDNN Flash‑Attention (Mixed Precision)

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
import Config # Import Config to access dtypes if needed, or pass explicitly

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = Config.compute_dtype # Use compute_dtype

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        # Dense layers for QKV projections.
        # They will use compute_dtype (bf16) for compute,
        # but their params will be float32 (Flax default/can be set).
        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.k_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.v_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray, # Expects x to be in compute_dtype (bf16)
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        b, l, _ = x.shape

        # Project to Q‑K‑V and reshape to (B, T, H, D)
        # Inputs are bf16, projections compute in bf16, outputs are bf16.
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_heads, self.head_dim)

        # q, k, v are now bf16, satisfying the cuDNN requirement.
        y = jax.nn.dot_product_attention(
            q,
            k,
            v,
            bias=None,
            is_causal=True,
            implementation="cudnn",
        )

        # Merge heads and apply output projection (still in bf16)
        y = y.reshape(b, l, -1)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = Config.compute_dtype # Use compute_dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False):
        # x is expected to be bf16
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            # --- Self‑Attention --------------------------------------------------
            residual = h # bf16
            # LayerNorm computes in f32, casts back to bf16
            h_norm = nn.LayerNorm(name="ln1", dtype=jnp.float32)(h) 
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype, # Pass bf16
            )(h_norm, deterministic=deterministic)
            h = residual + h_attn # bf16

            # --- Feed‑forward ----------------------------------------------------
            residual = h # bf16
            # LayerNorm computes in f32, casts back to bf16
            h_norm = nn.LayerNorm(name="ln2", dtype=jnp.float32)(h) 
            h_ffn = nn.Dense(module.d_ff, name="fc1", dtype=module.dtype, param_dtype=Config.param_dtype)(h_norm)
            h_ffn = nn.gelu(h_ffn, approximate=False)
            h_ffn = nn.Dense(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn # bf16

        return _block(self, x)
