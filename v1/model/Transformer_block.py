"""
Transformer_block.py

A single Transformer encoder/GPT block with memory‑efficient remat (gradient
checkpointing) for Flax/JAX.  Designed for small‑footprint training on
free‑tier GPUs.
"""

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class TinyTransformerBlock(nn.Module):

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic: bool = False):
        @nn.remat                           # checkpoint the whole block
        def _block(module, h):              # ① module first!
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.SelfAttention(
                num_heads   = module.n_heads,
                qkv_features= module.d_model,
                dropout_rate= module.dropout_rate,
                deterministic=deterministic,
                broadcast_dropout=False,
            )(h)
            h = residual + h

            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(module.d_ff)(h)
            h = nn.gelu(h, approximate=False)
            h = nn.Dense(module.d_model)(h)
            h = nn.Dropout(rate=module.dropout_rate)(h,
                                                     deterministic=deterministic)
            return residual + h

        return _block(self, x)              # ② pass *self* explicitly

