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
    """Minimal GPT‑style transformer block.

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed‑forward dimension (≈ 4 × ``d_model`` is common).
    dropout_rate : float, optional
        Dropout rate applied to attention and MLP outputs.  Defaults to ``0.1``.
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        """Run the transformer block.

        Parameters
        ----------
        x : jnp.ndarray
            Hidden states of shape ``(batch, seq_len, d_model)``.
        deterministic : bool, optional
            If ``True``, disables dropout (use for evaluation / generation).

        Returns
        -------
        jnp.ndarray
            Output hidden states, same shape as *x*.
        """

        @nn.remat  # ↔ gradient‑checkpoint the whole block to save GPU memory
        def _block(h: jnp.ndarray) -> jnp.ndarray:
            # ─── Multi‑head self‑attention ────────────────────────────────────
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                broadcast_dropout=False,
            )(h)
            h = residual + h

            # ─── Position‑wise MLP ───────────────────────────────────────────
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(self.d_ff)(h)
            h = nn.gelu(h, approximate=False)
            h = nn.Dense(self.d_model)(h)
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            return residual + h

        return _block(x)

