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
Transformer_block.py
A single Transformer decoder/GPT block with FlashAttention for Flax/JAX.
"""

import jax
# Enable JAX's FlashAttention kernels (Ampere+ support)
jax.config.update("jax_enable_flash_attention", True)

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import make_causal_mask
from jax.lax.attention import dot_product_attention


class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic: bool = False):
        # Pre-norm
        residual = x
        h = nn.LayerNorm()(x)

        # Build causal mask: shape [batch, 1, seq_len, seq_len]
        mask = make_causal_mask(x)

        # Project to QKV (fused) -> shape [batch, seq_len, 3, n_heads, head_dim]
        head_dim = self.d_model // self.n_heads
        qkv = nn.DenseGeneral(
            features=(3, self.n_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="qkv"
        )(h)
        # Split Q, K, V
        q, k, v = jnp.split(qkv, 3, axis=2)
        # Remove the split axis: now each is [batch, seq_len, n_heads, head_dim]
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        # Transpose to [batch, n_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scale queries
        q = q * jnp.sqrt(head_dim).astype(q.dtype)

        # FlashAttention call
        attn = dot_product_attention(
            query=q,
            key=k,
            value=v,
            bias=mask,
            dropout_rate=0.0 if deterministic else self.dropout_rate,
            deterministic=deterministic,
            precision=jax.lax.Precision.HIGHEST,
        )
        # attn: [batch, n_heads, seq_len, head_dim]

        # Recombine heads
        attn = jnp.transpose(attn, (0, 2, 1, 3))      # [batch, seq_len, n_heads, head_dim]
        attn = attn.reshape(attn.shape[0], attn.shape[1], -1)  # [batch, seq_len, d_model]

        # Output projection + residual
        h = nn.Dense(self.d_model, name="out_proj")(attn)
        h = residual + nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)

        # Feed-forward
        residual = h
        h = nn.LayerNorm()(h)
        h = nn.Dense(self.d_ff, name="fc1")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, name="fc2")(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return residual + h


# GiantGPT.py (snippet)
"""
At the top of your model definition, ensure FlashAttention is enabled before any model.init/use.
"""
import jax
# Globally enable FlashAttention for speed and memory efficiency
jax.config.update("jax_enable_flash_attention", True)

from flax import linen as nn
from Transformer_block import TinyTransformerBlock

class GiantGPT(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic: bool = False):
        # Token embeddings
        h = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, name="embed")(x)
        # Positional embeddings
        pos = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (1, x.shape[1], self.d_model))
        h = h + pos
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)

        # Transformer decoder blocks with FlashAttention
        for i in range(self.n_layers):
            h = TinyTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}"
            )(h, deterministic=deterministic)

        # Final norm + head
        h = nn.LayerNorm(name="final_ln")(h)
        logits = nn.Dense(self.vocab_size, name="lm_head")(h)
        return logits

