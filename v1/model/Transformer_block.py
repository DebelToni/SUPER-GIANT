# """
# Memory-efficient Transformer block using Flax + JAX remat (gradient checkpointing).
#
# Main tricks:
# 1. Wrap *each residual branch* (Attention, MLP) in flax.linen.remat → saves activations.
# 2. Keep the subbranches pure functions so remat works.
# 3. Dropout is outside remat so rng is not recomputed during backward.
# 4. Block remains compatible with earlier TinyTransformerLM class.
#
# If you need even more savings, wrap the for-loop of layers
# with linen.scan or remat_scan(). See README for details.
# """
#
# from typing import Callable
# import jax.numpy as jnp
# import flax.linen as nn
#
# remat = nn.remat   # alias
#
# class MLP(nn.Module):
#     d_model: int
#     d_ff: int
#     dropout: float = 0.1
#
#     @nn.compact
#     def __call__(self, x, deterministic: bool):
#         x = nn.Dense(self.d_ff)(x)
#         x = nn.gelu(x)
#         x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
#         x = nn.Dense(self.d_model)(x)
#         return x
#
#
# class TinyTransformerBlock(nn.Module):
#     """
#     One Transformer block with gradient-checkpointing on
#     (a) the Self-Attention branch  (b) the Feed-Forward branch.
#
#     Args:
#         d_model: model hidden size
#         n_heads: number of attention heads
#         d_ff: feed-forward expansion
#         dropout: dropout rate
#     """
#     d_model: int
#     n_heads: int
#     d_ff: int
#     dropout: float = 0.1
#
#     @nn.compact
#     def __call__(self, x, *, deterministic: bool = False):
#         # ---- Attention branch ----------
#         def attn_branch(y):
#             y = nn.LayerNorm()(y)
#             y = nn.SelfAttention(
#                 num_heads=self.n_heads,
#                 qkv_features=self.d_model,
#                 use_bias=True,
#                 deterministic=deterministic,
#             )(y)
#             return y
#
#         # remat = checkpoint
#         attn_out = remat(attn_branch)(x)
#         x = x + nn.Dropout(rate=self.dropout)(attn_out, deterministic=deterministic)
#
#         # ---- Feed-forward branch -------
#         def ffn_branch(y):
#             y = nn.LayerNorm()(y)
#             y = MLP(self.d_model, self.d_ff, self.dropout)(y, deterministic)
#             return y
#
#         ffn_out = remat(ffn_branch)(x)
#         x = x + nn.Dropout(rate=self.dropout)(ffn_out, deterministic=deterministic)
#         return x
#
#
# class TinyTransformerLM(nn.Module):
#     vocab_size: int
#     max_len: int
#     d_model: int
#     n_heads: int
#     d_ff: int
#     n_layers: int
#     dropout: float = 0.1
#
#     @nn.compact
#     def __call__(self, x, *, deterministic: bool = False):
#         tok_emb = nn.Embed(self.vocab_size, self.d_model)(x)
#         pos_emb = nn.Embed(self.max_len, self.d_model)(jnp.arange(self.max_len))[None, :, :]
#         h = tok_emb + pos_emb[:, :x.shape[1], :]
#         h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
#
#         # Checkpoint each full layer; cheapest is to wrap entire loop with scan/remat_scan
#         for _ in range(self.n_layers):
#             h = TinyTransformerBlock(
#                 self.d_model, self.n_heads, self.d_ff, self.dropout
#             )(h, deterministic=deterministic)
#
#         h = nn.LayerNorm()(h)
#         logits = nn.Dense(self.vocab_size)(h)
#         return logits
#
"""
Memory-efficient Transformer block using Flax + JAX remat (gradient checkpointing).

This version avoids closures over Module state by explicitly wrapping submodules in remat.
"""

from typing import Any
import jax.numpy as jnp
import flax.linen as nn

# alias for checkpoint
remat = nn.remat

class MLP(nn.Module):
    d_model: int
    d_ff: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = nn.Dense(self.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        return x

class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        # ---- Attention branch (pre-LN -> rematted SelfAttention -> dropout -> residual) ----
        ln1 = nn.LayerNorm()
        sa = remat(
            nn.SelfAttention,
            static_argnums=(3,),  # deterministic is arg index 3
        )(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            use_bias=True,
        )
        y = ln1(x)
        # SelfAttention signature: (inputs_q, inputs_kv=None, mask=None, deterministic=False)
        attn_out = sa(y, y, None, deterministic)
        x = x + nn.Dropout(rate=self.dropout)(attn_out, deterministic=deterministic)

        # ---- Feed-forward branch (pre-LN -> rematted MLP -> dropout -> residual) ----
        ln2 = nn.LayerNorm()
        ffn = remat(
            MLP,
            static_argnums=(1,),  # deterministic is arg index 1
        )(self.d_model, self.d_ff, self.dropout)
        y2 = ln2(x)
        ffn_out = ffn(y2, deterministic)
        x = x + nn.Dropout(rate=self.dropout)(ffn_out, deterministic=deterministic)

        return x

class TinyTransformerLM(nn.Module):
    vocab_size: int
    max_len: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        # token embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(x)
        # positional embeddings
        seq_len = x.shape[1]
        pos_ids = jnp.arange(self.max_len)[None, :seq_len]
        pos_emb = nn.Embed(self.max_len, self.d_model)(pos_ids)
        h = tok_emb + pos_emb
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)

        # transformer layers
        for _ in range(self.n_layers):
            h = TinyTransformerBlock(
                self.d_model, self.n_heads, self.d_ff, self.dropout
            )(h, deterministic=deterministic)

        h = nn.LayerNorm()(h)
        logits = nn.Dense(self.vocab_size)(h)
        return logits

