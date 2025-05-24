# import Config as Config 
# """"""
# import jax.numpy as jnp
# import flax.linen as nn
#
# class TinyTransformerBlock(nn.Module):
#     d_model: int
#     n_heads: int
#     d_ff: int
#
#     @nn.compact
#     def __call__(self, x, *, deterministic=False):
#         # Self-attention sub-layer
#         # attn = nn.SelfAttention(num_heads=self.n_heads, qkv_features=self.d_model,
#         #                          use_bias=True, broadcast_dropout=False,
#         #                          deterministic=deterministic,
#         #                          dropout_rate=0.1)(x)   # shape: [batch, seq, d_model]
#         attn = nn.SelfAttention(
#                 num_heads=self.n_heads,
#                 qkv_features=self.d_model,
#                 use_bias=True,
#                 deterministic=deterministic,
#         )(x)
#         attn = nn.Dropout(0.1, deterministic=deterministic)(attn)
#         x = nn.LayerNorm()(x + attn)
#         # Feed-forward sub-layer
#         ff = nn.Dense(self.d_ff)(x)
#         ff = nn.gelu(ff)                     # activation
#         ff = nn.Dense(self.d_model)(ff)
#         ff = nn.Dropout(0.1, deterministic=deterministic)(ff)
#         x = nn.LayerNorm()(x + ff)
#         return x
#
# class TinyTransformerLM(nn.Module):
#     vocab_size: int
#     max_len: int
#     d_model: int
#     n_heads: int
#     d_ff: int
#     n_layers: int
#
#     @nn.compact
#     def __call__(self, token_ids, *, deterministic=False):
#         # token_ids shape: [batch, seq_length]
#         # 1. Embed tokens and positions
#         tok_emb = nn.Embed(self.vocab_size, self.d_model)(token_ids)    # [batch, seq, d_model]
#         pos_idx = jnp.arange(token_ids.shape[1])  # [seq]
#         pos_emb = self.param('pos_embedding',  # learned positional emb
#                               nn.initializers.normal(stddev=0.02),
#                               (self.max_len, self.d_model))
#         pos_emb = pos_emb[pos_idx]                       # [seq, d_model]
#         x = tok_emb + pos_emb                           # [batch, seq, d_model]
#         # 2. Transformer blocks
#         for _ in range(self.n_layers):
#             x = TinyTransformerBlock(self.d_model, self.n_heads, self.d_ff)(x, deterministic=deterministic)
#         # 3. Output projection
#         # logits = nn.Dense(self.vocab_size, use_bias=False)(x)  # [batch, seq, vocab_size]
#         # return logits
#         return nn.Dense(self.vocab_size, use_bias=False)(x)  # [batch, seq, vocab_size]
#
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import flax.linen as nn

# Datatypes for stability and memory savings
DTYPE_ACT  = jnp.float16
DTYPE_NORM = jnp.float32

class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = DTYPE_ACT

    @nn.compact
    def __call__(self, x, *, deterministic):
        # Pre-norm
        h = nn.LayerNorm(dtype=DTYPE_NORM)(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            use_bias=True,
            dtype=self.dtype,
            deterministic=deterministic
        )(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic)
        x = x + h

        # Feed-forward
        h = nn.LayerNorm(dtype=DTYPE_NORM)(x)
        h = nn.Dense(self.d_ff, dtype=self.dtype)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, dtype=self.dtype)(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic)
        return x + h

class TinyTransformerLM(nn.Module):
    vocab_size: int
    max_len: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = DTYPE_ACT

    @nn.compact
    def __call__(self, token_ids, *, deterministic=False):
        # Embed tokens and positions in fp16
        x = nn.Embed(
            self.vocab_size, self.d_model,
            dtype=self.dtype
        )(token_ids)
        pos_emb = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        ).astype(self.dtype)
        x = x + pos_emb[:token_ids.shape[1]]

        # Remat+scan for transformer stack
        TransformerLayer = TinyTransformerBlock.partial(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )
        x = nn.remat_scan(
            TransformerLayer,
            variable_broadcast="params",
            split_rngs={"params": False},
            lengths=(self.n_layers,)
        )(x, deterministic=deterministic)

        # Project to vocab logits in fp32
        logits = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.float32)(x)
        return logits

