# import Config as Config 
# """"""
# import jax.numpy as jnp
# # import flax.linen as nn
# # from functools import partial
# # print ( nn.remat_policy )
# # remat = partial(nn.remat, policy=nn.remat_policy.POLICY_CHECKPOINT)
# # remat = partial(nn.remat, policy=nn.remat_policy.POLICY_CHECKPOINT)
#
# import flax.linen as nn
# remat = nn.remat
#
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
#         # for _ in range(self.n_layers):
#         #     if Config.use_remat:
#         #         x = remat(TinyTransformerBlock(self.d_model, self.n_heads, self.d_ff))(x, deterministic=deterministic)
#         #         # x = TinyTransformerBlock(self.d_model, self.n_heads, self.d_ff)(x, deterministic=deterministic)
#         #     else:
#         #         x = TinyTransformerBlock(self.d_model, self.n_heads, self.d_ff)(x, deterministic=deterministic)
#         for _ in range(self.n_layers):
#             BlockClass = TinyTransformerBlock
#             if Config.use_remat:
#                 BlockClass = remat(BlockClass)        # wrap the *class*
#
#             x = BlockClass(                           # then instantiate
#                     self.d_model,
#                     self.n_heads,
#                     self.d_ff
#                 )(x, deterministic=deterministic)
#         # 3. Output projection
#         # logits = nn.Dense(self.vocab_size, use_bias=False)(x)  # [batch, seq, vocab_size]
#         # return logits
#         return nn.Dense(self.vocab_size, use_bias=False)(x)  # [batch, seq, vocab_size]
#
import jax
import jax.numpy as jnp
import flax.linen as nn
import Config

# Memory-efficient transformer block with optional checkpointing
# Each block's activations can be re-computed on the backward pass,
# reducing peak memory by ~40-50%.
# We mark `deterministic` as static so we avoid tracer-boolean errors.
# remat = nn.remat

class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Multi-head self-attention
        attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            use_bias=True,
            deterministic=deterministic,
        )(x)
        # Dropout + residual
        attn = nn.Dropout(rate=0.1, deterministic=deterministic)(attn)
        x = x + attn
        # Feed-forward
        ff = nn.Dense(self.d_ff)(x)
        ff = nn.gelu(ff)
        ff = nn.Dropout(rate=0.1, deterministic=deterministic)(ff)
        ff = nn.Dense(self.d_model)(ff)
        x = x + ff
        return x

class TinyTransformerLM(nn.Module):
    vocab_size: int
    max_len: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int = 4

    @nn.compact
    def __call__(self, ids, deterministic: bool = False):
        # Token + position embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(ids)
        pos = jnp.arange(ids.shape[1])[None, :]
        pos_emb = nn.Embed(self.max_len, self.d_model)(pos)
        x = tok_emb + pos_emb

        # Transformer layers
        for _ in range(self.n_layers):
            BlockClass = TinyTransformerBlock
            # if Config.use_remat:
            #     # Wrap the class, marking 'deterministic' as a static arg
            #     # BlockClass = remat(BlockClass, static_argnames=('deterministic',))
            #     BlockClass = remat(
            #         TinyTransformerBlock,
            #         static_argnums=(2,)           # mark 'deterministic' (arg index 2) static
            #     )
            # Instantiate and apply
            x = BlockClass(
                self.d_model,
                self.n_heads,
                self.d_ff,
            )(x, deterministic=deterministic)

        # Final layer norm + classifier
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)
        return logits

