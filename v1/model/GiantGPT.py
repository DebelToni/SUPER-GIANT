from typing import Optional
import functools
import jax

import jax.numpy as jnp
from flax import linen as nn
from Transformer_block import TinyTransformerBlock, transformer_block_apply
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

class GiantGPT(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1

    @nn.compact
    def __call__(self,
                 tokens,
                 *,
                 deterministic: bool = False,
                 enable_kv_cache: bool = False,
                 cur_index: Optional[int] = None):
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=Config.compute_dtype,
            param_dtype=Config.param_dtype,
        )
        x = embed(tokens)

        x = nn.Dropout(rate=self.dropout_rate)(x,
                                                 deterministic=deterministic)

        for idx in range(self.n_layers):
            layer_params = self.scope.get_variable("params",
                                                   f"layer_{idx}",
                                                   None)
            if layer_params is None:
                block = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype,
                    name=f"layer_{idx}",
                )
                init_out = block.init(
                    self.make_rng("params"),
                    x,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                layer_params = init_out["params"]
                self.scope.put_variable("params",
                                        f"layer_{idx}",
                                        layer_params)

            layer_rng = self.make_rng("dropout")
            x = transformer_block_apply(
                layer_params,
                x,
                rng=layer_rng,
                deterministic=deterministic,
                enable_kv_cache=enable_kv_cache,
                cur_index=cur_index,
            )

        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits


# @functools.partial(
#     jax.jit,
#     static_argnames=("deterministic", "enable_kv_cache", "cur_index"),
# )
# def giant_gpt_apply(params,
#                     tokens,
#                     *,
#                     rng = None,
#                     deterministic: bool = False,
#                     enable_kv_cache: bool = False,
#                     cur_index: Optional[int] = None):
#
#     if Config.use_custom_tokenizer:
#         tok = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
#     else:
#         tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)
#
#     model = GiantGPT(
#         vocab_size=tok.vocab_size,
#         context_length=Config.context_length,
#         d_model=Config.embedding_size,
#         n_heads=Config.num_heads,
#         d_ff=Config.feed_forward_size,
#         n_layers=Config.num_layers,
#         dropout_rate=Config.dropout_rate,
#     )
#
#     extra_kwargs = {}
#     if rng is not None:
#         extra_kwargs["rngs"] = {"dropout": rng}
#
#     return model.apply(
#         {"params": params},
#         tokens,
#         deterministic=deterministic,
#         enable_kv_cache=enable_kv_cache,
#         cur_index=cur_index,
#         **extra_kwargs,
#     )

# GiantGPT.py  --------------------------------------------------------------

@functools.partial(
    jax.jit,
    # static_argnames=("deterministic", "enable_kv_cache", "cur_index"),
    static_argnames=("deterministic", "enable_kv_cache"),
)
def giant_gpt_apply(params,
                    tokens,
                    *,
                    cache=None,                    # ← NEW
                    rng=None,                      # ← unchanged
                    deterministic: bool = False,
                    enable_kv_cache: bool = False,
                    cur_index: Optional[int] = None):

    # -------- tokenizer / model instantiation (unchanged) -----------------
    if Config.use_custom_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    model = GiantGPT(
        vocab_size=tok.vocab_size,
        context_length=Config.context_length,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.feed_forward_size,
        n_layers=Config.num_layers,
        dropout_rate=Config.dropout_rate,
    )

    # -------- build the variables dict ------------------------------------
    variables = {"params": params}
    if cache is not None:
        variables["cache"] = cache            # hold previous KV tensors

    extra = {}
    if rng is not None:
        extra["rngs"] = {"dropout": rng}      # key only when supplied

    # If we carry a cache we must mark it mutable so we get the
    # *updated* cache back.
    mutable = ["cache"] if cache is not None else False

    out = model.apply(
        variables,
        tokens,
        deterministic=deterministic,
        enable_kv_cache=enable_kv_cache,
        cur_index=cur_index,
        mutable=mutable,
        **extra,
    )

    if cache is None:                # no caching path → just logits
        return out                        # logits tensor
    else:                           # caching path → (logits, new_cache)
        logits, new_vars = out
        return logits, new_vars["cache"]

