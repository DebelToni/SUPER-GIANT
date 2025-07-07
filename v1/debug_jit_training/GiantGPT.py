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

            # layer_rng = self.make_rng("dropout")
            # x = transformer_block_apply(
            #     layer_params,
            #     x,
            #     rng=layer_rng,
            #     deterministic=deterministic,
            #     enable_kv_cache=enable_kv_cache,
            #     cur_index=cur_index,
            # )
                # if deterministic:            # inference → no dropout → no rng needed
                #     x = transformer_block_apply(
                #         layer_params, x,
                #         rng=None,                # no dropout, so no rng
                #         deterministic=True,
                #         enable_kv_cache=enable_kv_cache,
                #         cur_index=cur_index,
                #     )
                # else:                        # training → need a fresh sub-key
                #     layer_rng = self.make_rng("dropout")
                #     x = transformer_block_apply(
                #         layer_params, x,
                #         rng=layer_rng,
                #         deterministic=False,
                #         enable_kv_cache=enable_kv_cache,
                #         cur_index=cur_index,
                #     )
            # ───────────────────────────────────────── cache handling
                layer_cache = self.scope.get_variable("cache", f"layer_{idx}", None)

                if deterministic:             # inference – no dropout key needed
                    x, new_cache = transformer_block_apply(
                        layer_params, layer_cache, x,
                        deterministic=True,
                        enable_kv_cache=enable_kv_cache,
                        cur_index=cur_index,
                    )
                else:                         # training – supply a fresh key
                    layer_rng = self.make_rng("dropout")
                    x, new_cache = transformer_block_apply(
                        layer_params, layer_cache, x,
                        rng=layer_rng,
                        deterministic=False,
                        enable_kv_cache=enable_kv_cache,
                        cur_index=cur_index,
                    )

                # write the cache back so it’s available next token
                if enable_kv_cache and new_cache is not None:
                    self.scope.put_variable("cache", f"layer_{idx}", new_cache)


        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits

@functools.partial(
    jax.jit,
    static_argnames=("deterministic", "enable_kv_cache"),  # cur_index NOT static
)
def giant_gpt_apply(
    params,
    cache,                   # ← NEW positional arg
    tokens,
    *,                       # keyword-only from here
    rng=None,
    deterministic: bool = False,
    enable_kv_cache: bool = False,
    cur_index: Optional[int] = None,
):
    # ── 1. Get vocab size (unchanged) ───────────────────────────
    if Config.use_custom_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(
            Config.custom_tokenizer_path)
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
    # ── 1) Prepare variable collections ─────────────────────────
    variables = {"params": params}
    if enable_kv_cache and cache is not None:
        variables["cache"] = cache

    # ── 2) RNG dict if needed ───────────────────────────────────
    rngs_kw = {"rngs": {"dropout": rng}} if rng is not None else {}

    if enable_kv_cache:
        # ── Inference: return logits + updated cache ─────────────
        logits, mutated = model.apply(
            variables,
            tokens,
            deterministic=deterministic,
            enable_kv_cache=True,
            cur_index=cur_index,
            mutable=["cache"],
            **rngs_kw,
        )
        return logits, mutated["cache"]
    else:
        # ── Training/Eval: no cache → only logits ─────────────────
        logits = model.apply(
            variables,
            tokens,
            deterministic=deterministic,
            enable_kv_cache=False,
            cur_index=cur_index,
            **rngs_kw,
        )
        return logits
