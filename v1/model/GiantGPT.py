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
            layer_name = f"layer_{idx}"
            layer_params = self.scope.get_variable("params", layer_name, None)
            block = TinyTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=Config.compute_dtype,
                name=layer_name,
            )
            if layer_params is None:
                init_out = block.init(
                    self.make_rng("params"),
                    x,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                layer_params = init_out["params"]
                self.scope.put_variable("params", layer_name, layer_params)
            layer_cache = self.scope.get_variable("cache", layer_name, None)

            apply_fn = jax.jit(
                block.apply,
                static_argnames=("deterministic", "enable_kv_cache", "mutable"),
            )
            vars = {"params": layer_params}
            if enable_kv_cache and layer_cache is not None:
                vars["cache"] =  layer_cache

            if enable_kv_cache:
                kw = {
                    "deterministic": deterministic,
                    "enable_kv_cache": True,
                    "cur_index": cur_index,
                    "mutable": ("cache",),
                }
                if not deterministic:
                    kw["rngs"] = {"dropout": self.make_rng("dropout")}
                y, mutated = apply_fn(vars, x, **kw)
                new_cache = mutated["cache"]
                self.scope.put_variable("cache", layer_name, new_cache)
                x = y
            else:
                kw = {
                    "deterministic": deterministic,
                    "enable_kv_cache": False,
                    "cur_index": cur_index,
                }
                if not deterministic:
                    kw["rngs"] = {"dropout": self.make_rng("dropout")}
                x = apply_fn(vars, x, **kw)


            if enable_kv_cache:
                new_cache = mutated["cache"]
                self.scope.put_variable("cache", layer_name, new_cache)



        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits

@functools.partial(
    jax.jit,
    static_argnames=("deterministic", "enable_kv_cache"),
)
def giant_gpt_apply(
    params,
    cache,
    tokens,
    *,
    rng=None,
    deterministic: bool = False,
    enable_kv_cache: bool = False,
    cur_index: Optional[int] = None,
):
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
    variables = {"params": params}
    if enable_kv_cache and cache is not None:
        variables["cache"] = cache

    rngs_kw = {"rngs": {"dropout": rng}} if rng is not None else {}

    if enable_kv_cache:
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
        logits = model.apply(
            variables,
            tokens,
            deterministic=deterministic,
            enable_kv_cache=False,
            cur_index=cur_index,
            **rngs_kw,
        )
        return logits
