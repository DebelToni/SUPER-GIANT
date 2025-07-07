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
                 cache=None,
                 deterministic=True,
                 rng=None,
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
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        new_cache = {} if cache is not None else None

        for idx in range(self.n_layers):
            layer_name = f"layer_{idx}"
            if cache is None:
                # --- init-mode: params=None, cache=None → handled inside block ---
                x, _ = transformer_block_apply(
                    params=None,
                    cache=None,
                    x=x,
                    rng=None,
                    layer_name=layer_name,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                # no cache yet
            else:
                # --- train/infer mode: unwrap params+cache, supply rngs properly ---
                layer_params = self.scope.get_variable("params", layer_name)
                layer_cache  = cache.get(layer_name, None)
                layer_rng = self.make_rng("dropout") if not deterministic else None

                x, new_layer_cache = transformer_block_apply(
                    params=layer_params,
                    cache=layer_cache,
                    x=x,
                    rng=layer_rng,
                    layer_name=layer_name,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                if enable_kv_cache and new_layer_cache is not None:
                    new_cache[layer_name] = new_layer_cache

        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits if cache is None else (logits, new_cache)

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
