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
        # --- token embedding -------------------------------------------------
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=Config.compute_dtype,
            param_dtype=Config.param_dtype,
        )
        x = embed(tokens)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # --- transformer layers ---------------------------------------------
        for idx in range(self.n_layers):
            # 1. Fetch parameters (initialise on first call)
            layer_params = self.scope.get_variable("params", f"block_{idx}", None)
            if layer_params is None:
                block = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype,
                    name=f"block_{idx}",
                )
                init_out = block.init(
                    self.make_rng("params"),
                    x,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                layer_params = init_out["params"]
                self.scope.put_variable("params", f"block_{idx}", layer_params)

            # 2. Fetch per‑layer KV‑cache (if any)
            layer_cache = self.scope.get_variable("cache", f"block_{idx}", None)

            # 3. Apply the *compiled* block
            if deterministic:
                x, new_cache = transformer_block_apply(
                    layer_params,
                    layer_cache,
                    x,
                    deterministic=True,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                    layer_index=idx,
                )
            else:
                layer_rng = self.make_rng("dropout")
                x, new_cache = transformer_block_apply(
                    layer_params,
                    layer_cache,
                    x,
                    rng=layer_rng,
                    deterministic=False,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                    layer_index=idx,
                )

            # 4. Store updated cache
            if enable_kv_cache and new_cache is not None:
                self.scope.put_variable("cache", f"block_{idx}", new_cache)

        # --- final projection to logits --------------------------------------
        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits


 – jitted forward pass (optional)
# -------------------------------------------------------------------------- #
@functools.partial(
    jax.jit,
    static_argnames=("deterministic", "enable_kv_cache")
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
    """Fast wrapper for inference/decoding.

    *Does not* create the model inside the jitted function – the model is
    constructed once and captured as a static closure so that weight‑decay
    masking remains valid.
    """
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

    variables = {"params": params}
    if enable_kv_cache and cache is not None:
        variables["cache"] = cache

    rng_kw = {"rngs": {"dropout": rng}} if rng is not None else {}

    if enable_kv_cache:
        logits, mutated = model.apply(
            variables,
            tokens,
            deterministic=deterministic,
            enable_kv_cache=True,
            cur_index=cur_index,
            mutable=["cache"],
            **rng_kw,
        )
        return logits, mutated["cache"]
    else:
        logits = model.apply(
            variables,
            tokens,
            deterministic=deterministic,
            enable_kv_cache=False,
            cur_index=cur_index,
            **rng_kw,
        )
        return logits


# -------------------------------------------------------------------------- #
# Convenience wrapper – jitted forward pass (training & inference)
# -------------------------------------------------------------------------- #
def build_apply_fn(model):
    """Create a JIT‑compiled `apply_fn` bound to *model*.

    The returned function has the signature::

        (params, cache, tokens, *, rng=None,
         deterministic=False, enable_kv_cache=False, cur_index=None)

    It is safe to close over *model* because the object itself is treated as a
    **static** argument by XLA – it is captured only once during compilation
    (fix #4).
    """
    @functools.partial(
        jax.jit,
        static_argnames=("deterministic", "enable_kv_cache")
    )
    def _apply_fn(
        params,
        cache,
        tokens,
        *,
        rng=None,
        deterministic: bool = False,
        enable_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        variables = {"params": params}
        if enable_kv_cache and cache is not None:
            variables["cache"] = cache

        rng_kw = {"rngs": {"dropout": rng}} if rng is not None else {}

        if enable_kv_cache:
            logits, mutated = model.apply(
                variables,
                tokens,
                deterministic=deterministic,
                enable_kv_cache=True,
                cur_index=cur_index,
                mutable=["cache"],
                **rng_kw,
            )
            return logits, mutated["cache"]
        else:
            logits = model.apply(
                variables,
                tokens,
                deterministic=deterministic,
                enable_kv_cache=False,
                cur_index=cur_index,
                **rng_kw,
            )
            return logits

    return _apply_fn
