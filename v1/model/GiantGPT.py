# GiantGPT.py
from __future__ import annotations

from typing import Optional
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from Transformer_block import TinyTransformerBlock, transformer_block_apply
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from omegaconf import OmegaConf

Config = OmegaConf.load("Config.yml")


class GiantGPT(nn.Module):
    """Decoder‑only GPT‑style model composed of TinyTransformerBlocks."""

    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        *,
        deterministic: bool = False,
        enable_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ) -> jnp.ndarray:
        """Tokens → logits.  Works both with & without KV‑cache."""

        # 1. Token embeddings + dropout
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=Config.compute_dtype,
            param_dtype=Config.param_dtype,
        )
        x = embed(tokens)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=deterministic
        )

        # 2. Transformer layers ------------------------------------------------
        for idx in range(self.n_layers):
            layer_name = f"layer_{idx}"

            # Grab parameters & (if active) cache for this layer
            layer_params = self.scope.get_variable("params", layer_name, None)
            if layer_params is None:
                # First run (typically inside model.init): create parameters
                block = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype,
                    name=layer_name,
                )
                init_vars = block.init(
                    self.make_rng("params"),
                    x,
                    deterministic=deterministic,
                    enable_kv_cache=enable_kv_cache,
                    cur_index=cur_index,
                )
                layer_params = init_vars["params"]
                self.scope.put_variable("params", layer_name, layer_params)

            layer_cache = self.scope.get_variable("cache", layer_name, None)

            # Per‑layer dropout key (only when not deterministic)
            if deterministic:
                layer_rng = None
            else:
                layer_rng = self.make_rng("dropout")

            # ---------- Call the **jitted** helper ---------------------------
            x, new_cache = transformer_block_apply(
                layer_params,
                layer_cache,
                x,
                rng=layer_rng,
                deterministic=deterministic,
                enable_kv_cache=enable_kv_cache,
                # cur_index=cur_index,
                # layer_name=layer_name,           # <-- UNIQUE NAME PER LAYER
                cur_index=cur_index,
            )

            if enable_kv_cache and new_cache is not None:
                self.scope.put_variable("cache", layer_name, new_cache)

        x = RMSNorm(dtype=Config.compute_dtype, name="rms_final")(x)

        # 3. Unembedding
        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding,
        )
        return logits


# ----------------------------------------------------------------------
# A convenience wrapper (remains mostly unchanged)
# ----------------------------------------------------------------------
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
    """Stateless helper: params + (optional) cache → logits (+new cache)."""

    # Build tokenizer only once per JIT trace
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

