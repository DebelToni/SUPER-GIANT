from typing import Optional
import functools
import jax

import jax.numpy as jnp
from flax import linen as nn
# ───────────────────────────────────────────────
# bring in the JIT-compiled transformer block
# ───────────────────────────────────────────────
from Transformer_block import TinyTransformerBlock, transformer_block_apply

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
        # Embedding
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=Config.compute_dtype,
            param_dtype=Config.param_dtype,
        )
        x = embed(tokens)

        # Input dropout
        x = nn.Dropout(rate=self.dropout_rate)(x,
                                                 deterministic=deterministic)

        # Transformer layers (JIT-compiled)
        for idx in range(self.n_layers):
            # Each layer’s params live under "layer_{idx}" in the param tree.
            layer_params = self.scope.get_variable("params",
                                                   f"layer_{idx}",
                                                   None)
            if layer_params is None:
                # First invocation: initialize & stash parameters
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
            # Apply the JIT-compiled transformer block
            x = transformer_block_apply(
                layer_params,
                x,
                rng=layer_rng,
                deterministic=deterministic,
                enable_kv_cache=enable_kv_cache,
                cur_index=cur_index,
            )

        # Output logits via tied embedding
        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding
        )
        return logits

# ---------------------------------------------------------------------------
# One-shot JIT for the whole model
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("deterministic", "enable_kv_cache", "cur_index"),
)
def giant_gpt_apply(params,
                    tokens,
                    *,
                    rng,
                    deterministic: bool = False,
                    enable_kv_cache: bool = False,
                    cur_index: Optional[int] = None):
    """Compiled forward pass for GiantGPT."""
    return GiantGPT(
        vocab_size=Config.vocab_size,
        context_length=Config.context_length,
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        d_ff=Config.d_ff,
        n_layers=Config.n_layers,
        dropout_rate=Config.dropout_rate,
    ).apply(
        {"params": params},
        tokens,
        deterministic=deterministic,
        enable_kv_cache=enable_kv_cache,
        cur_index=cur_index,
        rngs={"dropout": rng},
    )

