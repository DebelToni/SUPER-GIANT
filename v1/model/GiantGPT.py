"""
GPT wrapper with an optional `use_cache` flag that wires all transformer blocks
into caching mode.  Nothing else changes.
"""

import functools, jax
import jax.numpy as jnp
from flax import linen as nn
from Transformer_block import TinyTransformerBlock
import Config

class GiantGPT(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1
    use_cache:      bool = False    # NEW
    cpu:            bool = False

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 deterministic: bool = True,
                 init_cache: bool = False,   # if True, zero‑init caches
                 ) -> jnp.ndarray:
        # ----- device placement ----------------------------------------------
        if self.cpu:
            x = jax.device_put(x, jax.devices("cpu")[0])

        # ----- embed ----------------------------------------------------------
        emb = nn.Embed(self.vocab_size,
                       features=self.d_model,
                       embedding_init=nn.initializers.normal(stddev=0.02),
                       dtype=Config.compute_dtype,
                       param_dtype=Config.param_dtype,
                       name="tok_emb")(x)

        pos_emb = self.param("pos_emb",
                             nn.initializers.normal(stddev=0.02),
                             (self.context_length, self.d_model),
                             Config.param_dtype)
        emb = emb + pos_emb[:emb.shape[1]].astype(Config.compute_dtype)
        h   = nn.Dropout(rate=self.dropout_rate)(emb, deterministic=deterministic)

        # ----- transformer layers --------------------------------------------
        for i in range(self.n_layers):
            h = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype,
                    use_cache=self.use_cache,
                    name=f"block_{i}")(h, deterministic=deterministic)

        h = nn.LayerNorm(dtype=jnp.float32, name="ln_f")(h)
        logits = nn.Dense(self.vocab_size,
                          dtype=jnp.float32,           # final logits in fp32
                          param_dtype=Config.param_dtype,
                          kernel_init=nn.initializers.normal(stddev=0.02),
                          name="head")(h)
        return logits
