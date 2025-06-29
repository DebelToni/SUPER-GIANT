
from typing import Optional
import jax.numpy as jnp
from flax import linen as nn
from Transformer_block import TinyTransformerBlock
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
    def __call__(self, tokens, *, deterministic: bool = False, decode: bool = False, cur_index: Optional[int] = None):
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=Config.compute_dtype,
            param_dtype=Config.param_dtype,
        )
        x = embed(tokens)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype,
            )(x, deterministic=deterministic, decode=decode, cur_index=cur_index)


        logits = jnp.einsum("bld,vd->blv",
                            x.astype(jnp.float32),
                            embed.embedding)
        return logits
