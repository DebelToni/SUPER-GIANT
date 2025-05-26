import jax.numpy as jnp
from flax import linen as nn
from Transformer_block import TinyTransformerBlock


class GiantGPT(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, deterministic: bool = False):
        x = nn.Embed(num_embeddings=self.vocab_size,
                     features=self.d_model,
                     embedding_init=nn.initializers.normal(stddev=0.02))(tokens)

        pos_emb = self.param("pos_emb",
                             nn.initializers.normal(stddev=0.02),
                             (self.context_length, self.d_model))
        x = x + pos_emb[:x.shape[1]]

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)

        logits = nn.Dense(self.vocab_size,
                          kernel_init=nn.initializers.normal(stddev=0.02))(x)
        return logits

