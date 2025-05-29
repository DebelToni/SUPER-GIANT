# GiantGPT.py

import jax.numpy as jnp
from flax import linen as nn
from Transformer_block import TinyTransformerBlock
import Config # Import Config to access dtypes

class GiantGPT(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1
    cpu:            bool = False  # If True, use CPU for inference

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, deterministic: bool = False):
        # Embed and cast to compute_dtype (bf16)
        x = nn.Embed(num_embeddings=self.vocab_size,
                     features=self.d_model,
                     embedding_init=nn.initializers.normal(stddev=0.02),
                     dtype=Config.compute_dtype, # Output bf16
                     param_dtype=Config.param_dtype)(tokens)

        # Positional embeddings should also be bf16
        pos_emb = self.param("pos_emb",
                             nn.initializers.normal(stddev=0.02),
                             (self.context_length, self.d_model),
                             Config.param_dtype) # Store in f32
        x = x + pos_emb[:x.shape[1]].astype(Config.compute_dtype) # Add as bf16

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Transformer blocks operate in bf16
        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    dtype=Config.compute_dtype, # Ensure blocks use bf16
                    cpu=self.cpu  # Pass CPU flag if needed
            )(x, deterministic=deterministic)
        
        # --- IMPORTANT ---
        # Before the final Dense layer, it's good practice to have a final
        # LayerNorm in float32. Let's add one if it's missing or ensure
        # the last block's output is suitable. Here we assume the last block's
        # output 'x' is bf16. We should cast it or use a float32 LN.
        # For simplicity, we'll cast just before the final Dense layer.
        
        # Final Dense layer for logits - should output float32 for stability
        # with the loss function.
        logits = nn.Dense(self.vocab_size,
                          kernel_init=nn.initializers.normal(stddev=0.02),
                          dtype=jnp.float32, # Output float32
                          param_dtype=Config.param_dtype)(x.astype(jnp.float32)) # Input must be float32
                          
        return logits
