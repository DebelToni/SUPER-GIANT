from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import make_causal_mask


class NativeJaxSelfAttention(nn.Module):
    """
    Multi-head self-attention using jax.nn.dot_product_attention
    with the 'cudnn' backend request for potential Flash Attention optimization.
    """
    num_heads: int
    qkv_features: int  # Dimension of the model, d_model
    dropout_rate: float = 0.0
    # For broadcast_dropout behavior, nn.Dropout's broadcast_dims can be used.
    # If broadcast_dropout=False (user's original preference), it means dropout mask
    # is different per example. Flax's nn.Dropout default usually handles this well
    # by applying dropout on the feature dimension.

    def setup(self):
        assert self.qkv_features % self.num_heads == 0, \
            "qkv_features (d_model) must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        # Projections for Q, K, V
        self.query_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj")
        self.key_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj")
        self.value_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj")

        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray], deterministic: bool) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        query = self.query_proj(x)  # (batch_size, seq_len, qkv_features)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape for multi-head attention: (batch_size, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose Q, K, V to (batch_size, num_heads, seq_len, head_dim)
        # as jax.nn.dot_product_attention often expects this layout or can adapt.
        # Let's keep it (B, T, N, H) and let JAX handle it, or adjust if errors occur.
        # The documentation for jax.nn.dot_product_attention implies it can handle various
        # dimension orders as long as they are consistent.
        # For clarity, (B, N, T, H) is a common convention for raw attention functions.
        # query = jnp.transpose(query, (0, 2, 1, 3)) # (B, N, T, H)
        # key = jnp.transpose(key, (0, 2, 1, 3))   # (B, N, T, H)
        # value = jnp.transpose(value, (0, 2, 1, 3)) # (B, N, T, H)
        # The `mask` from `make_causal_mask` is (B, 1, T, T). This should broadcast to (B, N, T, T).

        # Apply scaled dot-product attention using JAX's native function
        # `is_causal=True` enables causal masking.
        # The `mask` argument can take the boolean mask from `make_causal_mask`.
        # If `is_causal=True`, the explicit `mask` is combined (ANDed).
        # For pure causal attention, `is_causal=True` is the main driver for cuDNN.
        # If `mask` is purely for causality, `is_causal=True` is preferred and `mask` can be None
        # or the causal mask itself. If `mask` includes padding, it should be passed.
        # The `bias` argument is for additive masks.
        attention_output = jax.nn.dot_product_attention(
            query, # (B, T, N, H)
            key,   # (B, T, N, H)
            value, # (B, T, N, H)
            mask=mask, # Boolean mask (B, 1, T, T) from make_causal_mask
            is_causal=True, # Indicates causal attention for potential cuDNN optimization
            dropout_rate=self.dropout_rate if not deterministic else 0.0, # Pass dropout here
            deterministic=deterministic,
            implementation="cudnn" # Request cuDNN backend
        )
        # Output shape: (batch_size, seq_len, num_heads, head_dim) if inputs were (B,T,N,H)
        # Or (batch_size, num_heads, seq_len, head_dim) if inputs were (B,N,T,H)
        # Assuming output is (B, T, N, H) matching input Q shape convention

        # If transposed earlier, transpose back:
        # attention_output = jnp.transpose(attention_output, (0, 2, 1, 3)) # (B, T, N, H)

        # Merge heads: (batch_size, seq_len, qkv_features)
        attention_output_merged = attention_output.reshape(batch_size, seq_len, -1)

        # Final output projection
        # Dropout is handled by dot_product_attention, so not applying self.dropout_layer here
        output = self.out_proj(attention_output_merged)
        # If dot_product_attention's internal dropout is not preferred,
        # set dropout_rate=0.0 there and apply self.dropout_layer here.
        # For simplicity, using its internal dropout.

        return output


class TinyTransformerBlock(nn.Module):
    """A single decoder (GPT-style) transformer block with checkpointing."""
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        @nn.remat
        def _block(module: TinyTransformerBlock, current_x: jnp.ndarray) -> jnp.ndarray:
            # --- Multi-head Self-Attention + residual ------------------------
            residual = current_x
            h = nn.LayerNorm(name="ln1")(current_x)
            
            causal_mask = make_causal_mask(h) # Shape: (batch, 1, seq_len, seq_len)

            h = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                name="native_jax_self_attention"
            )(h, mask=causal_mask, deterministic=deterministic)
            h = residual + h

            # --- MLP block + residual ---------------------------------------
            residual = h
            h_mlp = nn.LayerNorm(name="ln2")(h)
            h_mlp = nn.Dense(module.d_ff, name="fc1")(h_mlp)
            h_mlp = nn.gelu(h_mlp, approximate=False)
            h_mlp = nn.Dense(module.d_model, name="fc2")(h_mlp)
            h_mlp = nn.Dropout(rate=module.dropout_rate)(h_mlp, deterministic=deterministic)
            h = residual + h_mlp
            
            return h

        return _block(self, x)

# Example Usage (Conceptual)
# if __name__ == '__main__':
#     key = jax.random.PRNGKey(0)
#     batch_size, seq_len, d_model = 4, 64, 256
#     n_heads, d_ff = 4, 512
#     dropout_rate = 0.1
#
#     dummy_input = jnp.ones((batch_size, seq_len, d_model))
#
#     transformer_block = TinyTransformerBlock(
#         d_model=d_model,
#         n_heads=n_heads,
#         d_ff=d_ff,
#         dropout_rate=dropout_rate
#     )
#
#     # Check XLA flags for cuDNN FMHA if needed
#     # import os
#     # os.environ = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_cudnn_fmha=true'
#     # Note: Default for xla_gpu_enable_cudnn_fmha is false.
#     # NVIDIA sometimes recommends false if not using Transformer Engine. Test carefully.
#
#     variables = transformer_block.init(key, dummy_input, deterministic=True)
#     output = transformer_block.apply(variables, dummy_input, deterministic=True)
#
#     print("Input shape:", dummy_input.shape)
#     print("Output shape:", output.shape)
#     print("Using native JAX dot_product_attention (requested cuDNN backend).")
