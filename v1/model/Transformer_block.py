# Transformer_block.py – Native JAX cuDNN Flash‑Attention (Mixed Precision)

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
import Config # Import Config to access dtypes if needed, or pass explicitly

def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(q_or_k, sin, cos):
    # q_or_k:  (b, heads, L, d)
    # sin/cos: (L, d)
    #  broadcast sin/cos to batch & heads, apply rotation on last dim
    return (q_or_k * cos) + (_rotate_every_two(q_or_k) * sin)

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = Config.compute_dtype # Use compute_dtype

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        # Dense layers for QKV projections.
        # They will use compute_dtype (bf16) for compute,
        # but their params will be float32 (Flax default/can be set).
        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # self.k_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # self.v_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # 1-KV-head → much smaller cache
        self.k_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.v_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)


        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, *, deterministic: bool, decode: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        head_dim = self.qkv_features // self.num_heads

        q = self.q_proj(x).reshape(b, l, self.num_heads, head_dim)
        # k = self.k_proj(x).reshape(b, l, self.num_heads, head_dim)
        # v = self.v_proj(x).reshape(b, l, self.num_heads, head_dim)

        k = self.k_proj(x).reshape(b, l, self.num_kv, head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_kv, head_dim)

        k = jnp.repeat(k, self.num_heads // self.num_kv, axis=2)  # (B, L, H, D)
        v = jnp.repeat(v, self.num_heads // self.num_kv, axis=2)  # (B, L, H, D)

        rot_dim = head_dim
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, rot_dim, 2) / rot_dim))
        seq      = jnp.array([cur_index]) if decode else jnp.arange(l)
        angles   = jnp.einsum('i,j->ij', seq, inv_freq)           # (L, rot_dim/2)
        emb      = jnp.repeat(angles, 2, axis=-1)                 # (L, rot_dim)
        sin, cos = jnp.sin(emb).astype(self.dtype), jnp.cos(emb).astype(self.dtype)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]   # (1,L,1,D)
        q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)


        if decode:
            assert cur_index is not None, "Need cur_index when decode=True"
            cached_k = self.variable( "cache", "k", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)
            cached_v = self.variable( "cache", "v", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)

                # cached_k = self.variables["cache"]["k"]
                # cached_v = self.variables["cache"]["v"]

            cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k.squeeze(1))
            cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v.squeeze(1))
            # --- after you write k/v into the cache -------------------------------
            # k = cached_k.value                      # (B, H, T, D)
            # v = cached_v.value                      # (B, H, T, D)
            k = jnp.swapaxes(cached_k.value, 1, 2)  # (B, T, H, D)
            v = jnp.swapaxes(cached_v.value, 1, 2)  # (B, T, H, D)

            if False:
                q = q / jnp.sqrt(head_dim)
            # q = q.transpose(0, 2, 1, 3)             # (B, H, 1, D)

            # Build an additive bias: 0 for valid keys, –1e10 for padding keys
            # key_len   = k.shape[2]                  # == Config.context_length (static)
            key_len   = k.shape[1]                  # == Config.context_length (static)
            valid     = jnp.arange(key_len) <= cur_index       # (T,) dynamic mask
            # attn_bias = jnp.where(valid, 0.0, -1e10)
            attn_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)  # Ensure dtype matches
            attn_bias = attn_bias[None, None, None, :]          # (1,1,1,T)

            try:
                y = jax.nn.dot_product_attention(
                        q, k, v,
                        bias=attn_bias,         
                        is_causal=True,
                        # is_causal = not decode,  
                        # implementation="cudnn",
                        implementation="flash",  # may not work
                )
            except Exception:
                # Fallback to non-causal attention if flash fails
                y = jax.nn.dot_product_attention(
                    q, k, v,
                    bias=attn_bias,
                    is_causal=False,
                    implementation="cudnn",
                )

            # y = y.transpose(0, 2, 1, 3).reshape(b, 1, self.qkv_features)
            y = y.reshape(b, 1, self.qkv_features)

            # k = cached_k.value[:, :, : cur_index + 1, :]
            # v = cached_v.value[:, :, : cur_index + 1, :]
            # q = q / jnp.sqrt(head_dim)
            # q = q.transpose(0, 2, 1, 3)
            # y = jax.nn.dot_product_attention(q, k, v, is_causal=False, implementation="cudnn")
            # y = y.transpose(0, 2, 1, 3).reshape(b, 1, self.qkv_features)
        else:
            # Training path (unchanged)
            if False:
                q = q / jnp.sqrt(head_dim)

            y = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y
    # def __call__(
    #     self,
    #     x: jnp.ndarray, # Expects x to be in compute_dtype (bf16)
    #     *,
    #     deterministic: bool,
    # ) -> jnp.ndarray:
    #     b, l, _ = x.shape
    #
    #     # Project to Q‑K‑V and reshape to (B, T, H, D)
    #     # Inputs are bf16, projections compute in bf16, outputs are bf16.
    #     q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #     k = self.k_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #     v = self.v_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #
    #     # q, k, v are now bf16, satisfying the cuDNN requirement.
    #     y = jax.nn.dot_product_attention(
    #         q,
    #         k,
    #         v,
    #         bias=None,
    #         is_causal=True,
    #         implementation="cudnn",
    #     )
    #
    #     # Merge heads and apply output projection (still in bf16)
    #     y = y.reshape(b, l, -1)
    #     y = self.o_proj(y)
    #     y = self.dropout(y, deterministic=deterministic)
    #     return y


class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = Config.compute_dtype # Use compute_dtype

    @nn.compact
    # def __call__(self, x: jnp.ndarray, *, deterministic: bool = False):
    def __call__(self, x, *, deterministic: bool, decode: bool = False, cur_index: Optional[int] = None):
        # x is expected to be bf16
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            # --- Self‑Attention --------------------------------------------------
            residual = h # bf16
            # LayerNorm computes in f32, casts back to bf16
            # h_norm = nn.LayerNorm(name="ln1", dtype=jnp.float32)(h) 
            h_norm = RMSNorm(name="rms1", dtype=self.dtype)(h)
            # h_attn = NativeJaxSelfAttention(
            #     num_heads=module.n_heads,
            #     qkv_features=module.d_model,
            #     dropout_rate=module.dropout_rate,
            #     dtype=module.dtype, # Pass bf16
            # )(h_norm, deterministic=deterministic)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic, decode=decode, cur_index=cur_index)
            h = residual + h_attn # bf16

            # --- Feed‑forward ----------------------------------------------------
            residual = h # bf16
            # LayerNorm computes in f32, casts back to bf16
            # h_norm = nn.LayerNorm(name="ln2", dtype=jnp.float32)(h) 
            h_norm = RMSNorm(name="rms2", dtype=self.dtype)(h)
            # h_ffn = nn.Dense(module.d_ff, name="fc1", dtype=module.dtype, param_dtype=Config.param_dtype)(h_norm)
            # h_ffn = nn.gelu(h_ffn, approximate=False)
            # h_ffn = nn.Dense( module.d_ff * 2 // 3, name="fc1", dtype=module.dtype, param_dtype=Config.param_dtype)(h_norm)
            # h_ffn = nn.silu(h_ffn[:, :, :-module.d_ff//3]) * h_ffn[:, :, -module.d_ff//3:]

            gate_dim = module.d_ff // 3          # «⅓·d_ff» (rounded down)
            proj_dim = gate_dim * 2              # Always even ⇒ easier split

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=Config.param_dtype,
            )(h_norm)                            # shape: (..., 2·gate_dim)

            u, v = jnp.split(h_proj, 2, axis=-1) # each (..., gate_dim)
            h_ffn = nn.silu(u) * v               # shapes now match

            h_ffn = nn.Dense(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn # bf16

        return _block(self, x)
