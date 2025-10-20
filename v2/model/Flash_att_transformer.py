from __future__ import annotations

from typing import Optional
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from flash_attn_jax import flash_mha

from omegaconf import OmegaConf
CONFIG_PATH = Path(__file__).resolve().parent / "Config.yml"
Config = OmegaConf.load(CONFIG_PATH)

from jax import config as jax_config
jax_config.update("jax_default_matmul_precision", Config.compute_dtype)  

def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(q_or_k, sin, cos):
    return (q_or_k * cos) + (_rotate_every_two(q_or_k) * sin)

def apply_partial_rope(x, sin, cos, rot_dim):
    """Apply RoPE to the first `rot_dim` scalars of `x` (… H, D)."""
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


class NativeJaxSelfAttention(nn.Module):
    """Multi-head self-attention using FlashAttention (via flash_attn_jax)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = Config.compute_dtype
    rotary_dim: int = Config.rope_dim

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads
        assert (self.rotary_dim <= self.head_dim), "less than or equal to head_dim"
        assert (self.rotary_dim % 2 == 0), "rotary_dim must be even"
        # Enforce *vanilla* MHA (no grouped/multi-query): HK == H
        assert self.num_kv == self.num_heads, "vanilla attention only: set num_kv == num_heads"

        # Combined qkv projection (q_size + 2*kv_size)
        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim  # == 3 * qkv_features when num_kv == num_heads
        self.qkv_proj = nn.Dense(
            total_out,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            param_dtype=Config.param_dtype,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=Config.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        head_dim = self.head_dim
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv * head_dim

        qkv = self.qkv_proj(x)  # (b, l, q_size + 2*kv_size)
        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        # Vanilla MHA: K,V have H heads (no grouped query)
        k = k_chunk.reshape(b, l, self.num_heads, head_dim)
        v = v_chunk.reshape(b, l, self.num_heads, head_dim)

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.rotary_dim, 2) / self.rotary_dim))
        seq = jnp.array([cur_index]) if use_kv_cache else jnp.arange(l)
        angles = jnp.einsum('i,j->ij', seq, inv_freq)
        emb = jnp.repeat(angles, 2, axis=-1)

        sin, cos = jnp.sin(emb)[None, :, None, :], jnp.cos(emb)[None, :, None, :]
        sin = sin.astype(self.dtype); cos = cos.astype(self.dtype)

        q = apply_partial_rope(q, sin, cos, self.rotary_dim)
        k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        if use_kv_cache:
            assert cur_index is not None, "Need cur_index when use_kv_cache=True"
            cached_k = self.variable("cache", "k", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)
            cached_v = self.variable("cache", "v", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)

            # write current k/v into cache (k,v shape: (b, l, H, d) -- typically l==1 in streaming)
            cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k.squeeze(1))
            cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v.squeeze(1))

            # Build (b, key_len, H, d) and call FlashAttention with causal mask
            k_full = jnp.swapaxes(cached_k.value, 1, 2)
            v_full = jnp.swapaxes(cached_v.value, 1, 2)
            y_heads = flash_mha(q, k_full, v_full, is_causal=True)   # (b, l, H, d)
            y = y_heads.reshape(b, l, self.qkv_features)

        else:
            # Full-seq path (b, l, H, d) with causal mask via FlashAttention
            y_heads = flash_mha(q, k, v, is_causal=True)             # (b, l, H, d)
            y = y_heads.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y

class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = Config.compute_dtype

    @nn.compact
    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=self.dtype)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=Config.num_kv_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=self.dtype)(h)

            gate_dim = module.d_ff // 3
            proj_dim = gate_dim * 2

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=Config.param_dtype,
            )(h_norm)

            u, v = jnp.split(h_proj, 2, axis=-1)
            h_ffn = nn.silu(u) * v

            h_ffn = nn.Dense(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        return _block(self, x)
