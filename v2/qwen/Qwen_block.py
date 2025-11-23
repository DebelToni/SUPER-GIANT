from __future__ import annotations

from typing import Optional
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
from omegaconf import OmegaConf


QWEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(QWEN_DIR / "Config.yml"),
)
MODEL_CFG = cfg.model


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(MODEL_CFG.compute_dtype)

RMS_EPS = float(getattr(MODEL_CFG, "rms_norm_eps", 1e-6))


def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_partial_rope(x, sin, cos, rot_dim):
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _build_rope_cache(seq_len: int, rotary_dim: int, theta: float, dtype: jnp.dtype):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.repeat(angles, 2, axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


class QwenAttention(nn.Module):
    num_heads: int
    num_kv: int
    head_dim: int
    dropout_rate: float = 0.0
    rope_dim: int = 0
    rope_theta: float = 1e6
    dtype: jnp.dtype = COMPUTE_DTYPE

    def setup(self):
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv * self.head_dim

        self.q_proj = nn.Dense(
            q_out,
            use_bias=True,
            name="q_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.k_proj = nn.Dense(
            kv_out,
            use_bias=True,
            name="k_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.v_proj = nn.Dense(
            kv_out,
            use_bias=True,
            name="v_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.o_proj = nn.Dense(
            q_out,
            use_bias=True,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self._rope_sin, self._rope_cos = _build_rope_cache(
            MODEL_CFG.context_length, self.rope_dim, self.rope_theta, self.dtype
        )

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        # x: [B, L, d_model]
        b, l, _ = x.shape

        # Projections -> [B, L, heads, head_dim]
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.num_kv,   self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_kv,   self.head_dim)

        # ---- RoPE ----
        # _rope_sin/_rope_cos: [1, context_length, 1, rope_dim]
        if use_kv_cache:
            assert cur_index is not None, "cur_index required with kv cache"
            # In your pipeline, L == 1 when use_kv_cache=True, but this works for L > 1 too.
            sin = jax.lax.dynamic_slice(
                self._rope_sin, (0, cur_index, 0, 0), (1, l, 1, self.rope_dim)
            )
            cos = jax.lax.dynamic_slice(
                self._rope_cos, (0, cur_index, 0, 0), (1, l, 1, self.rope_dim)
            )
        else:
            sin = self._rope_sin[:, :l, :, :]  # [1, L, 1, rope_dim]
            cos = self._rope_cos[:, :l, :, :]

        # q,k: [B, L, heads, head_dim] / [B, L, kv_heads, head_dim]
        q = apply_partial_rope(q, sin, cos, self.rope_dim)
        k = apply_partial_rope(k, sin, cos, self.rope_dim)

        # ---- KV cache ----
        if use_kv_cache:
            # Cache layout: [B, S, K, H] to match jax.nn.dot_product_attention
            cache_shape = (b, MODEL_CFG.context_length, self.num_kv, self.head_dim)

            cached_k = self.variable(
                "cache", "k", jnp.zeros, cache_shape, self.dtype
            )
            cached_v = self.variable(
                "cache", "v", jnp.zeros, cache_shape, self.dtype
            )

            # Write current block [B, L, K, H] into sequence axis at cur_index
            cached_k.value = jax.lax.dynamic_update_slice(
                cached_k.value, k, (0, cur_index, 0, 0)
            )
            cached_v.value = jax.lax.dynamic_update_slice(
                cached_v.value, v, (0, cur_index, 0, 0)
            )

            k_full = cached_k.value  # [B, S, K, H]
            v_full = cached_v.value

            key_len = k_full.shape[1]
            cur_max = cur_index + (l - 1)
            valid = jnp.arange(key_len) <= cur_max
            bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)  # [S]
            bias = bias[None, None, None, :]  # [1, 1, 1, S]

            # JAX handles GQA/MQA: q: [B,T,N,H], k/v: [B,S,K,H]
            y = jax.nn.dot_product_attention(
                q, k_full, v_full, bias=bias, is_causal=False
            )
        else:
            # No cache: just self-attention with full sequence
            # q: [B,L,N,H], k/v: [B,L,K,H]
            y = jax.nn.dot_product_attention(
                q, k, v, is_causal=True
            )

        # y: [B, L, N, H] -> [B, L, d_model]
        y = y.reshape(b, l, self.num_heads * self.head_dim)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class QwenMLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        gate = nn.Dense(
            self.intermediate_size,
            use_bias=False,
            name="gate_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )(x)
        up = nn.Dense(
            self.intermediate_size,
            use_bias=False,
            name="up_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )(x)
        hidden = nn.silu(gate) * up
        down = nn.Dense(
            self.hidden_size,
            use_bias=False,
            name="down_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )(hidden)
        down = nn.Dropout(rate=self.dropout_rate)(down, deterministic=deterministic)
        return down


class QwenBlock(nn.Module):
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    dropout_rate: float = 0.0
    rope_dim: int = 64
    rope_theta: float = 1e6
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        head_dim = self.d_model // self.n_heads
        # Attention
        residual = x
        h = RMSNorm(epsilon=RMS_EPS, dtype=self.dtype, name="rms1")(x)
        h = QwenAttention(
            num_heads=self.n_heads,
            num_kv=self.n_kv_heads,
            head_dim=head_dim,
            dropout_rate=self.dropout_rate,
            rope_dim=self.rope_dim,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
            name="attn",
        )(h, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)
        x = residual + h

        # MLP
        residual = x
        h = RMSNorm(epsilon=RMS_EPS, dtype=self.dtype, name="rms2")(x)
        h = QwenMLP(
            hidden_size=self.d_model,
            intermediate_size=self.d_ff,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="mlp",
        )(h, deterministic=deterministic)
        return residual + h
