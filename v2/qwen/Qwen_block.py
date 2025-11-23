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
    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.num_kv, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_kv, self.head_dim)

        if use_kv_cache:
            sin = jax.lax.dynamic_slice(
                self._rope_sin, (0, cur_index, 0, 0), (1, 1, 1, self.rope_dim)
            )
            cos = jax.lax.dynamic_slice(
                self._rope_cos, (0, cur_index, 0, 0), (1, 1, 1, self.rope_dim)
            )
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self.rope_dim)
        k = apply_partial_rope(k, sin, cos, self.rope_dim)

        group = max(1, self.num_heads // self.num_kv)

        if use_kv_cache:
            assert cur_index is not None, "cur_index required with kv cache"
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, self.head_dim),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, self.head_dim),
                self.dtype,
            )

            k_to_cache = jnp.swapaxes(k, 1, 2)
            v_to_cache = jnp.swapaxes(v, 1, 2)

            cached_k.value = jax.lax.dynamic_update_slice(
                cached_k.value, k_to_cache, (0, 0, cur_index, 0)
            )
            cached_v.value = jax.lax.dynamic_update_slice(
                cached_v.value, v_to_cache, (0, 0, cur_index, 0)
            )

            k_full = jnp.swapaxes(cached_k.value, 1, 2)  # [b, ctx, n_kv, hd]
            v_full = jnp.swapaxes(cached_v.value, 1, 2)
            k_full = jnp.repeat(k_full, repeats=group, axis=2)  # -> [b, ctx, n_heads, hd]
            v_full = jnp.repeat(v_full, repeats=group, axis=2)
            key_len = k_full.shape[1]
            cur_max = cur_index + (l - 1)
            valid = jnp.arange(key_len) <= cur_max
            bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)
            bias = bias[None, None, None, :]
            y = jax.nn.dot_product_attention(q, k_full, v_full, bias=bias, is_causal=False)
        else:
            k_full = jnp.repeat(k, repeats=group, axis=2)
            v_full = jnp.repeat(v, repeats=group, axis=2)
            y = jax.nn.dot_product_attention(q, k_full, v_full, is_causal=True)

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
