from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
from omegaconf import OmegaConf
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(MODEL_DIR / "Config.yml"),
)
MODEL_CFG = cfg.model


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(MODEL_CFG.compute_dtype)

from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", str(MODEL_CFG.compute_dtype))

IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())


def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_partial_rope(x, sin, cos, rot_dim):
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


def _get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    name = str(name).lower()
    if name == "silu":
        return nn.silu
    if name == "gelu":
        return nn.gelu
    if name == "relu":
        return nn.relu
    raise ValueError(f"Unknown activation: {name}")


class NativeJaxSelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    context_length: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    rotary_dim: int = int(MODEL_CFG.rope_dim)

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        self.head_dim = self.qkv_features // self.num_heads
        if self.rotary_dim < 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rope_dim must be in [0, head_dim]")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rope_dim must be even")

        self.qkv_proj = nn.Dense(
            3 * self.qkv_features,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        self._rope_sin, self._rope_cos = _build_rope_cache(int(self.context_length), self.rotary_dim, self.dtype)

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        b, l, _ = x.shape
        impl = "cudnn" if (IS_GPU and l >= 128 and l % 2 == 0) else "xla"

        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, l, self.num_heads, self.head_dim)
        k = k.reshape(b, l, self.num_heads, self.head_dim)
        v = v.reshape(b, l, self.num_heads, self.head_dim)

        if self.rotary_dim > 0:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]
            q = apply_partial_rope(q, sin, cos, self.rotary_dim)
            k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        y = jax.nn.dot_product_attention(q, k, v, is_causal=False, implementation=impl)
        y = y.reshape(b, l, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TokenMixMLP(nn.Module):
    context_length: int
    hidden_size: int
    dropout_rate: float = 0.0
    activation: str = str(MODEL_CFG.activation)
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        b, l, d = x.shape
        if l != self.context_length:
            raise ValueError(
                f"TokenMixMLP requires fixed length L={self.context_length}, got L={l}"
            )
        act = _get_activation(self.activation)

        h = jnp.swapaxes(x, 1, 2)  # (b, d, l)
        h = nn.Dense(
            self.hidden_size,
            name="tm1",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            use_bias=False,
        )(h)
        h = act(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        h = nn.Dense(
            self.context_length,
            name="tm2",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            use_bias=False,
        )(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return jnp.swapaxes(h, 1, 2)


class TinyTRMLayer(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    context_length: int
    variant: str = "attn"  # attn | mlp
    rotary_dim: int = int(MODEL_CFG.rope_dim)
    mixer_hidden: int = int(MODEL_CFG.mixer_hidden)
    dropout_rate: float = 0.0
    activation: str = str(MODEL_CFG.activation)
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        def _layer(module: "TinyTRMLayer", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=module.dtype, epsilon=1e-5)(h)

            variant = str(module.variant).lower()
            if variant == "attn":
                h_mix = NativeJaxSelfAttention(
                    num_heads=module.n_heads,
                    qkv_features=module.d_model,
                    context_length=module.context_length,
                    dropout_rate=module.dropout_rate,
                    dtype=module.dtype,
                    rotary_dim=module.rotary_dim,
                )(h_norm, deterministic=deterministic)
            elif variant == "mlp":
                h_mix = TokenMixMLP(
                    context_length=module.context_length,
                    hidden_size=module.mixer_hidden,
                    dropout_rate=module.dropout_rate,
                    activation=module.activation,
                    dtype=module.dtype,
                )(h_norm, deterministic=deterministic)
            else:
                raise ValueError(f"Unknown TRM variant: {module.variant}")

            h = residual + h_mix

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=module.dtype, epsilon=1e-5)(h)

            act = _get_activation(module.activation)
            proj_dim = module.d_ff * 2
            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_norm)
            u, v = jnp.split(h_proj, 2, axis=-1)
            h_ffn = act(u) * v
            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        use_remat = bool(getattr(MODEL_CFG, "use_remat", False))
        layer_fn = nn.remat(_layer) if use_remat else _layer
        return layer_fn(self, x)


class TinyRecurrentNet(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    context_length: int
    variant: str = str(MODEL_CFG.variant)
    rotary_dim: int = int(MODEL_CFG.rope_dim)
    mixer_hidden: int = int(MODEL_CFG.mixer_hidden)
    dropout_rate: float = 0.0
    activation: str = str(MODEL_CFG.activation)
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        h = x
        for i in range(self.n_layers):
            h = TinyTRMLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                context_length=self.context_length,
                variant=self.variant,
                rotary_dim=self.rotary_dim,
                mixer_hidden=self.mixer_hidden,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                dtype=self.dtype,
                name=f"layer_{i}",
            )(h, deterministic=deterministic)
        h = RMSNorm(name="rms_out", dtype=self.dtype, epsilon=1e-5)(h)
        return h
