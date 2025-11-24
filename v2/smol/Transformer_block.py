from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from pathlib import Path

from omegaconf import OmegaConf


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
jax_config.update("jax_default_matmul_precision", MODEL_CFG.compute_dtype)  

IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())

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


def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.repeat(angles, 2, axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = COMPUTE_DTYPE
    rotary_dim: int = MODEL_CFG.rope_dim

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads
        assert (
            self.num_heads % self.num_kv == 0
        ), "num_heads must be divisible by num_kv_heads for grouped attention"
        assert(self.rotary_dim <= self.head_dim), "less than or equal to head_dim"
        assert(self.rotary_dim % 2 == 0), "rotary_dim must be even"

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(
            total_out,
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
        # Precompute rotary embeddings once and slice per call.
        self._rope_sin, self._rope_cos = _build_rope_cache(
            MODEL_CFG.context_length, self.rotary_dim, self.dtype
        )

    @nn.compact
    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        # Flash attention (cuDNN) supports bias only when sequence length is even; fall back otherwise.
        impl = "cudnn" if (IS_GPU and l >= 128 and l % 2 == 0) else "xla"

        head_dim = self.head_dim
        q_size   = self.num_heads * head_dim
        kv_size  = self.num_kv * head_dim

        qkv = self.qkv_proj(x)

        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        k = k_chunk.reshape(b, l, self.num_kv,  head_dim)
        v = v_chunk.reshape(b, l, self.num_kv,  head_dim)

        group = max(1, self.num_heads // self.num_kv)
        kv_indices = None
        if self.num_kv != self.num_heads:
            kv_indices = jnp.arange(self.num_heads) // group

        if use_kv_cache:
            sin = jax.lax.dynamic_slice(
                self._rope_sin,
                (0, cur_index, 0, 0),
                (1, 1, 1, self.rotary_dim),
            )
            cos = jax.lax.dynamic_slice(
                self._rope_cos,
                (0, cur_index, 0, 0),
                (1, 1, 1, self.rotary_dim),
            )
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self.rotary_dim)
        k = apply_partial_rope(k, sin, cos, self.rotary_dim)


        if use_kv_cache:
            assert cur_index is not None, "Need cur_index when use_kv_cache=True"
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, head_dim),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, head_dim),
                self.dtype,
            )


            k_to_cache = jnp.swapaxes(k, 1, 2)  # (b, num_kv, l, hd)
            v_to_cache = jnp.swapaxes(v, 1, 2)
            if l == 1:
                cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k_to_cache[:, :, 0, :])
                cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v_to_cache[:, :, 0, :])
            else:
                cached_k.value = cached_k.value.at[:, :, cur_index : cur_index + l, :].set(k_to_cache)
                cached_v.value = cached_v.value.at[:, :, cur_index : cur_index + l, :].set(v_to_cache)

            k_full = jnp.swapaxes(cached_k.value, 1, 2)  # (b, context, num_kv, hd)
            v_full = jnp.swapaxes(cached_v.value, 1, 2)
            if kv_indices is not None:
                k_full = jnp.take(k_full, kv_indices, axis=2)
                v_full = jnp.take(v_full, kv_indices, axis=2)
            key_len = k_full.shape[1]
            cur_max = cur_index + (l - 1)
            valid = jnp.arange(key_len) <= cur_max
            attn_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)
            attn_bias = attn_bias[None, None, None, :]
            y = jax.nn.dot_product_attention(
                q, k_full, v_full, bias=attn_bias, is_causal=False, implementation=impl
            )
            y = y.reshape(b, l, self.qkv_features)

        else:
            if False:
                q = q / jnp.sqrt(head_dim)

            k_full = k if kv_indices is None else jnp.take(k, kv_indices, axis=2)
            v_full = v if kv_indices is None else jnp.take(v, kv_indices, axis=2)

            y = jax.nn.dot_product_attention(q, k_full, v_full, is_causal=True, implementation=impl)
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x, *, deterministic: bool, use_kv_cache: bool = False, cur_index: Optional[int] = None):
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=self.dtype, epsilon=1e-5)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=MODEL_CFG.num_kv_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=self.dtype, epsilon=1e-5)(h)

            # Standard SwiGLU: project to 2 * d_ff, split, SiLU gate
            gate_dim = module.d_ff
            proj_dim = gate_dim * 2

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_norm)

            u, v = jnp.split(h_proj, 2, axis=-1)
            h_gate = nn.silu(u)
            h_ffn = h_gate * v

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
        block_fn = nn.remat(_block) if use_remat else _block
        return block_fn(self, x)
