from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

# If multi-GPU cuDNN issues reappear, refer to commit
# 9e75c6de7bac69414c68eb7c23342123ca2db50c.


def _to_dtype(value: jnp.dtype | str) -> jnp.dtype:
    if isinstance(value, jnp.dtype):
        return value
    if isinstance(value, str):
        try:
            return getattr(jnp, value)
        except AttributeError:
            return jnp.dtype(value)
    return jnp.dtype(value)


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
    # Duplicate the full frequency matrix (not each element) to form pairs.
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dtype: jnp.dtype | str
    param_dtype: jnp.dtype | str
    context_length: int = 2048
    dropout_rate: float = 0.0
    num_kv: int = 1
    enable_xsa: bool = False
    rotary_dim: Optional[int] = None
    causal: bool = True

    def setup(self):
        self._compute_dtype = _to_dtype(self.dtype)
        self._param_dtype = _to_dtype(self.param_dtype)
        if not bool(self.causal):
            raise ValueError("GIANT v3 attention only supports causal decoder mode")
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads
        assert (
            self.num_heads % self.num_kv == 0
        ), "num_heads must be divisible by num_kv_heads for grouped attention"

        rotary_dim = self.rotary_dim if self.rotary_dim is not None else self.head_dim
        self._rotary_dim = int(rotary_dim)
        assert self._rotary_dim <= self.head_dim, "rotary_dim must be <= head_dim"
        assert self._rotary_dim % 2 == 0, "rotary_dim must be even"

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(
            total_out,
            use_bias=False,
            name="qkv_proj",
            dtype=self._compute_dtype,
            param_dtype=self._param_dtype,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self._compute_dtype,
            param_dtype=self._param_dtype,
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        # Precompute rotary embeddings once and slice per call.
        self._rope_sin, self._rope_cos = _build_rope_cache(
            self.context_length, self._rotary_dim, self._compute_dtype
        )

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        use_kv_cache: bool = False,
        cur_index: Optional[jnp.ndarray | int] = None,
    ):
        b, l, _ = x.shape
        impl = "cudnn" if IS_GPU else "xla"

        head_dim = self.head_dim
        q_size   = self.num_heads * head_dim
        kv_size  = self.num_kv * head_dim

        qkv = self.qkv_proj(x)

        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        k = k_chunk.reshape(b, l, self.num_kv,  head_dim)
        v = v_chunk.reshape(b, l, self.num_kv,  head_dim)

        if use_kv_cache:
            assert cur_index is not None, "Need cur_index when use_kv_cache=True"
            cur_index = jnp.asarray(cur_index, jnp.int32)
            if cur_index.ndim == 0:
                sin = jax.lax.dynamic_slice(
                    self._rope_sin,
                    (0, cur_index, 0, 0),
                    (1, l, 1, self._rotary_dim),
                )
                cos = jax.lax.dynamic_slice(
                    self._rope_cos,
                    (0, cur_index, 0, 0),
                    (1, l, 1, self._rotary_dim),
                )
                sin = jnp.broadcast_to(sin, (b, l, 1, self._rotary_dim))
                cos = jnp.broadcast_to(cos, (b, l, 1, self._rotary_dim))
            else:
                positions = cur_index[:, None] + jnp.arange(l, dtype=jnp.int32)[None, :]
                sin = jnp.take(self._rope_sin[0], positions, axis=0)
                cos = jnp.take(self._rope_cos[0], positions, axis=0)
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self._rotary_dim)
        k = apply_partial_rope(k, sin, cos, self._rotary_dim)


        if use_kv_cache:
            assert cur_index is not None, "Need cur_index when use_kv_cache=True"
            cache_shape = (b, self.num_kv, self.context_length, head_dim)
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                cache_shape,
                self._compute_dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                cache_shape,
                self._compute_dtype,
            )

            k_to_cache = jnp.swapaxes(k, 1, 2)  # (b, num_kv, l, hd)
            v_to_cache = jnp.swapaxes(v, 1, 2)
            cur_index = jnp.asarray(cur_index, jnp.int32)
            is_scalar = (cur_index.ndim == 0)

            def _update_scalar(k_val, v_val, idx):
                if l == 1:
                    k_val = k_val.at[:, :, idx, :].set(k_to_cache[:, :, 0, :])
                    v_val = v_val.at[:, :, idx, :].set(v_to_cache[:, :, 0, :])
                else:
                    k_val = k_val.at[:, :, idx : idx + l, :].set(k_to_cache)
                    v_val = v_val.at[:, :, idx : idx + l, :].set(v_to_cache)
                return k_val, v_val

            def _update_vector(k_val, v_val, idx_vec):
                pos = idx_vec[:, None] + jnp.arange(l, dtype=jnp.int32)[None, :]
                batch_idx = jnp.arange(b)[:, None]
                k_to_cache_t = jnp.transpose(k_to_cache, (0, 2, 1, 3))
                v_to_cache_t = jnp.transpose(v_to_cache, (0, 2, 1, 3))
                k_val = k_val.at[batch_idx, :, pos, :].set(k_to_cache_t)
                v_val = v_val.at[batch_idx, :, pos, :].set(v_to_cache_t)
                return k_val, v_val

            if cur_index.ndim == 0:
                new_k, new_v = _update_scalar(cached_k.value, cached_v.value, cur_index)
                cur_max = cur_index + (l - 1)
                valid = jnp.arange(self.context_length) <= cur_max
                attn_bias = jnp.where(valid, 0.0, -1e10).astype(self._compute_dtype)
                attn_bias = attn_bias[None, None, None, :]
            else:
                new_k, new_v = _update_vector(cached_k.value, cached_v.value, cur_index)
                cur_max = cur_index + (l - 1)
                valid = jnp.arange(self.context_length)[None, :] <= cur_max[:, None]
                attn_bias = jnp.where(valid, 0.0, -1e10).astype(self._compute_dtype)
                attn_bias = attn_bias[:, None, None, :]

            cached_k.value = new_k
            cached_v.value = new_v

            k_full = jnp.swapaxes(cached_k.value, 1, 2)  # (b, context, num_kv, hd)
            v_full = jnp.swapaxes(cached_v.value, 1, 2)
            y = jax.nn.dot_product_attention(
                q, k_full, v_full, bias=attn_bias, is_causal=False, implementation=impl
            )
            if self.enable_xsa:
                v_proj = v
                if self.num_heads != self.num_kv:
                    repeat = self.num_heads // self.num_kv
                    v_proj = jnp.repeat(v_proj, repeat, axis=2)
                v_proj = v_proj / jnp.sqrt(
                    jnp.sum(jnp.square(v_proj), axis=-1, keepdims=True)
                    + jnp.asarray(1e-6, dtype=v_proj.dtype)
                )
                y = y - jnp.sum(y * v_proj, axis=-1, keepdims=True) * v_proj
            y = y.reshape(b, l, self.qkv_features)

        else:
            k_full = k
            v_full = v
            y = jax.nn.dot_product_attention(
                q,
                k_full,
                v_full,
                bias=None,
                is_causal=True,
                implementation=impl,
            )
            if self.enable_xsa:
                v_proj = v
                if self.num_heads != self.num_kv:
                    repeat = self.num_heads // self.num_kv
                    v_proj = jnp.repeat(v_proj, repeat, axis=2)
                v_proj = v_proj / jnp.sqrt(
                    jnp.sum(jnp.square(v_proj), axis=-1, keepdims=True)
                    + jnp.asarray(1e-6, dtype=v_proj.dtype)
                )
                y = y - jnp.sum(y * v_proj, axis=-1, keepdims=True) * v_proj
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Pre-norm Transformer block with RMSNorm + native JAX attention."""

    d_model: int
    n_heads: int
    d_ff: int
    dtype: jnp.dtype | str
    param_dtype: jnp.dtype | str
    context_length: int = 2048
    dropout_rate: float = 0.1
    num_kv_heads: Optional[int] = None
    rotary_dim: Optional[int] = None
    use_remat: bool = False
    enable_xsa: bool = False
    causal: bool = True

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        use_kv_cache: bool = False,
        cur_index: Optional[jnp.ndarray | int] = None,
    ):
        compute_dtype = _to_dtype(self.dtype)
        param_dtype = _to_dtype(self.param_dtype)
        num_kv = self.num_kv_heads if self.num_kv_heads is not None else self.n_heads

        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=compute_dtype, epsilon=1e-5)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=num_kv,
                qkv_features=module.d_model,
                context_length=module.context_length,
                dropout_rate=module.dropout_rate,
                dtype=compute_dtype,
                param_dtype=param_dtype,
                rotary_dim=module.rotary_dim,
                enable_xsa=module.enable_xsa,
                causal=module.causal,
            )(
                h_norm,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
            )
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=compute_dtype, epsilon=1e-5)(h)

            # Standard SwiGLU: project to 2 * d_ff, split, SiLU gate
            gate_dim = module.d_ff
            proj_dim = gate_dim * 2

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=compute_dtype,
                param_dtype=param_dtype,
                use_bias=False,
            )(h_norm)

            u, v = jnp.split(h_proj, 2, axis=-1)
            h_gate = nn.silu(u)
            h_ffn = h_gate * v

            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=compute_dtype,
                param_dtype=param_dtype,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        block_fn = nn.remat(_block) if self.use_remat else _block
        return block_fn(self, x)
