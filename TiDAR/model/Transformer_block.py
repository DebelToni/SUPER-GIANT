from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

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
    """Multi-head self-attention using jax.nn.dot_product_attention."""

    num_heads: int
    qkv_features: int
    context_length: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    rotary_dim: int = 64
    draft_len: int = 0

    def setup(self):
        assert self.qkv_features % self.num_heads == 0, "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads
        assert self.num_heads % self.num_kv == 0, "num_heads must be divisible by num_kv_heads"
        assert self.rotary_dim <= self.head_dim, "rotary_dim must be <= head_dim"
        assert self.rotary_dim % 2 == 0, "rotary_dim must be even"

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(
            total_out,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        rope_len = int(self.context_length) + (2 * int(self.draft_len))
        self._rope_sin, self._rope_cos = _build_rope_cache(
            rope_len, self.rotary_dim, self.dtype
        )

    def _rope_from_position_ids(self, position_ids: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        pos = position_ids.astype(jnp.int32)
        sin_base = self._rope_sin[0, :, 0, :]
        cos_base = self._rope_cos[0, :, 0, :]
        sin = jnp.take(sin_base, pos, axis=0)
        cos = jnp.take(cos_base, pos, axis=0)
        sin = sin[:, :, None, :]
        cos = cos[:, :, None, :]
        return sin, cos

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        attn_bias: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        write_to_cache: bool = True,
        prefix_len: Optional[int] = None,
        cache_write_len: Optional[int] = None,
        kv_cache_len: Optional[int] = None,
    ):
        b, l, _ = x.shape
        use_cudnn = IS_GPU and l % 2 == 0 and l >= 2
        impl = "cudnn" if use_cudnn else "xla"

        head_dim = self.head_dim
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv * head_dim

        qkv = self.qkv_proj(x)
        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        k = k_chunk.reshape(b, l, self.num_kv, head_dim)
        v = v_chunk.reshape(b, l, self.num_kv, head_dim)

        group = max(1, self.num_heads // self.num_kv)
        kv_indices = None
        if self.num_kv != self.num_heads:
            kv_indices = jnp.arange(self.num_heads) // group

        if position_ids is not None:
            sin, cos = self._rope_from_position_ids(position_ids)
        elif use_kv_cache:
            assert cur_index is not None, "Need cur_index when use_kv_cache=True"
            sin = jax.lax.dynamic_slice(
                self._rope_sin,
                (0, cur_index, 0, 0),
                (1, l, 1, self.rotary_dim),
            )
            cos = jax.lax.dynamic_slice(
                self._rope_cos,
                (0, cur_index, 0, 0),
                (1, l, 1, self.rotary_dim),
            )
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self.rotary_dim)
        k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        if use_kv_cache:
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, self.num_kv, self.context_length, head_dim),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, self.num_kv, self.context_length, head_dim),
                self.dtype,
            )

            k_to_cache = jnp.swapaxes(k, 1, 2)
            v_to_cache = jnp.swapaxes(v, 1, 2)

            if write_to_cache:
                # Standard KV-cache path: append tokens at cur_index with causal bias.
                assert cur_index is not None, "Need cur_index when use_kv_cache=True"
                start = (0, 0, jnp.asarray(cur_index, dtype=jnp.int32), 0)
                cached_k.value = jax.lax.dynamic_update_slice(cached_k.value, k_to_cache, start)
                cached_v.value = jax.lax.dynamic_update_slice(cached_v.value, v_to_cache, start)

                k_full = jnp.swapaxes(cached_k.value, 1, 2)
                v_full = jnp.swapaxes(cached_v.value, 1, 2)
                if kv_cache_len is not None:
                    k_full = k_full[:, :kv_cache_len, :, :]
                    v_full = v_full[:, :kv_cache_len, :, :]
                if kv_indices is not None:
                    k_full = jnp.take(k_full, kv_indices, axis=2)
                    v_full = jnp.take(v_full, kv_indices, axis=2)
                key_len = k_full.shape[1]
                q_abs = cur_index + jnp.arange(l)
                k_abs = jnp.arange(key_len)
                valid = k_abs[None, :] <= q_abs[:, None]
                base_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)
                base_bias = base_bias[None, None, :, :]
                if attn_bias is not None:
                    base_bias = base_bias + attn_bias.astype(self.dtype)
                y = jax.nn.dot_product_attention(
                    q, k_full, v_full, bias=base_bias, is_causal=False, implementation=impl
                )
                y = y.reshape(b, l, self.qkv_features)
            else:
                # TiDAR decode path: keep prefix cache fixed and apply structured attn_bias.
                assert prefix_len is not None, "prefix_len is required when write_to_cache=False"
                assert attn_bias is not None, "attn_bias is required when write_to_cache=False"

                # Optionally commit a subset of step tokens into the cache.
                if cache_write_len is not None and cache_write_len > 0:
                    write_index = jnp.asarray(prefix_len, dtype=jnp.int32)
                    k_update = k_to_cache[:, :, :cache_write_len, :]
                    v_update = v_to_cache[:, :, :cache_write_len, :]
                    start = (0, 0, write_index, 0)
                    cached_k.value = jax.lax.dynamic_update_slice(cached_k.value, k_update, start)
                    cached_v.value = jax.lax.dynamic_update_slice(cached_v.value, v_update, start)

                k_prefix = cached_k.value
                v_prefix = cached_v.value
                if kv_cache_len is not None:
                    k_prefix = k_prefix[:, :, :kv_cache_len, :]
                    v_prefix = v_prefix[:, :, :kv_cache_len, :]
                prefix_capacity = k_prefix.shape[2]
                k_prefix = jnp.swapaxes(k_prefix, 1, 2)
                v_prefix = jnp.swapaxes(v_prefix, 1, 2)
                if kv_indices is not None:
                    k_prefix = jnp.take(k_prefix, kv_indices, axis=2)
                    v_prefix = jnp.take(v_prefix, kv_indices, axis=2)

                k_step = k if kv_indices is None else jnp.take(k, kv_indices, axis=2)
                v_step = v if kv_indices is None else jnp.take(v, kv_indices, axis=2)
                prefix_len_safe = jnp.minimum(jnp.asarray(prefix_len, dtype=jnp.int32), prefix_capacity)
                prefix_valid = jnp.arange(prefix_capacity) < prefix_len_safe
                prefix_bias_row = jnp.where(prefix_valid, 0.0, -1e10).astype(self.dtype)
                prefix_bias_row = prefix_bias_row[None, None, None, :]
                is_full_width_bias = (attn_bias.ndim == 4) and (attn_bias.shape[-2] == l) and (attn_bias.shape[-1] == (prefix_capacity + l))
                is_step_only_bias = (attn_bias.ndim == 4) and (attn_bias.shape[-2] == l) and (attn_bias.shape[-1] == l)

                if is_full_width_bias:
                    # Detect key layout from bias pattern:
                    # - step-prefix layout has blocked entries inside the first l key columns.
                    # - prefix-step layout keeps the first cache columns fully allowed.
                    is_step_prefix_layout = jnp.any(attn_bias[:, :, :, :l] < 0.0)

                    def _step_prefix_attn(_):
                        # Layout: [STEP | PREFIX]. Valid keys are contiguous:
                        # q_len + prefix_len.
                        k_full = jnp.concatenate([k_step, k_prefix], axis=1)
                        v_full = jnp.concatenate([v_step, v_prefix], axis=1)
                        kv_len = jnp.minimum(
                            prefix_len_safe + jnp.asarray(l, dtype=jnp.int32),
                            jnp.asarray(k_full.shape[1], dtype=jnp.int32),
                        )
                        key_value_seq_lengths = jnp.full((b,), kv_len, dtype=jnp.int32)
                        return jax.nn.dot_product_attention(
                            q,
                            k_full,
                            v_full,
                            bias=attn_bias.astype(self.dtype),
                            is_causal=False,
                            key_value_seq_lengths=key_value_seq_lengths,
                            implementation=impl,
                        )

                    def _prefix_step_attn(_):
                        # Legacy layout: [PREFIX | STEP]. Needs explicit prefix-valid bias.
                        k_full = jnp.concatenate([k_prefix, k_step], axis=1)
                        v_full = jnp.concatenate([v_prefix, v_step], axis=1)
                        step_bias = jnp.zeros((1, 1, 1, l), dtype=self.dtype)
                        key_valid_bias = jnp.concatenate([prefix_bias_row, step_bias], axis=-1)
                        full_bias = attn_bias.astype(self.dtype) + key_valid_bias
                        return jax.nn.dot_product_attention(
                            q,
                            k_full,
                            v_full,
                            bias=full_bias,
                            is_causal=False,
                            implementation=impl,
                        )

                    y = jax.lax.cond(is_step_prefix_layout, _step_prefix_attn, _prefix_step_attn, operand=None)
                    y = y.reshape(b, l, self.qkv_features)
                elif is_step_only_bias:
                    # Step-only bias path: compute prefix and step attentions separately,
                    # then merge exactly via logsumexp residuals.
                    has_prefix = prefix_len_safe > 0
                    prefix_bias = jnp.broadcast_to(prefix_bias_row, (1, 1, l, prefix_capacity))

                    def _prefix_attn(_):
                        return jax.nn.dot_product_attention(
                            q,
                            k_prefix,
                            v_prefix,
                            bias=prefix_bias,
                            is_causal=False,
                            implementation=impl,
                            return_residual=True,
                        )

                    def _prefix_empty(_):
                        y0 = jnp.zeros_like(q)
                        lse0 = jnp.full(q.shape[:-1], -jnp.inf, dtype=q.dtype)
                        return y0, lse0

                    y_prefix, lse_prefix = jax.lax.cond(has_prefix, _prefix_attn, _prefix_empty, operand=None)
                    y_step, lse_step = jax.nn.dot_product_attention(
                        q,
                        k_step,
                        v_step,
                        bias=attn_bias.astype(self.dtype),
                        is_causal=False,
                        implementation=impl,
                        return_residual=True,
                    )

                    lse_prefix = lse_prefix.astype(jnp.float32)
                    lse_step = lse_step.astype(jnp.float32)
                    lse = jnp.logaddexp(lse_prefix, lse_step)
                    finite_lse = jnp.isfinite(lse)

                    w_prefix = jnp.where(finite_lse, jnp.exp(lse_prefix - lse), 0.0).astype(self.dtype)
                    w_step = jnp.where(finite_lse, jnp.exp(lse_step - lse), 0.0).astype(self.dtype)
                    y = (w_prefix[..., None] * y_prefix) + (w_step[..., None] * y_step)
                    y = y.reshape(b, l, self.qkv_features)
                else:
                    raise ValueError(f"Unexpected attn_bias shape for decode path: {attn_bias.shape}")
        else:
            k_full = k if kv_indices is None else jnp.take(k, kv_indices, axis=2)
            v_full = v if kv_indices is None else jnp.take(v, kv_indices, axis=2)
            if attn_bias is None:
                y = jax.nn.dot_product_attention(q, k_full, v_full, is_causal=True, implementation=impl)
            else:
                y = jax.nn.dot_product_attention(
                    q, k_full, v_full, bias=attn_bias.astype(self.dtype), is_causal=False, implementation=impl
                )
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder-style transformer block (GPT)."""

    d_model: int
    n_heads: int
    d_ff: int
    num_kv_heads: int
    rope_dim: int
    context_length: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    draft_len: int = 0
    use_remat: bool = False

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        attn_bias: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        write_to_cache: bool = True,
        prefix_len: Optional[int] = None,
        cache_write_len: Optional[int] = None,
        kv_cache_len: Optional[int] = None,
    ):
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=self.dtype, epsilon=1e-5)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=module.num_kv_heads,
                qkv_features=module.d_model,
                context_length=module.context_length,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
                param_dtype=module.param_dtype,
                rotary_dim=module.rope_dim,
                draft_len=module.draft_len,
            )(
                h_norm,
                deterministic=deterministic,
                attn_bias=attn_bias,
                position_ids=position_ids,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                write_to_cache=write_to_cache,
                prefix_len=prefix_len,
                cache_write_len=cache_write_len,
                kv_cache_len=kv_cache_len,
            )
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=self.dtype, epsilon=1e-5)(h)

            gate_dim = module.d_ff
            proj_dim = gate_dim * 2

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=module.param_dtype,
                use_bias=False,
            )(h_norm)

            u, v = jnp.split(h_proj, 2, axis=-1)
            h_gate = nn.silu(u)
            h_ffn = h_gate * v

            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=module.dtype,
                param_dtype=module.param_dtype,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        block_fn = nn.remat(_block) if self.use_remat else _block
        return block_fn(self, x)
