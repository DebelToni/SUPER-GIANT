from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

# ------------------------------- Config ------------------------------------ #


@dataclass(frozen=True)
class ModelConfig:
    # Model sizes
    vocab_size: int = 32000
    context_length: int = 512
    embedding_size: int = 1280
    num_heads: int = 20
    num_kv_heads: int = 10
    num_layers: int = 28
    feed_forward_size: int = 7680
    rope_dim: int = 64
    dropout_rate: float = 0.0
    use_remat: bool = False
    # Dtypes (bf16 end-to-end)
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"


MODEL_CFG = ModelConfig()

# DeepSeek-style MLA defaults (reasonable compression starting point)
# KV cache per token shrinks from 2*head_dim to (latent+rope) + latent
MLA_LATENT_DIM_DEFAULT = 32
MLA_ROPE_DIM_DEFAULT = 16


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(MODEL_CFG.compute_dtype)

from jax import config as jax_config  # noqa: E402

jax_config.update("jax_default_matmul_precision", MODEL_CFG.compute_dtype)

IS_GPU = (jax.default_backend() == "gpu")


def choose_attention_impl(force_xla: bool = False) -> str:
    # Default: cudnn on GPU, xla elsewhere.
    if force_xla:
        return "xla"
    return "cudnn" if IS_GPU else "xla"


# ----------------------------- RoPE helpers -------------------------------- #


def _rotate_every_two(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_partial_rope(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray, rot_dim: int) -> jnp.ndarray:
    """Apply RoPE to the first rot_dim scalars of x (..., H, D)."""
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)  # (1, S, 1, rot_dim)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


# ---------------------------- Attention ------------------------------------ #


class NativeJaxSelfAttention(nn.Module):
    """
    Multi-head self-attention using jax.nn.dot_product_attention.

    Uses native GQA: Query heads = N, Key/Value heads = K (K can differ from N) WITHOUT duplicating KV.
    """

    num_heads: int
    qkv_features: int
    num_kv: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    rotary_dim: int = MODEL_CFG.rope_dim
    attn_impl: str = "cudnn"  # "cudnn" or "xla"

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        if self.num_heads % self.num_kv != 0:
            raise ValueError("num_heads must be divisible by num_kv")
        self.head_dim = self.qkv_features // self.num_heads
        if self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be <= head_dim")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")

        total_out = (self.num_heads * self.head_dim) + 2 * (self.num_kv * self.head_dim)

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

        self._rope_sin, self._rope_cos = _build_rope_cache(MODEL_CFG.context_length, self.rotary_dim, self.dtype)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        use_kv_cache: bool,
        cur_index: Optional[jnp.ndarray],
        kv_seq_len: jnp.ndarray,  # (B,)
        q_seq_len: jnp.ndarray,  # (B,)
        is_causal: bool,
    ) -> jnp.ndarray:
        b, l, _ = x.shape
        hd = self.head_dim

        q_size = self.num_heads * hd
        kv_size = self.num_kv * hd

        qkv = self.qkv_proj(x)
        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q_chunk.reshape(b, l, self.num_heads, hd)  # (B, T, N, H)
        k = k_chunk.reshape(b, l, self.num_kv, hd)  # (B, T, K, H)
        v = v_chunk.reshape(b, l, self.num_kv, hd)

        # RoPE slice: positions [cur_index : cur_index + l] when caching, else [0:l]
        if use_kv_cache:
            if cur_index is None:
                raise ValueError("cur_index is required when use_kv_cache=True")
            sin = jax.lax.dynamic_slice(self._rope_sin, (0, cur_index, 0, 0), (1, l, 1, self.rotary_dim))
            cos = jax.lax.dynamic_slice(self._rope_cos, (0, cur_index, 0, 0), (1, l, 1, self.rotary_dim))
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self.rotary_dim)
        k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        if use_kv_cache:
            # Cache layout matches dot_product_attention directly: (B, S, K, H)
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, MODEL_CFG.context_length, self.num_kv, hd),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, MODEL_CFG.context_length, self.num_kv, hd),
                self.dtype,
            )

            # contiguous write (no scatter)
            cached_k.value = jax.lax.dynamic_update_slice(cached_k.value, k, (0, cur_index, 0, 0))
            cached_v.value = jax.lax.dynamic_update_slice(cached_v.value, v, (0, cur_index, 0, 0))

            if l > 1:
                # Prefill: attend over the prompt chunk itself (fast long-seq kernel), causal within prompt.
                y = jax.nn.dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    query_seq_lengths=q_seq_len,
                    key_value_seq_lengths=kv_seq_len,
                    implementation=self.attn_impl,
                )
            else:
                # Decode: attend over cached prefix; lengths tell kernel to ignore tail.
                y = jax.nn.dot_product_attention(
                    q,
                    cached_k.value,
                    cached_v.value,
                    is_causal=is_causal,
                    query_seq_lengths=q_seq_len,
                    key_value_seq_lengths=kv_seq_len,
                    implementation=self.attn_impl,
                )
        else:
            # No-cache mode.
            y = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                query_seq_lengths=q_seq_len,
                key_value_seq_lengths=kv_seq_len,
                implementation=self.attn_impl,
            )

        y = y.reshape(b, l, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class DeepSeekMLASelfAttention(nn.Module):
    """
    DeepSeek-style MLA-ish attention (benchmark-oriented):

    - Build q/k for attention scores as concat([latent, rope]) where rope slice gets RoPE.
    - Values are stored in latent dim (compressed).
    - Output per head: latent -> head_dim using a learned per-head up-projection.
    - KV cache stores:
        K_cat: (latent_dim + rope_dim)
        V_lat: (latent_dim)
      => smaller than baseline KV cache (2*head_dim).

    This is intended for speed/cache footprint comparison, not exact DeepSeek parity.
    """

    num_heads: int
    qkv_features: int
    num_kv: int
    latent_dim: int = MLA_LATENT_DIM_DEFAULT
    rope_dim: int = MLA_ROPE_DIM_DEFAULT
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    attn_impl: str = "cudnn"  # "cudnn" or "xla"

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        if self.num_heads % self.num_kv != 0:
            raise ValueError("num_heads must be divisible by num_kv")
        if self.rope_dim % 2 != 0:
            raise ValueError("rope_dim must be even")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.rope_dim < 0:
            raise ValueError("rope_dim must be >= 0")

        self.head_dim = self.qkv_features // self.num_heads
        self.kq_dim = self.latent_dim + self.rope_dim

        # q -> (N * kq_dim), kv -> (K * (kq_dim + kq_dim))
        self.q_proj = nn.Dense(
            self.num_heads * self.kq_dim,
            use_bias=False,
            name="q_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.kv_proj = nn.Dense(
            self.num_kv * (2 * self.kq_dim),
            use_bias=False,
            name="kv_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )

        # Per-query-head up-projection (N, kq_dim, head_dim)
        self.out_up = self.param(
            "out_up",
            nn.initializers.normal(stddev=0.02),
            (self.num_heads, self.kq_dim, self.head_dim),
            PARAM_DTYPE,
        )

        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # RoPE cache only for rope_dim slice
        if self.rope_dim > 0:
            self._rope_sin, self._rope_cos = _build_rope_cache(MODEL_CFG.context_length, self.rope_dim, self.dtype)
        else:
            self._rope_sin, self._rope_cos = None, None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        use_kv_cache: bool,
        cur_index: Optional[jnp.ndarray],
        kv_seq_len: jnp.ndarray,  # (B,)
        q_seq_len: jnp.ndarray,  # (B,)
        is_causal: bool,
    ) -> jnp.ndarray:
        b, l, _ = x.shape

        # Project Q: (B, T, N, kq_dim)
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.kq_dim)
        if self.rope_dim > 0:
            q_lat, q_rope = jnp.split(q, [self.latent_dim], axis=-1)
        else:
            q_lat, q_rope = q, None

        # Project KV: (B, T, K, 2*kq_dim) -> split into k_cat + v_cat
        kv = self.kv_proj(x).reshape(b, l, self.num_kv, 2 * self.kq_dim)
        k_cat, v_cat = jnp.split(kv, [self.kq_dim], axis=-1)

        if self.rope_dim > 0:
            k_lat, k_rope = jnp.split(k_cat, [self.latent_dim], axis=-1)

            # RoPE positions [cur_index : cur_index + l] when caching, else [0:l]
            if use_kv_cache:
                if cur_index is None:
                    raise ValueError("cur_index is required when use_kv_cache=True")
                sin = jax.lax.dynamic_slice(self._rope_sin, (0, cur_index, 0, 0), (1, l, 1, self.rope_dim))
                cos = jax.lax.dynamic_slice(self._rope_cos, (0, cur_index, 0, 0), (1, l, 1, self.rope_dim))
            else:
                sin = self._rope_sin[:, :l, :, :]
                cos = self._rope_cos[:, :l, :, :]

            q_rope = apply_partial_rope(q_rope, sin, cos, self.rope_dim)
            k_rope = apply_partial_rope(k_rope, sin, cos, self.rope_dim)

            q_cat = jnp.concatenate([q_lat, q_rope], axis=-1)
            k_cat = jnp.concatenate([k_lat, k_rope], axis=-1)
        else:
            q_cat = q_lat
            # when rope_dim==0, k_cat already is latent-only
            k_cat = k_cat

        if use_kv_cache:
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, MODEL_CFG.context_length, self.num_kv, self.kq_dim),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, MODEL_CFG.context_length, self.num_kv, self.kq_dim),
                self.dtype,
            )

            cached_k.value = jax.lax.dynamic_update_slice(cached_k.value, k_cat, (0, cur_index, 0, 0))
            cached_v.value = jax.lax.dynamic_update_slice(cached_v.value, v_cat, (0, cur_index, 0, 0))

            if l > 1:
                y_cat = jax.nn.dot_product_attention(
                    q_cat,
                    k_cat,
                    v_cat,
                    is_causal=is_causal,
                    query_seq_lengths=q_seq_len,
                    key_value_seq_lengths=kv_seq_len,
                    implementation=self.attn_impl,
                )
            else:
                y_cat = jax.nn.dot_product_attention(
                    q_cat,
                    cached_k.value,
                    cached_v.value,
                    is_causal=is_causal,
                    query_seq_lengths=q_seq_len,
                    key_value_seq_lengths=kv_seq_len,
                    implementation=self.attn_impl,
                )
        else:
            y_cat = jax.nn.dot_product_attention(
                q_cat,
                k_cat,
                v_cat,
                is_causal=is_causal,
                query_seq_lengths=q_seq_len,
                key_value_seq_lengths=kv_seq_len,
                implementation=self.attn_impl,
            )

        # Up-project per query head: (B,T,N,kq_dim) x (N,kq_dim,head_dim) -> (B,T,N,head_dim)
        w_up = self.out_up.astype(self.dtype)
        y = jnp.einsum("btnd,ndh->btnh", y_cat, w_up)

        y = y.reshape(b, l, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


# ---------------------------- Transformer block ---------------------------- #


class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    attn_kind: str = "gqa"  # "gqa" or "mla"
    mla_latent_dim: int = MLA_LATENT_DIM_DEFAULT
    mla_rope_dim: int = MLA_ROPE_DIM_DEFAULT
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    attn_impl: str = "cudnn"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        use_kv_cache: bool,
        cur_index: Optional[jnp.ndarray],
        kv_seq_len: jnp.ndarray,
        q_seq_len: jnp.ndarray,
        is_causal: bool,
    ) -> jnp.ndarray:
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=module.dtype, epsilon=1e-5)(h)

            if module.attn_kind == "mla":
                attn = DeepSeekMLASelfAttention(
                    num_heads=module.n_heads,
                    num_kv=module.n_kv_heads,
                    qkv_features=module.d_model,
                    latent_dim=module.mla_latent_dim,
                    rope_dim=module.mla_rope_dim,
                    dropout_rate=module.dropout_rate,
                    dtype=module.dtype,
                    attn_impl=module.attn_impl,
                    name="attn_mla",
                )
            else:
                attn = NativeJaxSelfAttention(
                    num_heads=module.n_heads,
                    num_kv=module.n_kv_heads,
                    qkv_features=module.d_model,
                    dropout_rate=module.dropout_rate,
                    dtype=module.dtype,
                    attn_impl=module.attn_impl,
                    name="attn_gqa",
                )

            h_attn = attn(
                h_norm,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                kv_seq_len=kv_seq_len,
                q_seq_len=q_seq_len,
                is_causal=is_causal,
            )
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=module.dtype, epsilon=1e-5)(h)

            # SwiGLU-ish FFN
            proj_dim = module.d_ff * 2
            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_norm)
            u, v = jnp.split(h_proj, 2, axis=-1)
            h_ffn = nn.silu(u) * v

            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        block_fn = nn.remat(_block) if MODEL_CFG.use_remat else _block
        return block_fn(self, x)


# ------------------------------- Model ------------------------------------- #


class GiantGPT(nn.Module):
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_layers: int
    attn_kind: str = "gqa"  # "gqa" or "mla"
    mla_latent_dim: int = MLA_LATENT_DIM_DEFAULT
    mla_rope_dim: int = MLA_ROPE_DIM_DEFAULT
    dropout_rate: float = 0.0
    attn_impl: str = "cudnn"

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        *,
        deterministic: bool,
        use_kv_cache: bool,
        cur_index: Optional[jnp.ndarray],
        kv_seq_len: jnp.ndarray,
        q_seq_len: jnp.ndarray,
        is_causal: bool,
    ) -> jnp.ndarray:
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="tok_embed",
        )
        x = embed(tokens)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for i in range(self.n_layers):
            x = TinyTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                attn_kind=self.attn_kind,
                mla_latent_dim=self.mla_latent_dim,
                mla_rope_dim=self.mla_rope_dim,
                dropout_rate=self.dropout_rate,
                dtype=COMPUTE_DTYPE,
                attn_impl=self.attn_impl,
                name=f"block_{i}",
            )(
                x,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                kv_seq_len=kv_seq_len,
                q_seq_len=q_seq_len,
                is_causal=is_causal,
            )

        # BF16 logits (benchmark-friendly). If you want fp32 logits, cast here.
        logits = jnp.matmul(x, embed.embedding.T)  # (B, L, V)
        return logits


# ------------------------- KV cache inference ------------------------------ #

Array = jnp.ndarray
PyTree = Dict[str, Any]


def init_inference_state(
    model: GiantGPT,
    key_params: jax.Array,
    key_dropout: jax.Array,
    batch_size: int,
    *,
    pad_token_id: int = 0,
) -> Tuple[PyTree, PyTree]:
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=jnp.array(0, jnp.int32),
        kv_seq_len=jnp.ones((batch_size,), jnp.int32),
        q_seq_len=jnp.ones((batch_size,), jnp.int32),
        is_causal=False,
    )
    params = variables["params"]
    nonparam = {k: v for k, v in variables.items() if k != "params"}
    if "cache" not in nonparam:
        raise ValueError("Model did not create a 'cache' collection during init.")
    return params, nonparam


def _apply_with_cache(
    model: GiantGPT,
    params: PyTree,
    nonparam: PyTree,
    tokens: Array,  # (B, L)
    cur_idx: Array,  # scalar int32
    kv_seq_len: Array,  # (B,) int32
    q_seq_len: Array,  # (B,) int32
    is_causal: bool,
) -> Tuple[Array, PyTree]:
    variables = {"params": params, **nonparam}
    logits, new_vars = model.apply(
        variables,
        tokens,
        deterministic=True,
        use_kv_cache=True,
        cur_index=cur_idx,
        kv_seq_len=kv_seq_len,
        q_seq_len=q_seq_len,
        is_causal=is_causal,
        mutable=["cache"],
    )
    nonparam_out = {**nonparam, "cache": new_vars["cache"]}
    return logits, nonparam_out


def make_prefill_and_decode_fns(model: GiantGPT):
    # IMPORTANT: no donate_argnums here -> avoids "buffer deleted/donated" issues during bring-up.
    @jax.jit
    def prefill(
        params: PyTree,
        nonparam: PyTree,
        prompt_tokens: Array,  # (B, Lp)
    ):
        b, lp = prompt_tokens.shape
        if lp == 0:
            t_last = jnp.array(0, jnp.int32)
            last_tok_2d = jnp.zeros((b, 1), dtype=jnp.int32)
            return nonparam, t_last, last_tok_2d

        cur0 = jnp.array(0, jnp.int32)
        kv_len = jnp.full((b,), lp, dtype=jnp.int32)
        q_len = jnp.full((b,), lp, dtype=jnp.int32)

        _logits, nonparam_out = _apply_with_cache(
            model, params, nonparam, prompt_tokens, cur0, kv_len, q_len, is_causal=True
        )

        t_last = jnp.array(lp - 1, jnp.int32)
        last_tok_2d = prompt_tokens[:, -1:]
        return nonparam_out, t_last, last_tok_2d

    @partial(jax.jit, static_argnames=("steps",))
    def decode(
        params: PyTree,
        nonparam: PyTree,
        last_tok_2d: Array,  # (B, 1)
        t_last: Array,  # scalar int32 (position of last_tok_2d)
        *,
        steps: int,
    ):
        b = last_tok_2d.shape[0]
        out = jnp.zeros((b, steps), dtype=jnp.int32)

        def body(carry, i):
            nonparam, t, tok_prev_2d, out = carry

            kv_len = jnp.full((b,), t + 1, dtype=jnp.int32)
            q_len = jnp.ones((b,), dtype=jnp.int32)

            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_prev_2d, t, kv_len, q_len, is_causal=False)
            step_logits = logits[:, -1, :]
            next_tok = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)

            out = jax.lax.dynamic_update_slice(out, next_tok[:, None], (0, i))
            return (nonparam, t + 1, next_tok[:, None], out), None

        (nonparam, _t, _tok2d, out), _ = jax.lax.scan(
            body,
            init=(nonparam, t_last, last_tok_2d, out),
            xs=jnp.arange(steps, dtype=jnp.int32),
        )
        return out, nonparam

    return prefill, decode


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


# ------------------------------- Benchmark --------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speed test a JAX transformer with KV cache (bf16, cudnn on GPU).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--force_xla", action="store_true", help="Force implementation='xla' even on GPU.")

    # MLA comparison is optional via flag (keeps old behavior by default)
    p.add_argument(
        "--mla",
        action="store_true",
        help="Run a second benchmark pass using DeepSeek-style MLA and print head-to-head results.",
    )
    p.add_argument("--mla_latent_dim", type=int, default=MLA_LATENT_DIM_DEFAULT)
    p.add_argument("--mla_rope_dim", type=int, default=MLA_ROPE_DIM_DEFAULT)

    return p.parse_args()


def assert_tree_on_backend(tree, name: str, platform: str):
    plats = []
    for x in jax.tree_util.tree_leaves(tree):
        if isinstance(x, jax.Array):
            plats.append(x.device.platform)
    if plats and any(p != platform for p in plats):
        raise RuntimeError(f"{name} not all on {platform}: {set(plats)}")


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.prompt_len < 0:
        raise ValueError("prompt_len must be >= 0")
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.prompt_len + args.steps > MODEL_CFG.context_length:
        raise ValueError(f"prompt_len + steps must be <= context_length ({MODEL_CFG.context_length})")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("runs must be > 0")

    print("JAX default backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    attn_impl_req = choose_attention_impl(force_xla=args.force_xla)
    print("Attention implementation request:", attn_impl_req)

    rng = jax.random.PRNGKey(args.seed)
    # keep prompt fixed; use separate param keys for baseline and MLA
    key_prompt, key_params_base, key_params_mla, key_dropout = jax.random.split(rng, 4)

    prompt = jax.random.randint(
        key_prompt,
        (args.batch_size, args.prompt_len),
        minval=0,
        maxval=MODEL_CFG.vocab_size,
        dtype=jnp.int32,
    )

    # Optional force placement to GPU if available (removes ambiguity about host work)
    if IS_GPU:
        gpu0 = jax.devices("gpu")[0]
        prompt = jax.device_put(prompt, gpu0)
        assert_tree_on_backend(prompt, "prompt", "gpu")

    def run_bench(
        *,
        label: str,
        attn_kind: str,
        key_params: jax.Array,
        mla_latent_dim: int,
        mla_rope_dim: int,
        attn_impl_requested: str,
    ):
        model = GiantGPT(
            vocab_size=MODEL_CFG.vocab_size,
            context_length=MODEL_CFG.context_length,
            d_model=MODEL_CFG.embedding_size,
            n_heads=MODEL_CFG.num_heads,
            n_kv_heads=MODEL_CFG.num_kv_heads,
            d_ff=MODEL_CFG.feed_forward_size,
            n_layers=MODEL_CFG.num_layers,
            dropout_rate=MODEL_CFG.dropout_rate,
            attn_impl=attn_impl_requested,
            attn_kind=attn_kind,
            mla_latent_dim=mla_latent_dim,
            mla_rope_dim=mla_rope_dim,
        )

        params, nonparam = init_inference_state(
            model,
            key_params,
            key_dropout,
            batch_size=args.batch_size,
            pad_token_id=0,
        )

        if IS_GPU:
            gpu0 = jax.devices("gpu")[0]
            params = jax.device_put(params, gpu0)
            nonparam = jax.device_put(nonparam, gpu0)
            assert_tree_on_backend(params, "params", "gpu")
            assert_tree_on_backend(nonparam, "nonparam", "gpu")

        def compile_all(m: GiantGPT):
            pf, df = make_prefill_and_decode_fns(m)
            compiled_pf = pf.lower(params, nonparam, prompt).compile()
            compiled_df = df.lower(
                params,
                nonparam,
                jnp.zeros((args.batch_size, 1), dtype=jnp.int32),
                jnp.array(0, jnp.int32),
                steps=args.steps,
            ).compile()
            return compiled_pf, compiled_df

        # Compile; if GPU+cudnn fails, fall back to xla on GPU.
        attn_impl_used = attn_impl_requested
        try:
            compiled_prefill, compiled_decode = compile_all(model)
        except Exception as e:
            msg = str(e).lower()
            if IS_GPU and attn_impl_requested == "cudnn" and "cudnn" in msg:
                print(f"\n[warn] {label}: cudnn attention compile failed; falling back to implementation='xla' on GPU.")
                attn_impl_used = "xla"
                model_xla = GiantGPT(
                    vocab_size=MODEL_CFG.vocab_size,
                    context_length=MODEL_CFG.context_length,
                    d_model=MODEL_CFG.embedding_size,
                    n_heads=MODEL_CFG.num_heads,
                    n_kv_heads=MODEL_CFG.num_kv_heads,
                    d_ff=MODEL_CFG.feed_forward_size,
                    n_layers=MODEL_CFG.num_layers,
                    dropout_rate=MODEL_CFG.dropout_rate,
                    attn_impl="xla",
                    attn_kind=attn_kind,
                    mla_latent_dim=mla_latent_dim,
                    mla_rope_dim=mla_rope_dim,
                )
                compiled_prefill, compiled_decode = compile_all(model_xla)
            else:
                raise

        # Warmup
        for _ in range(args.warmup):
            nonparam_filled, t_last, last_tok = compiled_prefill(params, nonparam, prompt)
            out, _nonparam_after = compiled_decode(params, nonparam_filled, last_tok, t_last)
            block_until_ready(out)

        # Timed runs
        prefill_times = []
        decode_times = []
        for _ in range(args.runs):
            prefill_start = time.perf_counter()
            nonparam_filled, t_last, last_tok = compiled_prefill(params, nonparam, prompt)
            block_until_ready(nonparam_filled)
            prefill_times.append(time.perf_counter() - prefill_start)

            decode_start = time.perf_counter()
            out, _ = compiled_decode(params, nonparam_filled, last_tok, t_last)
            out.block_until_ready()
            decode_times.append(time.perf_counter() - decode_start)

        prefill_time = sum(prefill_times) / len(prefill_times)
        decode_time = sum(decode_times) / len(decode_times)
        tokens_per_s = (args.steps * args.batch_size) / decode_time if decode_time > 0 else float("inf")

        # KV cache footprint per token (bf16 scalars)
        head_dim = MODEL_CFG.embedding_size // MODEL_CFG.num_heads
        if attn_kind == "mla":
            kq_dim = mla_latent_dim + mla_rope_dim
            kv_scalars_per_tok = MODEL_CFG.num_kv_heads * (2 * kq_dim)
        else:
            kv_scalars_per_tok = MODEL_CFG.num_kv_heads * (head_dim + head_dim)

        print(f"\n==================== {label} ====================")
        print("[bench]")
        print(f"attn_kind: {attn_kind}")
        if attn_kind == "mla":
            print(f"mla_latent_dim: {mla_latent_dim}")
            print(f"mla_rope_dim: {mla_rope_dim}")
        print(f"attn_impl_used: {attn_impl_used}")
        print(f"batch_size: {args.batch_size}")
        print(f"prompt_len: {args.prompt_len}")
        print(f"decode_steps: {args.steps}")
        print(f"prefill_time_s: {prefill_time:.6f}")
        print(f"decode_time_s: {decode_time:.6f}")
        print(f"tokens_per_second_decode: {tokens_per_s:.6f}")
        print(f"kv_cache_scalars_per_token_per_batch: {kv_scalars_per_tok} (dtype={MODEL_CFG.compute_dtype})")

        return {
            "label": label,
            "attn_kind": attn_kind,
            "attn_impl_used": attn_impl_used,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "tokens_per_second_decode": tokens_per_s,
            "kv_cache_scalars_per_tok": kv_scalars_per_tok,
        }

    # Print global config once (matches old script vibe)
    print("\n[config]")
    print(f"vocab_size: {MODEL_CFG.vocab_size}")
    print(f"context_length: {MODEL_CFG.context_length}")
    print(f"d_model: {MODEL_CFG.embedding_size}")
    print(f"num_heads: {MODEL_CFG.num_heads}")
    print(f"num_kv_heads: {MODEL_CFG.num_kv_heads}")
    print(f"num_layers: {MODEL_CFG.num_layers}")
    print(f"feed_forward_size: {MODEL_CFG.feed_forward_size}")
    print(f"rope_dim: {MODEL_CFG.rope_dim}")
    print(f"param_dtype: {MODEL_CFG.param_dtype}")
    print(f"compute_dtype: {MODEL_CFG.compute_dtype}")
    print(f"attn_impl_requested: {attn_impl_req}")
    if args.mla:
        print(f"mla_latent_dim: {args.mla_latent_dim}")
        print(f"mla_rope_dim: {args.mla_rope_dim}")

    # Always run baseline (keeps old behavior)
    base = run_bench(
        label="Baseline GQA (NativeJaxSelfAttention)",
        attn_kind="gqa",
        key_params=key_params_base,
        mla_latent_dim=args.mla_latent_dim,
        mla_rope_dim=args.mla_rope_dim,
        attn_impl_requested=attn_impl_req,
    )

    # Optional MLA pass + head-to-head summary
    if args.mla:
        mla = run_bench(
            label="DeepSeek-style MLA (latent KV + small RoPE slice)",
            attn_kind="mla",
            key_params=key_params_mla,
            mla_latent_dim=args.mla_latent_dim,
            mla_rope_dim=args.mla_rope_dim,
            attn_impl_requested=attn_impl_req,
        )

        speedup_decode = mla["tokens_per_second_decode"] / base["tokens_per_second_decode"]
        speedup_prefill = base["prefill_time_s"] / mla["prefill_time_s"]
        cache_ratio = base["kv_cache_scalars_per_tok"] / mla["kv_cache_scalars_per_tok"]

        print("\n==================== HEAD-TO-HEAD ====================")
        print(f"decode tokens/s speedup (MLA / Baseline): {speedup_decode:.4f}x")
        print(f"prefill time ratio (Baseline / MLA): {speedup_prefill:.4f}x")
        print(f"KV cache compression (Baseline / MLA): {cache_ratio:.4f}x")


if __name__ == "__main__":
    main()
