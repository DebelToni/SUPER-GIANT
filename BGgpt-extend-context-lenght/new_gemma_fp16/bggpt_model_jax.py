# bggpt_model_jax.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from bggpt_config import BgGPTConfig

PyTree = Dict[str, Any]


@dataclass(frozen=True)
class KVCache:
    """
    KV cache layout:

    k: (num_layers, batch, num_kv_heads, max_seq_len, head_dim)
    v: (num_layers, batch, num_kv_heads, max_seq_len, head_dim)
    """
    k: jnp.ndarray
    v: jnp.ndarray

    @property
    def num_layers(self) -> int:
        return self.k.shape[0]

    @property
    def max_seq_len(self) -> int:
        return self.k.shape[3]


def init_kv_cache(
    config: BgGPTConfig,
    batch_size: int,
    max_seq_len: int | None = None,
    dtype=jnp.float16,
    device: jax.Device | None = None,
) -> KVCache:
    """
    Allocate an empty KV cache on the specified device.

    Args:
        config: BgGPTConfig.
        batch_size: batch size for generation.
        max_seq_len: maximum total context length (prompt + new tokens).
        dtype: dtype for cache (float16 recommended).
        device: optional JAX device.

    Returns:
        KVCache with all zeros.
    """
    if max_seq_len is None:
        max_seq_len = config.max_position_embeddings

    shape = (
        config.num_hidden_layers,
        batch_size,
        config.num_key_value_heads,
        max_seq_len,
        config.head_dim,
    )
    k = jnp.zeros(shape, dtype=dtype)
    v = jnp.zeros(shape, dtype=dtype)

    if device is not None:
        k = jax.device_put(k, device=device)
        v = jax.device_put(v, device=device)

    return KVCache(k=k, v=v)


# --- Core math primitives ---


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """RMSNorm (no bias)."""
    # x: (..., hidden)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(variance + eps)
    return x_norm * weight


def gelu_pytorch_tanh(x: jnp.ndarray) -> jnp.ndarray:
    """
    The 'gelu_pytorch_tanh' used by Gemma/Gemma2.

    Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    return jax.nn.gelu(x, approximate="tanh")


def geglu(x_gate: jnp.ndarray, x_up: jnp.ndarray) -> jnp.ndarray:
    """
    Gemma-style GeGLU MLP: GELU(gate) * up.
    """
    return gelu_pytorch_tanh(x_gate) * x_up


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    rope_cache: Tuple[jnp.ndarray, jnp.ndarray],
    positions: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply RoPE to q and k.

    Args:
        q, k: (batch, heads, seq_len, head_dim)
        rope_cache: (cos, sin) each of shape (max_seq_len, head_dim)
        positions: (seq_len,) integer positions (0-based).

    Returns:
        (q_rot, k_rot) with same shape as inputs.
    """
    cos, sin = rope_cache  # (max_seq_len, head_dim)
    # Select positions used in this call
    cos_pos = cos[positions]  # (seq_len, head_dim)
    sin_pos = sin[positions]

    # Broadcast to (1,1,seq_len,head_dim)
    cos_pos = cos_pos[None, None, :, :]
    sin_pos = sin_pos[None, None, :, :]

    def _rotate(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Equation from standard RoPE
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos
        return jnp.stack(
            (x_rot_even, x_rot_odd), axis=-1
        ).reshape(x.shape)

    return _rotate(q), _rotate(k)


def build_rope_cache(
    config: BgGPTConfig,
    max_seq_len: int | None = None,
    dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute RoPE cos/sin tables.

    Returns:
        (cos, sin) each shape (max_seq_len, head_dim)
    """
    if max_seq_len is None:
        max_seq_len = config.max_position_embeddings

    head_dim = config.head_dim
    theta = config.rope_theta

    # Frequencies: shape (head_dim/2,)
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim)
    )  # (d/2,)

    # Positions: shape (max_seq_len, 1)
    positions = jnp.arange(max_seq_len, dtype=dtype)[:, None]  # (L,1)
    # Angles: (L, d/2)
    angles = positions * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    # Interleave to get full head_dim dimension
    # We'll build shape (L, head_dim) such that even/odd indices are cos/sin.
    cos_full = jnp.repeat(cos, 2, axis=-1)  # (L, d)
    sin_full = jnp.repeat(sin, 2, axis=-1)  # (L, d)

    return cos_full.astype(dtype), sin_full.astype(dtype)


# --- Attention and layers ---


def _linear(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    Linear layer without bias.

    Args:
        x: (..., in_features)
        w: (out_features, in_features)

    Returns:
        (..., out_features)
    """
    return jnp.einsum("...i,oi->...o", x, w)


def attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Standard scaled dot-product attention.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        mask: (1, 1, seq_len, seq_len) bool, True = keep, False = -inf

    Returns:
        out: (batch, heads, seq_len, head_dim)
    """
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d).astype(q.dtype)

    # (b,h,q,k)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if mask is not None:
        # mask False => large negative
        scores = jnp.where(mask, scores, jnp.array(-1e9, dtype=scores.dtype))

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    return out


def build_causal_mask(seq_len: int, dtype=bool) -> jnp.ndarray:
    """
    Causal mask of shape (1, 1, seq_len, seq_len), True on allowed positions.
    """
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    mask = i >= j  # lower-triangular
    return mask[None, None, :, :].astype(dtype)


def transformer_block_prefill(
    x: jnp.ndarray,
    layer: PyTree,
    config: BgGPTConfig,
    rope_cache: Tuple[jnp.ndarray, jnp.ndarray],
    layer_idx: int,
    causal_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass of a single transformer block in prefill mode.

    Args:
        x: (batch, seq_len, hidden)
        layer: layer params dict for this block.
        config: BgGPTConfig.
        rope_cache: (cos, sin)
        layer_idx: index (unused currently, kept for future).
        causal_mask: (1,1,seq_len,seq_len) bool

    Returns:
        x_out: (batch, seq_len, hidden)
    """
    B, T, H = x.shape
    cfg = config

    # --- Attention ---
    x_norm = rms_norm(x, layer["attn_norm"], cfg.rms_norm_eps)

    # Projections
    q = _linear(x_norm, layer["wq"])  # (B,T,q_dim)
    k = _linear(x_norm, layer["wk"])  # (B,T,kv_dim)
    v = _linear(x_norm, layer["wv"])

    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    # q_dim = n_heads * head_dim, kv_dim = n_kv * head_dim
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_kv, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_kv, head_dim).transpose(0, 2, 1, 3)

    # Apply RoPE
    positions = jnp.arange(T, dtype=jnp.int32)  # (T,)
    q, k = apply_rotary_pos_emb(q, k, rope_cache, positions)

    # Group-query attention: replicate K/V across head groups
    group_size = n_heads // n_kv
    if group_size > 1:
        k = jnp.repeat(k, group_size, axis=1)  # (B,n_heads,T,D)
        v = jnp.repeat(v, group_size, axis=1)

    attn_out = attention(q, k, v, mask=causal_mask)  # (B,n_heads,T,D)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, H)

    x = x + _linear(attn_out, layer["wo"])

    # --- MLP ---
    y = rms_norm(x, layer["ffn_norm"], cfg.rms_norm_eps)
    gate = _linear(y, layer["w_gate"])  # (B,T,ff_dim)
    up = _linear(y, layer["w_up"])
    z = geglu(gate, up)
    z = _linear(z, layer["w_down"])
    x = x + z

    return x


def transformer_block_decode(
    x: jnp.ndarray,
    layer: PyTree,
    config: BgGPTConfig,
    rope_cache: Tuple[jnp.ndarray, jnp.ndarray],
    layer_idx: int,
    kv_cache: KVCache,
    pos: int,
) -> Tuple[jnp.ndarray, KVCache]:
    """
    Single-token decode for one transformer block.

    Args:
        x: (batch, 1, hidden)
        layer: layer params
        config: BgGPTConfig
        rope_cache: (cos, sin)
        layer_idx: int
        kv_cache: KVCache (updated in-place)
        pos: integer position index (0-based)

    Returns:
        (x_out, new_kv_cache)
    """
    B, T, H = x.shape
    assert T == 1, "decode path expects T=1"

    cfg = config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    # --- Attention ---
    x_norm = rms_norm(x, layer["attn_norm"], cfg.rms_norm_eps)

    q = _linear(x_norm, layer["wq"])  # (B,1,q_dim)
    k = _linear(x_norm, layer["wk"])
    v = _linear(x_norm, layer["wv"])

    q = q.reshape(B, 1, n_heads, head_dim).transpose(0, 2, 1, 3)  # (B,h,1,D)
    k = k.reshape(B, 1, n_kv, head_dim).transpose(0, 2, 1, 3)  # (B,kv,1,D)
    v = v.reshape(B, 1, n_kv, head_dim).transpose(0, 2, 1, 3)

    # RoPE for position `pos`
    positions = jnp.array([pos], dtype=jnp.int32)  # (1,)
    q, k = apply_rotary_pos_emb(q, k, rope_cache, positions)

    # Update KV cache
    # cache_k/v: (L,B,kv,T_max,D)
    cache_k = kv_cache.k
    cache_v = kv_cache.v

    cache_k = cache_k.at[layer_idx, :, :, pos, :].set(
        k.squeeze(axis=2)
    )  # k: (B,kv,1,D) -> (B,kv,D)
    cache_v = cache_v.at[layer_idx, :, :, pos, :].set(v.squeeze(axis=2))

    # Keys & values up to current pos (inclusive)
    k_full = cache_k[layer_idx, :, :, : pos + 1, :]  # (B,kv,pos+1,D)
    v_full = cache_v[layer_idx, :, :, : pos + 1, :]

    # Group query: replicate KV across heads
    group_size = n_heads // n_kv
    if group_size > 1:
        k_full = jnp.repeat(k_full, group_size, axis=1)  # (B,h,pos+1,D)
        v_full = jnp.repeat(v_full, group_size, axis=1)

    # Compute attention
    q = q  # (B,h,1,D)
    attn_out = attention(q, k_full, v_full, mask=None)  # (B,h,1,D)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, H)

    x = x + _linear(attn_out, layer["wo"])

    # --- MLP ---
    y = rms_norm(x, layer["ffn_norm"], cfg.rms_norm_eps)
    gate = _linear(y, layer["w_gate"])
    up = _linear(y, layer["w_up"])
    z = geglu(gate, up)
    z = _linear(z, layer["w_down"])
    x = x + z

    new_cache = KVCache(k=cache_k, v=cache_v)
    return x, new_cache


# --- Full forward helpers ---


def forward_prefill(
    params: PyTree,
    config: BgGPTConfig,
    input_ids: jnp.ndarray,
    rope_cache: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """
    Forward pass over a full prompt (no KV cache).

    Args:
        params: parameter PyTree from bggpt_gguf_loader.
        config: BgGPTConfig.
        input_ids: (batch, seq_len) int32.
        rope_cache: (cos, sin) from build_rope_cache.

    Returns:
        logits: (batch, seq_len, vocab_size)
    """
    tok_emb = params["tok_embeddings"]
    layers = params["layers"]
    final_norm = params["final_norm"]
    lm_head = params["lm_head"]

    B, T = input_ids.shape
    H = config.hidden_size

    # Embed
    x = tok_emb[input_ids, :]  # (B,T,H)

    # Causal mask for full sequence
    causal_mask = build_causal_mask(T)

    # Blocks
    for layer_idx, layer in enumerate(layers):
        x = transformer_block_prefill(
            x, layer, config, rope_cache, layer_idx, causal_mask
        )

    # Final norm + LM head
    x = rms_norm(x, final_norm, config.rms_norm_eps)
    # LM head: assume lm_head: (vocab, hidden) or (hidden, vocab)
    if lm_head.shape[0] == config.vocab_size:
        # (B,T,H) x (V,H)^T -> (B,T,V)
        logits = jnp.einsum("bth,vh->btv", x, lm_head)
    else:
        # (B,T,H) x (H,V) -> (B,T,V)
        logits = jnp.einsum("bth,hv->btv", x, lm_head)

    return logits


def forward_decode_one(
    params: PyTree,
    config: BgGPTConfig,
    input_ids: jnp.ndarray,
    rope_cache: Tuple[jnp.ndarray, jnp.ndarray],
    kv_cache: KVCache,
    pos: int,
) -> Tuple[jnp.ndarray, KVCache]:
    """
    Single-token decode forward.

    Args:
        params: model params.
        config: BgGPTConfig.
        input_ids: (batch, 1) int32.
        rope_cache: (cos, sin).
        kv_cache: KVCache with past tokens (0..pos-1).
        pos: int position index for this new token.

    Returns:
        (logits: (batch, vocab_size), new_kv_cache)
    """
    tok_emb = params["tok_embeddings"]
    layers = params["layers"]
    final_norm = params["final_norm"]
    lm_head = params["lm_head"]

    x = tok_emb[input_ids, :]  # (B,1,H)

    cache = kv_cache
    for layer_idx, layer in enumerate(layers):
        x, cache = transformer_block_decode(
            x, layer, config, rope_cache, layer_idx, cache, pos
        )

    x = rms_norm(x, final_norm, config.rms_norm_eps)  # (B,1,H)
    x = x[:, 0, :]  # (B,H)

    if lm_head.shape[0] == config.vocab_size:
        logits = jnp.einsum("bh,vh->bv", x, lm_head)
    else:
        logits = jnp.einsum("bh,hv->bv", x, lm_head)

    return logits, cache

