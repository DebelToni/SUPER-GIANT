#!/usr/bin/env python3
"""
bench_generation_kvcache.py

End-to-end-ish decode benchmark with KV cache, comparing:
  - AR decode: 1 token per step
  - TiDAR-ish decode: K tokens per step (always accept), with step_len = K + K^2 tokens computed per step

Key properties:
  - Uses lax.scan, fixed shapes, JIT friendly
  - Keeps sampling + "rejection" on device (here we default to always-accept for speed measurement)
  - Avoids unembedding logits for all K^2 candidates:
      we only unembed needed slices:
        - AR: last position (1)
        - TiDAR: verify slice (K) and selected candidate slice (K)

Attention backends:
  - "jax": jax.nn.dot_product_attention
  - "jfa_triton": jax-flash-attn2, platform TRITON
  - "jfa_pallas": jax-flash-attn2, platform PALLAS
  - "jfa_jax": jax-flash-attn2, platform JAX

Notes:
  - jax-flash-attn2 supports attention masks/bias in some modes; for structured TiDAR masks we pass bias.
    If the backend rejects bias or mismatched Lq/Lk, we fall back to JAX attention. :contentReference[oaicite:2]{index=2}
  - Some Pallas GPU flash attention paths can be picky about shapes / block sizes. :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# -----------------------
# Config
# -----------------------

@dataclass(frozen=True)
class Cfg:
    # cache / generation
    context_len: int = 1024          # T
    prefix_init: int = 512           # P0 initial prefix length filled into cache
    steps: int = 128                 # number of decode iterations
    vocab: int = 32000
    k: int = 16                      # TiDAR K
    batch: int = 1

    # model size-ish
    d_model: int = 1024
    num_heads: int = 16
    head_dim: int = 64               # must match d_model == num_heads * head_dim
    num_layers: int = 8

    # sampling
    temperature: float = 1.0
    top_k: int = 0

    # backend
    backend: str = "jax"             # jax | jfa_triton | jfa_pallas | jfa_jax
    impl: str = "cudnn"              # for JAX dot_product_attention: cudnn|xla
    dtype: str = "bf16"              # bf16|fp16|fp32
    bias_value: float = -1.0e10

    # benchmark
    warmup: int = 1
    iters: int = 10

    # TiDAR mode
    tidar_always_accept: bool = True  # if True commit r=K each step (no second pass)
    pad_q_to_128: bool = True         # helps trigger cudnn paths sometimes


def _dtype(name: str) -> jnp.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if name in ("fp16", "float16"):
        return jnp.float16
    if name in ("fp32", "float32"):
        return jnp.float32
    raise ValueError(f"unknown dtype: {name}")


def _now() -> float:
    return time.perf_counter()


def _sync(x):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def _try_impl(impl: str) -> str:
    if impl == "cudnn" and jax.devices()[0].platform != "gpu":
        return "xla"
    return impl


def _ensure_jax_core_primitive() -> None:
    if hasattr(jax.core, "Primitive"):
        return
    try:
        from jax._src.core import Primitive
    except Exception:
        return
    setattr(jax.core, "Primitive", Primitive)


def _load_jfa_jax_only():
    import importlib
    import importlib.util
    import sys
    import types

    spec = importlib.util.find_spec("jax_flash_attn2")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("jax_flash_attn2 not installed")
    pkg = sys.modules.get("jax_flash_attn2")
    if pkg is None or not getattr(pkg, "__path__", None):
        pkg = types.ModuleType("jax_flash_attn2")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["jax_flash_attn2"] = pkg
    return importlib.import_module("jax_flash_attn2.flash_attention_jax")


# -----------------------
# Mask/Bias builders for TiDAR step (fixed shapes)
# -----------------------

def tidar_step_lens(k: int, pad_q_to_128: bool) -> Tuple[int, int]:
    """
    step tokens are: verify(K) + predraft(K*K)
    total step_len = K + K^2
    We optionally pad step_len up to >=128 and even, to help hit cudnn paths.
    """
    K = k
    q = K + K * K
    if pad_q_to_128:
        q = max(q, 128)
    if q % 2 == 1:
        q += 1
    return q, (K + K * K)


def build_tidar_struct_bias(
    *,
    context_len: int,
    k: int,
    step_len: int,          # actual step_len used in attention (may be padded)
    logical_step_len: int,  # K + K^2 (actual used tokens)
    bias_value: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Build structural bias for queries = step tokens only, keys = [prefix_cache(T) | step_tokens(step_len)].

    We define rules on the first logical_step_len queries/keys and mask out padded tail if step_len > logical_step_len.

    Layout inside step tokens:
      - verify queries/keys: [0..K-1]
      - candidate queries/keys: [K..K+K*K-1], flattened into K blocks of size K

    Rules (same as your earlier build_tidar_decode_bias_cached concept):
      - verify queries attend prefix keys + causal verify keys
      - verify queries do not attend candidates
      - candidate queries attend prefix keys
      - candidate block i (r=i+1) attends verify keys < r
      - candidate queries attend bidir within their own candidate block only
      - no attention between candidate blocks

    We treat prefix keys as always potentially visible; prefix_len validity is applied separately via key_valid bias.
    """
    T = context_len
    K = k
    Q = step_len
    LQ = logical_step_len
    KV = T + Q

    q = jnp.arange(Q)[:, None]      # (Q,1)
    kk = jnp.arange(KV)[None, :]    # (1,KV)

    # Identify padding in queries/keys
    q_is_real = q < LQ
    k_is_prefix = kk < T
    k_is_step = kk >= T
    k_step_idx = kk - T
    k_is_real_step = (k_step_idx < LQ) & k_is_step

    # Regions within step (for "real" part)
    q_is_verify = q_is_real & (q < K)
    q_is_cand = q_is_real & (q >= K) & (q < (K + K*K))

    k_is_verify = k_is_real_step & (k_step_idx < K)
    k_is_cand = k_is_real_step & (k_step_idx >= K) & (k_step_idx < (K + K*K))

    q_verify_pos = q
    k_verify_pos = k_step_idx

    q_cand_off = q - K
    k_cand_off = k_step_idx - K
    q_cand_block = q_cand_off // K
    k_cand_block = k_cand_off // K
    cand_r = q_cand_block + 1  # 1..K

    allow_verify = (
        (q_is_verify & k_is_prefix)
        | (q_is_verify & k_is_verify & (k_verify_pos <= q_verify_pos))
    )
    allow_cand = (
        (q_is_cand & k_is_prefix)
        | (q_is_cand & k_is_verify & (k_verify_pos < cand_r))
        | (q_is_cand & k_is_cand & (q_cand_block == k_cand_block))
    )

    allow = allow_verify | allow_cand

    # Disallow any attention involving padded queries or padded step keys
    allow = allow & q_is_real & (k_is_prefix | k_is_real_step)

    bias = jnp.where(allow, jnp.array(0.0, dtype=dtype), jnp.array(bias_value, dtype=dtype))
    return bias[None, None, :, :]  # (1,1,Q,KV)


def build_key_valid_bias(
    *,
    context_len: int,
    prefix_len: jnp.ndarray,   # scalar int32
    step_len: int,
    bias_value: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Key-valid bias for prefix cache:
      keys [0..context_len-1] are valid iff k < prefix_len, otherwise -inf.
      step keys are always valid.
    """
    T = context_len
    Q = step_len
    k = jnp.arange(T)
    valid = k < prefix_len
    prefix_bias = jnp.where(valid, jnp.array(0.0, dtype=dtype), jnp.array(bias_value, dtype=dtype))  # (T,)
    prefix_bias = prefix_bias[None, None, None, :]  # (1,1,1,T)
    step_bias = jnp.zeros((1, 1, 1, Q), dtype=dtype)
    return jnp.concatenate([prefix_bias, step_bias], axis=-1)  # (1,1,1,T+Q)


# -----------------------
# Sampling (JAX on-device)
# -----------------------

def topk_mask(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    if top_k <= 0:
        return logits
    vals, idx = jax.lax.top_k(logits, top_k)
    masked = jnp.full_like(logits, -jnp.inf)
    masked = jnp.take_along_axis(masked, jnp.zeros_like(idx), axis=-1)  # no-op to keep shape
    masked = masked.at[..., idx].set(vals)
    return masked


def sample_categorical(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_k: int,
) -> jnp.ndarray:
    logits = logits / jnp.maximum(temperature, 1e-6)
    logits = topk_mask(logits, top_k)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


# -----------------------
# Attention backend wrapper
# -----------------------

class AttnBackend:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.impl = _try_impl(cfg.impl)
        self._jfa = None
        self._jfa_kind = None
        self._jfa_ready = False
        self._init_jfa_if_needed()

    def _init_jfa_if_needed(self):
        if not self.cfg.backend.startswith("jfa_"):
            return
        _ensure_jax_core_primitive()
        try:
            import jax_flash_attn2 as jfa  # type: ignore
            platform = {
                "jfa_triton": jfa.Platform.TRITON,
                "jfa_pallas": jfa.Platform.PALLAS,
                "jfa_jax": jfa.Platform.JAX,
            }[self.cfg.backend]
            self._jfa = jfa.FlashAttention(
                jfa.AttentionConfig(platform=platform, backend=jfa.Backend.GPU)
            )
            self._jfa_kind = "flash_attention"
            self._jfa_ready = True
            return
        except Exception as e:
            try:
                jfa_jax = _load_jfa_jax_only()

                def _jfa_call(query, key, value, bias, causal):
                    return jfa_jax.jax_flash_attention(
                        query_state=query,
                        key_state=key,
                        value_state=value,
                        mask=None,
                        bias=bias,
                    )

                self._jfa = _jfa_call
                self._jfa_kind = "jax_fn"
                self._jfa_ready = True
                print(
                    f"[warn] jax-flash-attn2 backend unavailable ({e}). Using JAX backend."
                )
                return
            except Exception:
                print(
                    f"[warn] jax-flash-attn2 unavailable or failed to init ({e}). Falling back to JAX attention."
                )
                self._jfa_ready = False
                self._jfa = None
                self._jfa_kind = None

    def attn(
        self,
        q: jnp.ndarray,  # (B, Lq, H, D)
        k: jnp.ndarray,  # (B, Lk, H, D)
        v: jnp.ndarray,  # (B, Lk, H, D)
        bias: Optional[jnp.ndarray],  # (1,1,Lq,Lk) or (B,1,Lq,Lk)
        causal: bool,
    ) -> jnp.ndarray:
        """
        Returns y: (B, Lq, H, D)
        """
        if self.cfg.backend == "jax" or not self._jfa_ready:
            return jax.nn.dot_product_attention(
                q, k, v,
                bias=bias,
                is_causal=causal,
                implementation=self.impl,
            )

        if self._jfa_kind == "jax_fn":
            try:
                return self._jfa(q, k, v, bias, causal)
            except Exception:
                return jax.nn.dot_product_attention(
                    q, k, v,
                    bias=bias,
                    is_causal=causal,
                    implementation=self.impl,
                )

        try:
            out = self._jfa(query=q, key=k, value=v, bias=bias, causal=causal)
            return out
        except Exception:
            try:
                if bias is not None:
                    attn_mask = (bias == 0.0).astype(jnp.int32)
                else:
                    attn_mask = None
                out = self._jfa(query=q, key=k, value=v, attention_mask=attn_mask, causal=causal)
                return out
            except Exception:
                return jax.nn.dot_product_attention(
                    q, k, v,
                    bias=bias,
                    is_causal=causal,
                    implementation=self.impl,
                )


# -----------------------
# Tiny "decoder block" (just qkv + attn + out proj, optional ffn)
# -----------------------

def init_params(cfg: Cfg, key: jax.Array):
    """
    Create random weights for:
      - embedding: (vocab, d_model)
      - per-layer qkv: (d_model, 3*d_model)
      - per-layer o: (d_model, d_model)
      - optional ffn: w1 (d_model, 4*d_model), w2 (4*d_model, d_model)
    """
    d = cfg.d_model
    assert d == cfg.num_heads * cfg.head_dim, "d_model must equal num_heads*head_dim"
    dtype = _dtype(cfg.dtype)

    keys = jax.random.split(key, 1 + cfg.num_layers * 4)
    k0 = keys[0]
    ks = keys[1:]

    emb = jax.random.normal(k0, (cfg.vocab, d), dtype=dtype) * 0.02

    qkv_w = []
    o_w = []
    ffn_w1 = []
    ffn_w2 = []
    for i in range(cfg.num_layers):
        qkv_w.append(jax.random.normal(ks[4*i+0], (d, 3*d), dtype=dtype) * 0.02)
        o_w.append(jax.random.normal(ks[4*i+1], (d, d), dtype=dtype) * 0.02)
        ffn_w1.append(jax.random.normal(ks[4*i+2], (d, 4*d), dtype=dtype) * 0.02)
        ffn_w2.append(jax.random.normal(ks[4*i+3], (4*d, d), dtype=dtype) * 0.02)

    return {
        "emb": emb,
        "qkv_w": jnp.stack(qkv_w, axis=0),
        "o_w": jnp.stack(o_w, axis=0),
        "ffn_w1": jnp.stack(ffn_w1, axis=0),
        "ffn_w2": jnp.stack(ffn_w2, axis=0),
    }


def layer_forward(
    *,
    h: jnp.ndarray,                 # (B, Lq, D)
    k_cache: jnp.ndarray,           # (B, T, H, Dhead)
    v_cache: jnp.ndarray,           # (B, T, H, Dhead)
    prefix_len: jnp.ndarray,        # scalar int32
    layer_idx: int,
    params: dict,
    cfg: Cfg,
    attn_backend: AttnBackend,
    bias_struct: Optional[jnp.ndarray],
    causal: bool,
    step_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      - new h: (B, Lq, D)
      - k_step: (B, Lq, H, Dhead)
      - v_step: (B, Lq, H, Dhead)
    """
    B, Lq, D = h.shape
    H, Dh = cfg.num_heads, cfg.head_dim
    dtype = _dtype(cfg.dtype)

    w_qkv = params["qkv_w"][layer_idx]  # (D, 3D)
    w_o = params["o_w"][layer_idx]      # (D, D)
    w1 = params["ffn_w1"][layer_idx]
    w2 = params["ffn_w2"][layer_idx]

    qkv = jnp.einsum("bld,df->blf", h, w_qkv).astype(dtype)  # (B,L,3D)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    q = q.reshape(B, Lq, H, Dh)
    k = k.reshape(B, Lq, H, Dh)
    v = v.reshape(B, Lq, H, Dh)

    # Keys: [prefix_cache(T) | step_keys(Lq)]
    k_full = jnp.concatenate([k_cache, k], axis=1)  # (B, T+Lq, H, Dh)
    v_full = jnp.concatenate([v_cache, v], axis=1)

    # Bias: structural + key-valid (disable unused prefix slots)
    bias = None
    if bias_struct is not None:
        key_valid = build_key_valid_bias(
            context_len=cfg.context_len,
            prefix_len=prefix_len,
            step_len=step_len,
            bias_value=cfg.bias_value,
            dtype=dtype,
        )
        bias = bias_struct + key_valid

    y = attn_backend.attn(q=q, k=k_full, v=v_full, bias=bias, causal=causal)  # (B,Lq,H,Dh)
    y = y.reshape(B, Lq, D)
    y = jnp.einsum("bld,df->blf", y, w_o).astype(dtype)

    # tiny ffn (optional cost to better resemble a layer)
    # you can comment this out if you want pure attention speed
    ff = jnp.einsum("bld,df->blf", y, w1).astype(dtype)
    ff = jax.nn.silu(ff) * ff  # cheap-ish nonlinearity trick
    ff = jnp.einsum("bld,df->blf", ff, w2).astype(dtype)
    h_out = y + 0.1 * ff

    return h_out, k, v


# -----------------------
# Decode loops
# -----------------------

def ar_decode_benchmark(cfg: Cfg, params: dict, attn_backend: AttnBackend, seed: int):
    """
    AR generation:
      - cache holds prefix tokens K/V (randomly prefilled from prompt)
      - each step generates 1 token, appends to cache
    """
    dtype = _dtype(cfg.dtype)
    key = jax.random.PRNGKey(seed)

    B, T, D = cfg.batch, cfg.context_len, cfg.d_model
    H, Dh = cfg.num_heads, cfg.head_dim
    V = cfg.vocab

    emb = params["emb"]

    # Cache per layer: (L, B, T, H, Dh)
    k_cache = jnp.zeros((cfg.num_layers, B, T, H, Dh), dtype=dtype)
    v_cache = jnp.zeros((cfg.num_layers, B, T, H, Dh), dtype=dtype)

    # Prefill: random tokens length prefix_init -> write into cache with a single pass (not fully realistic, but good enough)
    prefix_len0 = jnp.array(cfg.prefix_init, dtype=jnp.int32)
    key, k_tokens = jax.random.split(key)
    prompt_tokens = jax.random.randint(k_tokens, (B, cfg.prefix_init), 0, V, dtype=jnp.int32)
    h_prompt = emb[prompt_tokens]  # (B,P,D)

    # We do a simple loop to fill caches with prompt tokens for each layer (no attention, just k/v projection)
    def fill_layer(layer_idx, carry):
        k_cache, v_cache = carry
        h = h_prompt
        # compute k/v for prompt tokens
        w_qkv = params["qkv_w"][layer_idx]
        qkv = jnp.einsum("bld,df->blf", h, w_qkv).astype(dtype)
        _, k, v = jnp.split(qkv, 3, axis=-1)
        k = k.reshape(B, cfg.prefix_init, H, Dh)
        v = v.reshape(B, cfg.prefix_init, H, Dh)

        k_cache = k_cache.at[layer_idx, :, :cfg.prefix_init, :, :].set(k)
        v_cache = v_cache.at[layer_idx, :, :cfg.prefix_init, :, :].set(v)
        return k_cache, v_cache

    k_cache, v_cache = jax.lax.fori_loop(0, cfg.num_layers, fill_layer, (k_cache, v_cache))

    # Start token (last token of prompt)
    last_tok = prompt_tokens[:, -1]  # (B,)

    def step(carry, _):
        key, prefix_len, last_tok, k_cache, v_cache, tokens_generated = carry
        key, k_step = jax.random.split(key)

        # embed current token as a length-1 sequence
        h = emb[last_tok][:, None, :]  # (B,1,D)

        # forward through layers, updating h each time
        def layer_loop(layer_idx, state):
            h, k_cache, v_cache = state
            # read caches for this layer
            kL = k_cache[layer_idx]
            vL = v_cache[layer_idx]

            # AR: causal, query len = 1, no structured bias needed; we just mask invalid prefix slots
            # Build only key-valid bias: (1,1,1,T+1) but we'll attend only to prefix keys (T) + this token key (1)
            # We'll use full cache + this token as k_full, and rely on key_valid + query-dependent causality for l=1.

            # layer forward: we don't need structural bias; we can do dot_attn with bias == key_valid for prefix and include self key
            # easiest: treat as "TiDAR-like" with step_len=1 but no candidate structure
            # We'll just do dot_product_attention with bias that masks unused prefix keys.
            # Create q,k,v from token
            w_qkv = params["qkv_w"][layer_idx]
            w_o = params["o_w"][layer_idx]
            qkv = jnp.einsum("bld,df->blf", h, w_qkv).astype(dtype)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            q = q.reshape(B, 1, H, Dh)
            k = k.reshape(B, 1, H, Dh)
            v = v.reshape(B, 1, H, Dh)

            k_full = jnp.concatenate([kL, k], axis=1)  # (B,T+1,H,Dh)
            v_full = jnp.concatenate([vL, v], axis=1)

            # key-valid bias for prefix positions only, self key always valid
            k_idx = jnp.arange(T)
            valid = k_idx < prefix_len
            prefix_bias = jnp.where(valid, 0.0, cfg.bias_value).astype(dtype)  # (T,)
            bias = jnp.concatenate([prefix_bias, jnp.zeros((1,), dtype=dtype)], axis=0)  # (T+1,)
            bias = bias[None, None, None, :]  # (1,1,1,T+1)

            y = attn_backend.attn(q=q, k=k_full, v=v_full, bias=bias, causal=False)  # (B,1,H,Dh)
            y = y.reshape(B, 1, D)
            y = jnp.einsum("bld,df->blf", y, w_o).astype(dtype)
            h2 = y  # no ffn here to keep baseline lean

            # write k/v into cache at prefix_len
            kL2 = kL.at[:, prefix_len, :, :].set(k[:, 0, :, :])
            vL2 = vL.at[:, prefix_len, :, :].set(v[:, 0, :, :])
            k_cache = k_cache.at[layer_idx].set(kL2)
            v_cache = v_cache.at[layer_idx].set(vL2)

            return h2, k_cache, v_cache

        h, k_cache, v_cache = jax.lax.fori_loop(0, cfg.num_layers, layer_loop, (h, k_cache, v_cache))

        # unembed logits for this 1 position
        logits = jnp.einsum("bld,vd->blv", h.astype(jnp.float32), emb.astype(jnp.float32))  # (B,1,V)
        next_tok = sample_categorical(logits[:, 0, :], k_step, cfg.temperature, cfg.top_k)  # (B,)

        prefix_len = prefix_len + 1
        tokens_generated = tokens_generated + 1
        return (key, prefix_len, next_tok, k_cache, v_cache, tokens_generated), None

    carry0 = (key, prefix_len0, last_tok, k_cache, v_cache, jnp.array(0, dtype=jnp.int32))
    carryf, _ = jax.lax.scan(step, carry0, xs=None, length=cfg.steps)
    return carryf  # contains tokens_generated


def tidar_decode_benchmark(cfg: Cfg, params: dict, attn_backend: AttnBackend, seed: int):
    """
    TiDAR-ish decode:
      - cache holds committed prefix KV
      - each step computes Q = (K + K^2) tokens (verify + predraft)
      - ALWAYS-ACCEPT mode: commit r=K verify tokens each step and pick candidate r=K as next verify.
      - We only unembed logits for:
          - verify slice (K) for sampling "verify tokens" (in real TiDAR you already have verify tokens from previous step)
          - chosen candidate slice (K) for next verify tokens
      - KV cache update uses K/V computed for verify tokens (first K positions of step), written to cache at [prefix_len:prefix_len+K].
    """
    dtype = _dtype(cfg.dtype)
    key = jax.random.PRNGKey(seed)

    B, T, D = cfg.batch, cfg.context_len, cfg.d_model
    H, Dh = cfg.num_heads, cfg.head_dim
    V = cfg.vocab
    K = cfg.k

    step_len, logical_step_len = tidar_step_lens(K, cfg.pad_q_to_128)
    Q = step_len  # used in attention
    emb = params["emb"]

    # Precompute constant structural bias once (JIT-constant)
    struct_bias = build_tidar_struct_bias(
        context_len=T,
        k=K,
        step_len=Q,
        logical_step_len=logical_step_len,
        bias_value=cfg.bias_value,
        dtype=dtype,
    )  # (1,1,Q,T+Q)

    # Cache per layer
    k_cache = jnp.zeros((cfg.num_layers, B, T, H, Dh), dtype=dtype)
    v_cache = jnp.zeros((cfg.num_layers, B, T, H, Dh), dtype=dtype)

    # Prefill cache with random prompt tokens length prefix_init
    prefix_len0 = jnp.array(cfg.prefix_init, dtype=jnp.int32)
    key, k_tokens = jax.random.split(key)
    prompt_tokens = jax.random.randint(k_tokens, (B, cfg.prefix_init), 0, V, dtype=jnp.int32)
    h_prompt = emb[prompt_tokens]

    def fill_layer(layer_idx, carry):
        k_cache, v_cache = carry
        h = h_prompt
        w_qkv = params["qkv_w"][layer_idx]
        qkv = jnp.einsum("bld,df->blf", h, w_qkv).astype(dtype)
        _, k, v = jnp.split(qkv, 3, axis=-1)
        k = k.reshape(B, cfg.prefix_init, H, Dh)
        v = v.reshape(B, cfg.prefix_init, H, Dh)
        k_cache = k_cache.at[layer_idx, :, :cfg.prefix_init, :, :].set(k)
        v_cache = v_cache.at[layer_idx, :, :cfg.prefix_init, :, :].set(v)
        return k_cache, v_cache

    k_cache, v_cache = jax.lax.fori_loop(0, cfg.num_layers, fill_layer, (k_cache, v_cache))

    # Initial "verify tokens" for first step: sample K tokens from a dummy distribution (simulate prefill)
    key, k_init = jax.random.split(key)
    verify_tokens = jax.random.randint(k_init, (B, K), 0, V, dtype=jnp.int32)

    def step(carry, _):
        key, prefix_len, verify_tokens, k_cache, v_cache, tokens_generated = carry

        # Build step input embeddings:
        # [verify tokens (K)] + [mask tokens (K^2)] + [optional pad zeros]
        verify_emb = emb[verify_tokens]  # (B,K,D)
        mask_emb = jnp.zeros((B, K*K, D), dtype=dtype)  # pretend [MASK] embedding; for speed test, zeros is fine
        step_h = jnp.concatenate([verify_emb, mask_emb], axis=1)  # (B, K+K^2, D)

        if Q > logical_step_len:
            pad = jnp.zeros((B, Q - logical_step_len, D), dtype=dtype)
            step_h = jnp.concatenate([step_h, pad], axis=1)  # (B,Q,D)

        # Forward through layers
        def layer_loop(layer_idx, state):
            h, k_cache, v_cache = state
            kL = k_cache[layer_idx]
            vL = v_cache[layer_idx]

            # forward with structured bias (non-causal)
            h2, k_step, v_step = layer_forward(
                h=h,
                k_cache=kL,
                v_cache=vL,
                prefix_len=prefix_len,
                layer_idx=layer_idx,
                params=params,
                cfg=cfg,
                attn_backend=attn_backend,
                bias_struct=struct_bias,
                causal=False,
                step_len=Q,
            )

            # In always-accept mode, commit first K verify tokens:
            # write K/V for verify positions (first K of k_step/v_step) into prefix cache at prefix_len.
            k_write = k_step[:, :K, :, :]  # (B,K,H,Dh)
            v_write = v_step[:, :K, :, :]

            # dynamic_update_slice along seq axis
            kL = jax.lax.dynamic_update_slice(kL, k_write, (0, prefix_len, 0, 0))
            vL = jax.lax.dynamic_update_slice(vL, v_write, (0, prefix_len, 0, 0))

            k_cache = k_cache.at[layer_idx].set(kL)
            v_cache = v_cache.at[layer_idx].set(vL)
            return h2, k_cache, v_cache

        step_h, k_cache, v_cache = jax.lax.fori_loop(0, cfg.num_layers, layer_loop, (step_h, k_cache, v_cache))

        # Unembed logits only for:
        #  - chosen candidate block (we use r=K => last candidate block)
        # Candidate blocks start at offset K, length K*K
        # block index (r-1) = K-1 => start = K + (K-1)*K
        cand_start = K + (K - 1) * K
        cand_slice = jax.lax.dynamic_slice(step_h, (0, cand_start, 0), (B, K, D))

        # compute logits and sample next verify tokens
        key, k_samp = jax.random.split(key)
        logits_cand = jnp.einsum("bkd,vd->bkv", cand_slice.astype(jnp.float32), emb.astype(jnp.float32))
        next_verify = sample_categorical(logits_cand, k_samp, cfg.temperature, cfg.top_k)  # (B,K)

        prefix_len = prefix_len + K
        tokens_generated = tokens_generated + K
        return (key, prefix_len, next_verify, k_cache, v_cache, tokens_generated), None

    carry0 = (
        key,
        prefix_len0,
        verify_tokens,
        k_cache,
        v_cache,
        jnp.array(0, dtype=jnp.int32),
    )
    carryf, _ = jax.lax.scan(step, carry0, xs=None, length=cfg.steps)
    return carryf


# -----------------------
# Benchmark harness
# -----------------------

def run_bench(cfg: Cfg):
    print(f"[device] {jax.devices()[0]}")
    print(f"[cfg] T={cfg.context_len} P0={cfg.prefix_init} steps={cfg.steps} "
          f"K={cfg.k} layers={cfg.num_layers} d={cfg.d_model} H={cfg.num_heads} Dh={cfg.head_dim} "
          f"backend={cfg.backend} impl={cfg.impl} dtype={cfg.dtype}")

    key = jax.random.PRNGKey(0)
    params = init_params(cfg, key)
    attn_backend = AttnBackend(cfg)

    @jax.jit
    def ar_run():
        return ar_decode_benchmark(cfg, params, attn_backend, seed=42)

    @jax.jit
    def tidar_run():
        return tidar_decode_benchmark(cfg, params, attn_backend, seed=123)

    # warmup
    for _ in range(cfg.warmup):
        out = ar_run()
        _sync(out)
        out = tidar_run()
        _sync(out)

    # timed AR
    t0 = _now()
    for _ in range(cfg.iters):
        out = ar_run()
        _sync(out)
    t1 = _now()
    ar_secs = (t1 - t0) / cfg.iters
    ar_tokens = int(out[-1])  # tokens_generated
    ar_tok_per_s = ar_tokens / ar_secs

    # timed TiDAR
    t0 = _now()
    for _ in range(cfg.iters):
        out2 = tidar_run()
        _sync(out2)
    t1 = _now()
    tidar_secs = (t1 - t0) / cfg.iters
    tidar_tokens = int(out2[-1])
    tidar_tok_per_s = tidar_tokens / tidar_secs

    print("\n=== Results ===")
    print(f"AR:    {ar_tok_per_s:,.1f} tok/s  (avg {ar_secs*1000:.2f} ms per {cfg.steps} steps)")
    print(f"TiDAR: {tidar_tok_per_s:,.1f} tok/s  (avg {tidar_secs*1000:.2f} ms per {cfg.steps} steps)")
    print(f"Speedup (tok/s): {tidar_tok_per_s / max(ar_tok_per_s, 1e-9):.2f}x")
    print("\nNotes:")
    print("- TiDAR here is 'always-accept' and commits K tokens/step; real rejection adds some overhead.")
    print("- This benchmark avoids unembedding logits for K^2 candidates; it only unembeds selected K tokens.")
    print("- If jfa_pallas fails on your shapes, it will fall back to JAX attention. (Watch stderr.)")


def parse_args() -> Cfg:
    p = argparse.ArgumentParser("KV-cache generation benchmark (AR vs TiDAR-ish)")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--prefix-init", type=int, default=None)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--k", type=int, default=16)

    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=8)

    p.add_argument("--backend", type=str, default="jax",
                   choices=["jax", "jfa_triton", "jfa_pallas", "jfa_jax"])
    p.add_argument("--impl", type=str, default="cudnn", choices=["cudnn", "xla"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)

    p.add_argument("--pad-q-to-128", action="store_true")

    args = p.parse_args()
    prefix_init = args.prefix_init
    if prefix_init is None:
        prefix_init = int(round(0.66 * args.context_len))

    return Cfg(
        context_len=args.context_len,
        prefix_init=prefix_init,
        steps=args.steps,
        vocab=args.vocab,
        k=args.k,
        d_model=args.d_model,
        num_heads=args.heads,
        head_dim=args.head_dim,
        num_layers=args.layers,
        backend=args.backend,
        impl=args.impl,
        dtype=args.dtype,
        temperature=args.temperature,
        top_k=args.top_k,
        iters=args.iters,
        warmup=args.warmup,
        pad_q_to_128=bool(args.pad_q_to_128),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_bench(cfg)

