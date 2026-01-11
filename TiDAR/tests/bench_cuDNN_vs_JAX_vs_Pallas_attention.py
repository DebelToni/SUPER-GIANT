"""
bench_tidar_attention_jax.py

Benchmark a few TiDAR-ish attention strategies in JAX, using a fixed-shape lax.scan loop
to simulate many decode steps.

We test (roughly):
  1) Dense attention with a big structured bias matrix (0 / -1e10) and implementation="cudnn" if possible.
  2) Dense attention with "tight keys" (prefix_len P instead of full context_len T) -- not JIT-stable in real decode,
     but useful as an upper bound for what slicing could buy you.
  3) "Smaller matrices" via split attention calls:
       - one attention for verify tokens
       - K attentions for candidate blocks
     (launch overhead is real; this is just to measure tradeoffs)
  4) Optional Pallas attention (only if a matching helper exists in your installed JAX).

Usage:
  python bench_tidar_attention_jax.py --context-len 1024 --k 16 --prefix-len 256 --steps 128 --dtype bf16

Notes:
- This script assumes fixed shapes to allow JIT.
- For real TiDAR, you'd also want to avoid unembedding vocab logits for K^2 slots. This is attention-only.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# ----------------------------
# Config / helpers
# ----------------------------

@dataclass(frozen=True)
class BenchCfg:
    context_len: int = 1024   # T
    k: int = 16               # K
    prefix_len: int = 256     # P (how many prefix cache positions are "valid")
    num_heads: int = 16
    head_dim: int = 64
    batch: int = 1
    steps: int = 128          # number of decode-steps simulated by scan
    warmup: int = 1           # warmup runs
    iters: int = 10           # timed runs
    dtype: str = "bf16"       # bf16|fp16|fp32
    impl: str = "cudnn"       # cudnn|xla (cudnn only relevant on GPU)
    bias_value: float = -1.0e10


def _dtype(name: str) -> jnp.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if name in ("fp16", "float16"):
        return jnp.float16
    if name in ("fp32", "float32"):
        return jnp.float32
    raise ValueError(f"Unknown dtype: {name}")


def _sync(x):
    """Block until ready (device sync)."""
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def _now() -> float:
    return time.perf_counter()


def try_attention_impl(impl: str) -> str:
    # On CPU, cudnn isn't valid.
    dev = jax.devices()[0]
    if impl == "cudnn" and dev.platform != "gpu":
        return "xla"
    return impl


# ----------------------------
# Build TiDAR-ish structured masks (fixed shapes)
# ----------------------------

def build_struct_bias_full(
    *,
    context_len: int,  # T
    k: int,            # K
    bias_value: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Build a fixed, structured attention bias for queries = step tokens only,
    keys = [prefix_cache(T) | step_tokens(Q)].

    Layout:
      step tokens length Q = K + K*K
        - first K queries are verify
        - remaining K*K queries are candidates, flattened as K blocks of size K

      keys length KV = T + Q
        - first T keys are prefix cache
        - next K keys are verify keys (same step tokens)
        - remaining K*K keys are candidate keys (same step tokens)

    Mask:
      - verify queries can attend to all prefix keys and causal verify keys
      - verify queries cannot attend to any candidate keys
      - candidate queries can attend to all prefix keys
      - candidate queries can attend to verify keys strictly before cand_r (cand block index + 1)
      - candidate queries can attend bidirectionally within their own candidate block only
    """
    T = context_len
    K = k
    Q = K + K * K
    KV = T + Q

    q = jnp.arange(Q)[:, None]      # (Q,1)
    kk = jnp.arange(KV)[None, :]    # (1,KV)

    # Key region classification
    k_is_prefix = kk < T
    k_is_verify = (kk >= T) & (kk < T + K)
    k_is_cand = kk >= (T + K)

    k_verify_pos = kk - T  # 0..K-1 (valid only where k_is_verify)
    k_cand_offset = kk - (T + K)   # 0..K*K-1 (valid only where k_is_cand)
    k_cand_block = k_cand_offset // K

    # Query region classification
    q_is_verify = q < K
    q_is_cand = q >= K

    q_verify_pos = q  # 0..K-1 for verify queries
    q_cand_offset = q - K
    q_cand_block = q_cand_offset // K     # 0..K-1
    cand_r = q_cand_block + 1             # 1..K

    allow_verify = (
        (k_is_prefix & q_is_verify)
        | (k_is_verify & q_is_verify & (k_verify_pos <= q_verify_pos))
    )
    allow_cand = (
        (k_is_prefix & q_is_cand)
        | (k_is_verify & q_is_cand & (k_verify_pos < cand_r))
        | (k_is_cand & q_is_cand & (k_cand_block == q_cand_block))
    )

    allow = allow_verify | allow_cand

    bias = jnp.where(allow, jnp.array(0.0, dtype=dtype), jnp.array(bias_value, dtype=dtype))
    bias = bias[None, None, :, :]  # (1,1,Q,KV)
    return bias


def build_key_valid_bias(
    *,
    context_len: int,
    prefix_len: int,
    step_len: int,
    bias_value: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Bias that disables unused prefix cache slots beyond prefix_len.
    Keys are [prefix_cache(T) | step_tokens(step_len)].
    """
    T = context_len
    Q = step_len

    k = jnp.arange(T)
    valid = k < prefix_len
    prefix_bias = jnp.where(valid, jnp.array(0.0, dtype=dtype), jnp.array(bias_value, dtype=dtype))
    prefix_bias = prefix_bias[None, None, None, :]  # (1,1,1,T)

    step_bias = jnp.zeros((1, 1, 1, Q), dtype=dtype)
    key_bias = jnp.concatenate([prefix_bias, step_bias], axis=-1)  # (1,1,1,T+Q)
    return key_bias


# ----------------------------
# Attention kernels to benchmark
# ----------------------------

def dot_attn(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    impl: str,
    is_causal: bool = False,
) -> jnp.ndarray:
    # q,k,v: (B, Lq, H, D)
    # bias: broadcastable to (B, H, Lq, Lk)
    return jax.nn.dot_product_attention(
        q, k, v,
        bias=bias,
        is_causal=is_causal,
        implementation=impl,
    )


# ----------------------------
# Bench variants
# ----------------------------

def make_inputs(cfg: BenchCfg) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create:
      prefix K/V (as full k,v arrays)
      step K/V/Q
    We simulate these as random tensors.
    """
    dtype = _dtype(cfg.dtype)
    key = jax.random.PRNGKey(0)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    B, H, D = cfg.batch, cfg.num_heads, cfg.head_dim
    T, K = cfg.context_len, cfg.k
    Q = K + K * K  # step_len queries

    # prefix keys/values (B, T, H, D)
    k_prefix = jax.random.normal(k1, (B, T, H, D), dtype=dtype)
    v_prefix = jax.random.normal(k2, (B, T, H, D), dtype=dtype)

    # step keys/values (B, Q, H, D)
    k_step = jax.random.normal(k3, (B, Q, H, D), dtype=dtype)
    v_step = jax.random.normal(k4, (B, Q, H, D), dtype=dtype)

    # initial queries (B, Q, H, D)
    q0 = jax.random.normal(jax.random.PRNGKey(123), (B, Q, H, D), dtype=dtype)

    return q0, k_prefix, v_prefix, k_step, v_step


def bench_dense_bigmask_scan(cfg: BenchCfg):
    """
    One attention call per step:
      queries = step tokens (Q)
      keys    = prefix_cache(T) + step tokens (Q)
      bias    = (struct_bias + key_valid_bias)
    """
    dtype = _dtype(cfg.dtype)
    impl = try_attention_impl(cfg.impl)
    T, K = cfg.context_len, cfg.k
    Q = K + K * K
    KV = T + Q

    q0, k_prefix, v_prefix, k_step, v_step = make_inputs(cfg)

    struct_bias = build_struct_bias_full(context_len=T, k=K, bias_value=cfg.bias_value, dtype=dtype)
    key_valid_bias = build_key_valid_bias(context_len=T, prefix_len=cfg.prefix_len, step_len=Q,
                                          bias_value=cfg.bias_value, dtype=dtype)
    bias = struct_bias + key_valid_bias  # (1,1,Q,KV)

    k_full = jnp.concatenate([k_prefix, k_step], axis=1)  # (B,KV,H,D)
    v_full = jnp.concatenate([v_prefix, v_step], axis=1)

    def step_fn(q, _):
        # q: (B,Q,H,D)
        y = dot_attn(q, k_full, v_full, bias=bias, impl=impl, is_causal=False)
        # update q to prevent trivial compilation oddities; keep shape fixed
        q_next = jnp.tanh(y)
        return q_next, y

    @jax.jit
    def run(q_init):
        qf, ys = jax.lax.scan(step_fn, q_init, xs=None, length=cfg.steps)
        return qf, ys

    # Warmup compile + run
    out = run(q0)
    _sync(out)

    # Timed
    t0 = _now()
    for _ in range(cfg.iters):
        out = run(q0)
        _sync(out)
    t1 = _now()

    return {
        "name": f"dense_bigmask_scan(impl={impl})",
        "seconds_total": t1 - t0,
        "seconds_per_iter": (t1 - t0) / cfg.iters,
        "steps": cfg.steps,
        "notes": f"Q={Q}, KV={KV}, one attn/layer-step, dense bias precomputed",
    }


def bench_dense_tightkeys_scan(cfg: BenchCfg):
    """
    Same as dense_bigmask_scan, but keys only use prefix_len P instead of full context_len T.
    This is not shape-stable if P changes in real decode, but good as a performance upper bound.
    """
    dtype = _dtype(cfg.dtype)
    impl = try_attention_impl(cfg.impl)
    P, K = cfg.prefix_len, cfg.k
    Q = K + K * K
    KV = P + Q

    q0, k_prefix, v_prefix, k_step, v_step = make_inputs(cfg)

    # Slice prefix to P (tight)
    k_prefix_t = k_prefix[:, :P, :, :]
    v_prefix_t = v_prefix[:, :P, :, :]

    # Build a "tight" struct bias: same query logic, but prefix keys only P.
    # Easiest: build full struct then slice prefix dimension. But we build directly.
    # We'll reuse build_struct_bias_full with context_len=P.
    struct_bias = build_struct_bias_full(context_len=P, k=K, bias_value=cfg.bias_value, dtype=dtype)
    # No key_valid_bias needed because prefix keys are exactly valid.
    bias = struct_bias  # (1,1,Q,KV)

    k_full = jnp.concatenate([k_prefix_t, k_step], axis=1)
    v_full = jnp.concatenate([v_prefix_t, v_step], axis=1)

    def step_fn(q, _):
        y = dot_attn(q, k_full, v_full, bias=bias, impl=impl, is_causal=False)
        return jnp.tanh(y), y

    @jax.jit
    def run(q_init):
        qf, ys = jax.lax.scan(step_fn, q_init, xs=None, length=cfg.steps)
        return qf, ys

    out = run(q0)
    _sync(out)

    t0 = _now()
    for _ in range(cfg.iters):
        out = run(q0)
        _sync(out)
    t1 = _now()

    return {
        "name": f"dense_tightkeys_scan(impl={impl})",
        "seconds_total": t1 - t0,
        "seconds_per_iter": (t1 - t0) / cfg.iters,
        "steps": cfg.steps,
        "notes": f"Q={Q}, KV={KV} (P+Q), one attn/step, dense bias precomputed",
    }


def bench_split_calls_scan(cfg: BenchCfg):
    """
    "Smaller matrices" by splitting attention into:
      - verify attention: q=K, kv=T+K  (prefix + verify)
      - K candidate attentions: each q=K, kv=T+2K  (prefix + verify + that cand block)
    This is NOT how you'd implement for speed usually, but it benchmarks the tradeoff.

    We keep keys fixed to full context_len T and apply key_valid_bias.
    """
    dtype = _dtype(cfg.dtype)
    impl = try_attention_impl(cfg.impl)
    T, P, K = cfg.context_len, cfg.prefix_len, cfg.k
    Q = K + K * K

    q0, k_prefix, v_prefix, k_step, v_step = make_inputs(cfg)

    # Split step keys into verify + candidates
    k_verify = k_step[:, :K, :, :]
    v_verify = v_step[:, :K, :, :]
    k_cands = k_step[:, K:, :, :].reshape(cfg.batch, K, K, cfg.num_heads, cfg.head_dim)  # (B, Kblocks, K, H, D)
    v_cands = v_step[:, K:, :, :].reshape(cfg.batch, K, K, cfg.num_heads, cfg.head_dim)

    # Key-valid bias for prefix (length T)
    # We'll build per-call biases.
    def prefix_key_bias(Tlocal):
        k = jnp.arange(Tlocal)
        valid = k < P
        b = jnp.where(valid, jnp.array(0.0, dtype=dtype), jnp.array(cfg.bias_value, dtype=dtype))
        return b[None, None, None, :]  # (1,1,1,Tlocal)

    prefix_bias_T = prefix_key_bias(T)

    # Verify call: kv = T + K
    # No candidates visible.
    kv_verify = T + K
    # Build a simple bias:
    # - verify queries can attend to prefix keys
    # - verify queries causal over verify keys
    q_idx = jnp.arange(K)[:, None]
    k_idx = jnp.arange(kv_verify)[None, :]
    k_is_prefix = k_idx < T
    k_is_verify = k_idx >= T
    k_verify_pos = k_idx - T
    allow = k_is_prefix | (k_is_verify & (k_verify_pos <= q_idx))
    verify_bias = jnp.where(allow, 0.0, cfg.bias_value).astype(dtype)[None, None, :, :]  # (1,1,K,T+K)
    # add prefix valid bias (broadcast to K queries)
    verify_bias = verify_bias + jnp.concatenate([prefix_bias_T, jnp.zeros((1,1,1,K), dtype=dtype)], axis=-1)

    k_full_verify = jnp.concatenate([k_prefix, k_verify], axis=1)  # (B,T+K,H,D)
    v_full_verify = jnp.concatenate([v_prefix, v_verify], axis=1)

    # Candidate calls: each block i uses kv = T + K + K = T + 2K
    kv_cand = T + 2 * K
    # Candidate bias template: (queries K) x (keys T+2K)
    # Candidate can attend prefix, subset verify (< r), and within its own cand block bidir.
    # We'll build biases per block index i because cand_r depends on i.
    k_idx2 = jnp.arange(kv_cand)[None, :]
    k_is_prefix2 = k_idx2 < T
    k_is_verify2 = (k_idx2 >= T) & (k_idx2 < T + K)
    k_is_cand2 = k_idx2 >= (T + K)
    k_verify_pos2 = k_idx2 - T
    k_cand_pos2 = k_idx2 - (T + K)

    # prefix valid bias for kv_cand
    cand_key_valid = jnp.concatenate([prefix_bias_T, jnp.zeros((1,1,1,2*K), dtype=dtype)], axis=-1)

    def cand_bias_for_block(i: int) -> jnp.ndarray:
        r = i + 1  # 1..K
        base_allow = (
            k_is_prefix2
            | (k_is_verify2 & (k_verify_pos2 < r))
            | (k_is_cand2)  # within-block bidir: since only this block's cand tokens are present, allow all cand keys
        )
        allow = jnp.broadcast_to(base_allow, (K, kv_cand))
        b = jnp.where(allow, 0.0, cfg.bias_value).astype(dtype)[None, None, :, :]
        return b + cand_key_valid

    # Precompute candidate biases for all blocks (constant)
    cand_biases = jnp.stack([cand_bias_for_block(i)[0,0,:,:] for i in range(K)], axis=0)  # (K, K, kv_cand)
    cand_biases = cand_biases[:, None, None, :, :]  # (K,1,1,K,kv_cand)

    def step_fn(q, _):
        # q: (B,Q,H,D)
        # Use dynamic_slice instead of dynamic indexing
        q_verify = jax.lax.dynamic_slice(q, (0, 0, 0, 0), (cfg.batch, K, cfg.num_heads, cfg.head_dim))

        # 1) verify attention
        y_verify = dot_attn(q_verify, k_full_verify, v_full_verify, bias=verify_bias, impl=impl, is_causal=False)

        # 2) candidate attentions (K calls)
        # For each block i:
        #   queries are that block's q slice (K)
        #   keys are prefix+verify+that cand block
        # We'll loop in JAX using lax.fori_loop to keep it JIT-able.
        def cand_body(i, acc):
            # Use dynamic_slice instead of dynamic indexing
            start_idx = K + i * K
            q_c = jax.lax.dynamic_slice(q, (0, start_idx, 0, 0), (cfg.batch, K, cfg.num_heads, cfg.head_dim))
            
            # For k_cands and v_cands, use dynamic_slice on the block dimension
            k_c_block = jax.lax.dynamic_slice(k_cands, (0, i, 0, 0, 0), (cfg.batch, 1, K, cfg.num_heads, cfg.head_dim))
            k_c_block = k_c_block.squeeze(axis=1)  # (B, K, H, D)
            
            v_c_block = jax.lax.dynamic_slice(v_cands, (0, i, 0, 0, 0), (cfg.batch, 1, K, cfg.num_heads, cfg.head_dim))
            v_c_block = v_c_block.squeeze(axis=1)  # (B, K, H, D)
            
            k_c = jnp.concatenate([k_prefix, k_verify, k_c_block], axis=1)
            v_c = jnp.concatenate([v_prefix, v_verify, v_c_block], axis=1)
            
            # Use dynamic_slice for bias as well
            bias_i = jax.lax.dynamic_slice(cand_biases, (i, 0, 0, 0, 0), (1, 1, 1, K, kv_cand))
            bias_i = bias_i.squeeze(axis=0)  # (1,1,K,kv_cand)
            
            y_c = dot_attn(q_c, k_c, v_c, bias=bias_i, impl=impl, is_causal=False)
            
            # Use dynamic_update_slice instead of .at[].set()
            acc = jax.lax.dynamic_update_slice(acc, y_c, (0, i*K, 0, 0))
            return acc

        y_cands_out = jnp.zeros((cfg.batch, K*K, cfg.num_heads, cfg.head_dim), dtype=dtype)
        y_cands_out = jax.lax.fori_loop(0, K, cand_body, y_cands_out)

        # stitch outputs back into Q-shaped tensor
        y_full = jnp.concatenate([y_verify, y_cands_out], axis=1)
        q_next = jnp.tanh(y_full)
        return q_next, y_full

    @jax.jit
    def run(q_init):
        qf, ys = jax.lax.scan(step_fn, q_init, xs=None, length=cfg.steps)
        return qf, ys

    out = run(q0)
    _sync(out)

    t0 = _now()
    for _ in range(cfg.iters):
        out = run(q0)
        _sync(out)
    t1 = _now()

    return {
        "name": f"split_calls_scan(impl={impl})",
        "seconds_total": t1 - t0,
        "seconds_per_iter": (t1 - t0) / cfg.iters,
        "steps": cfg.steps,
        "notes": f"verify: q=K,kv=T+K; candidates: K calls of q=K,kv=T+2K; T={T},K={K},Q={Q}",
    }

def bench_pallas_optional(cfg: BenchCfg):
    """
    Optional: try to use a Pallas attention implementation if present in your JAX.

    IMPORTANT:
    - Pallas APIs vary across JAX versions.
    - This is provided as a "probe" harness, not guaranteed to run everywhere.

    If no suitable Pallas attention is importable, we return None.
    """
    try:
        # Some JAX builds include GPU attention in pallas ops; many don't.
        # Try a couple of common-ish import paths.
        from jax.experimental.pallas.ops.gpu import attention as pallas_attention  # type: ignore
    except Exception:
        return None

    dtype = _dtype(cfg.dtype)
    impl = try_attention_impl(cfg.impl)
    T, K = cfg.context_len, cfg.k
    Q = K + K * K
    KV = T + Q

    q0, k_prefix, v_prefix, k_step, v_step = make_inputs(cfg)
    k_full = jnp.concatenate([k_prefix, k_step], axis=1)
    v_full = jnp.concatenate([v_prefix, v_step], axis=1)

    struct_bias = build_struct_bias_full(context_len=T, k=K, bias_value=cfg.bias_value, dtype=dtype)
    key_valid_bias = build_key_valid_bias(context_len=T, prefix_len=cfg.prefix_len, step_len=Q,
                                          bias_value=cfg.bias_value, dtype=dtype)
    bias = struct_bias + key_valid_bias  # (1,1,Q,KV)

    # pallas_attention APIs differ; many expect packed QKV and no arbitrary bias.
    # We'll just demonstrate a pattern: fall back to dot_product_attention if pallas can't accept bias.
    # You can adapt this to your local pallas attention signature.

    def step_fn(q, _):
        # Attempt pallas attention call if supports bias; else fallback
        y = None
        try:
            # Hypothetical signature; adjust to your local version if needed.
            y = pallas_attention.mha(q, k_full, v_full, mask=bias)  # type: ignore
        except Exception:
            y = dot_attn(q, k_full, v_full, bias=bias, impl=impl, is_causal=False)

        return jnp.tanh(y), y

    @jax.jit
    def run(q_init):
        qf, ys = jax.lax.scan(step_fn, q_init, xs=None, length=cfg.steps)
        return qf, ys

    out = run(q0)
    _sync(out)

    t0 = _now()
    for _ in range(cfg.iters):
        out = run(q0)
        _sync(out)
    t1 = _now()

    return {
        "name": f"pallas_optional_scan(fallback_impl={impl})",
        "seconds_total": t1 - t0,
        "seconds_per_iter": (t1 - t0) / cfg.iters,
        "steps": cfg.steps,
        "notes": "Uses pallas_attention if available; otherwise falls back to dot_product_attention.",
    }


# ----------------------------
# CLI / runner
# ----------------------------

def parse_args() -> BenchCfg:
    p = argparse.ArgumentParser("TiDAR attention benchmark (JAX)")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--prefix-len", type=int, default=256)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--impl", type=str, default="cudnn", choices=["cudnn", "xla"])
    args = p.parse_args()

    return BenchCfg(
        context_len=args.context_len,
        k=args.k,
        prefix_len=args.prefix_len,
        num_heads=args.heads,
        head_dim=args.head_dim,
        batch=args.batch,
        steps=args.steps,
        iters=args.iters,
        dtype=args.dtype,
        impl=args.impl,
    )


def main():
    cfg = parse_args()

    dev = jax.devices()[0]
    print(f"[device] {dev.platform} :: {dev}")
    print(f"[cfg] T={cfg.context_len}, K={cfg.k}, P={cfg.prefix_len}, Q={cfg.k + cfg.k*cfg.k}, "
          f"heads={cfg.num_heads}, d={cfg.head_dim}, steps={cfg.steps}, iters={cfg.iters}, "
          f"dtype={cfg.dtype}, impl={cfg.impl}")

    results = []
    results.append(bench_dense_bigmask_scan(cfg))
    results.append(bench_dense_tightkeys_scan(cfg))
    results.append(bench_split_calls_scan(cfg))

    pallas_res = bench_pallas_optional(cfg)
    if pallas_res is not None:
        results.append(pallas_res)
    else:
        print("[pallas] not available (skipping)")

    print("\n=== Results ===")
    for r in results:
        print(f"- {r['name']}")
        print(f"  total: {r['seconds_total']:.4f}s | per-iter: {r['seconds_per_iter']:.6f}s "
              f"| per-step: {(r['seconds_per_iter']/cfg.steps):.8f}s")
        print(f"  notes: {r['notes']}")

    print("\nTip: run with --impl xla and --impl cudnn (on GPU) to see how much mask+bias impacts kernel choice.")
    print("Tip: if split_calls is slower, it's usually launch overhead dominating. Dense + precomputed bias often wins in JAX.")


if __name__ == "__main__":
    main()

