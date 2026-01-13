#!/usr/bin/env python3
"""
bench_attention_jax.py

Benchmarks 3 attention forward implementations on a "huge" sequence (default seq_len=10000):

1) jax.nn.dot_product_attention(..., implementation="cudnn")  (cuDNN FlashAttention backend)
2) A naive JAX attention forward (materializes the [T,S] attention matrix) wrapped in jax.jit
3) JAX Pallas (Triton backend) FlashAttention-style kernel via:
     jax.experimental.pallas.ops.gpu.attention.mha

Notes
-----
- The cuDNN and Pallas variants require an NVIDIA GPU + compatible jaxlib build.
- For Pallas GPU attention, sequence lengths are often happiest when padded to a multiple
  of the chosen block size. This script pads Q/K/V to the next multiple of block_size,
  runs all 3 methods on the padded tensors for apples-to-apples timing, then slices back
  to the requested seq_len.
- The naive implementation can use a lot of memory: it builds an attention matrix of
  shape (B, N, T, S). With B=1, N=8, T=S=10000, that is 800M elements.

Example
-------
  python bench_attention_jax.py --seq-len 10000 --num-heads 8 --head-dim 64 --dtype fp16 --iters 5

If you see GPU OOM, try:
  - smaller --num-heads / --seq-len
  - dtype fp16
  - export XLA_PYTHON_CLIENT_PREALLOCATE=false
  - export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Avoid grabbing the whole GPU up-front (best set in your shell before running).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp


def _key(seed: int):
    # Newer JAX has jax.random.key; older has PRNGKey.
    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    return jax.random.PRNGKey(seed)


def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pad_to_len(x: jax.Array, target_len: int, axis: int = 1) -> jax.Array:
    """Zero-pad tensor x along 'axis' to length target_len."""
    cur = x.shape[axis]
    if cur == target_len:
        return x
    if cur > target_len:
        raise ValueError(f"pad_to_len: current len {cur} > target_len {target_len}")
    pad_amt = target_len - cur
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_amt)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0)


def bytes_for_attention_matrix(b: int, n: int, t: int, s: int, dtype: jnp.dtype) -> int:
    return b * n * t * s * jnp.dtype(dtype).itemsize


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f}{u}"
        f /= 1024.0
    return f"{f:.2f}B"


def naive_attention(q: jax.Array, k: jax.Array, v: jax.Array, *, is_causal: bool = False) -> jax.Array:
    """
    Naive scaled dot-product attention:
      softmax(q @ k^T / sqrt(H)) @ v

    Shapes:
      q: (B, T, N, H)
      k: (B, S, N, H)
      v: (B, S, N, H)
    Returns:
      out: (B, T, N, H)
    """
    h = q.shape[-1]
    scale = 1.0 / math.sqrt(h)

    # logits: (B, N, T, S)
    logits = jnp.einsum("btnh,bsnh->bnts", q, k) * scale

    if is_causal:
        # Causal mask: allow attending only to <= current position.
        # (T, S) boolean, broadcast to (B, N, T, S)
        t = q.shape[1]
        s = k.shape[1]
        causal = jnp.tril(jnp.ones((t, s), dtype=bool))
        logits = jnp.where(causal[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bnts,bsnh->btnh", weights, v)
    return out


naive_attention_jit = jax.jit(naive_attention, static_argnames=("is_causal",))


def cudnn_attention(q: jax.Array, k: jax.Array, v: jax.Array, *, is_causal: bool = False) -> jax.Array:
    """
    cuDNN-backed attention via jax.nn.dot_product_attention(implementation="cudnn").
    """
    return jax.nn.dot_product_attention(q, k, v, implementation="cudnn", is_causal=is_causal)


cudnn_attention_jit = jax.jit(cudnn_attention, static_argnames=("is_causal",))


@dataclass
class PallasMHA:
    mha: Callable
    BlockSizes: type


def try_import_pallas_mha() -> Optional[PallasMHA]:
    """
    Import the Pallas GPU attention kernel if available.

    Expected import path (as used in the JAX repo / issues):
      from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha
    """
    try:
        from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha  # type: ignore
        return PallasMHA(mha=mha, BlockSizes=BlockSizes)
    except Exception:
        return None


def make_pallas_attention(block_size: int) -> Tuple[Optional[Callable], str]:
    """
    Returns (callable, message). If callable is None, message explains why.
    """
    pmha = try_import_pallas_mha()
    if pmha is None:
        return None, "Pallas GPU attention kernel not available in this JAX build."

    BlockSizes = pmha.BlockSizes
    mha = pmha.mha

    # Forward-only call site.
    def pallas_attn(q: jax.Array, k: jax.Array, v: jax.Array, *, is_causal: bool = False) -> jax.Array:
        if is_causal:
            # The built-in GPU mha kernel API is evolving; many versions only support non-causal.
            # You can still benchmark non-causal here, or adapt to the kernel's causal options if present.
            raise NotImplementedError("This script's Pallas path is wired for non-causal attention.")
        bs = BlockSizes(
            block_q=block_size,
            block_k=block_size,
            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_q_dq=block_size,
            block_kv_dq=block_size,
        )
        # bias=None
        return mha(q, k, v, None, block_sizes=bs)

    return jax.jit(pallas_attn, static_argnames=("is_causal",)), "ok"


def bench(fn: Callable[[], jax.Array], *, warmup: int, iters: int) -> Tuple[float, float, float]:
    """
    Returns (min_ms, mean_ms, max_ms) for 'iters' timed runs (after warmup).
    """
    # Warmup (compile + first run)
    for _ in range(warmup):
        y = fn()
        y.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        y.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return min(times), sum(times) / len(times), max(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=10_000)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--block-size", type=int, default=128, help="Pallas attention tile size; also padding multiple.")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype_map = {"fp16": jnp.float16, "bf16": jnp.bfloat16, "fp32": jnp.float32}
    dtype = dtype_map[args.dtype]

    backend = jax.default_backend()
    devs = jax.devices()
    print(f"JAX backend: {backend} | devices: {devs}")

    b, t, n, h = args.batch, args.seq_len, args.num_heads, args.head_dim
    s = t  # self-attn for this benchmark

    # Padding to help Pallas kernels and keep shapes aligned across all 3 paths.
    padded_len = ceil_to_multiple(t, args.block_size)
    if padded_len != t:
        print(f"Padding seq_len {t} -> {padded_len} (multiple of block_size={args.block_size})")

    # Rough memory estimate for the naive attention matrix.
    attn_bytes = bytes_for_attention_matrix(b, n, padded_len, padded_len, dtype)
    print(f"Naive attention matrix size (B*N*T*S): {b}*{n}*{padded_len}*{padded_len} @ {args.dtype} ~= {human_bytes(attn_bytes)}")

    key = _key(args.seed)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (b, t, n, h), dtype=dtype)
    k = jax.random.normal(k2, (b, s, n, h), dtype=dtype)
    v = jax.random.normal(k3, (b, s, n, h), dtype=dtype)

    # Pad along sequence axis=1
    q_pad = pad_to_len(q, padded_len, axis=1)
    k_pad = pad_to_len(k, padded_len, axis=1)
    v_pad = pad_to_len(v, padded_len, axis=1)

    # 1) cuDNN dot_product_attention
    def run_cudnn():
        out = cudnn_attention_jit(q_pad, k_pad, v_pad, is_causal=args.causal)
        return out[:, :t, :, :]

    # 2) naive
    def run_naive():
        out = naive_attention_jit(q_pad, k_pad, v_pad, is_causal=args.causal)
        return out[:, :t, :, :]

    # 3) Pallas flash-attention style kernel
    pallas_fn, pallas_msg = make_pallas_attention(args.block_size)

    def run_pallas():
        assert pallas_fn is not None
        out = pallas_fn(q_pad, k_pad, v_pad, is_causal=args.causal)
        return out[:, :t, :, :]

    # Benchmark
    results = []

    # cuDNN
    try:
        mn, mean, mx = bench(run_cudnn, warmup=args.warmup, iters=args.iters)
        results.append(("jax.nn.dot_product_attention (cudnn)", mn, mean, mx))
    except Exception as e:
        results.append(("jax.nn.dot_product_attention (cudnn)", float("nan"), float("nan"), float("nan")))
        print(f"[SKIP] cuDNN attention failed: {e}")

    # naive
    try:
        mn, mean, mx = bench(run_naive, warmup=args.warmup, iters=args.iters)
        results.append(("naive jitted attention", mn, mean, mx))
    except Exception as e:
        results.append(("naive jitted attention", float("nan"), float("nan"), float("nan")))
        print(f"[SKIP] naive attention failed (likely OOM): {e}")

    # pallas
    if pallas_fn is None:
        results.append((f"pallas mha (flash-attn style) [{pallas_msg}]", float("nan"), float("nan"), float("nan")))
        print(f"[SKIP] pallas attention: {pallas_msg}")
    else:
        try:
            mn, mean, mx = bench(run_pallas, warmup=args.warmup, iters=args.iters)
            results.append(("pallas mha (flash-attn style)", mn, mean, mx))
        except Exception as e:
            results.append(("pallas mha (flash-attn style)", float("nan"), float("nan"), float("nan")))
            print(f"[SKIP] pallas attention failed: {e}")

    # Print summary
    print("\n=== Timing (ms) ===")
    width = max(len(r[0]) for r in results)
    for name, mn, mean, mx in results:
        if math.isnan(mean):
            print(f"{name:<{width}} : (skipped)")
        else:
            print(f"{name:<{width}} : min {mn:8.2f} | mean {mean:8.2f} | max {mx:8.2f}")

    # Optional correctness sanity-check (only if all 3 worked)
    try:
        y_cudnn = run_cudnn()
        y_naive = run_naive()
        max_diff_cudnn = jnp.max(jnp.abs(y_cudnn - y_naive)).item()
        print(f"\nmax|cudnn - naive| = {max_diff_cudnn:.6g}")

        if pallas_fn is not None:
            y_pallas = run_pallas()
            max_diff_pallas = jnp.max(jnp.abs(y_pallas - y_naive)).item()
            print(f"max|pallas - naive| = {max_diff_pallas:.6g}")
    except Exception as e:
        print(f"\n(correctness check skipped: {e})")


if __name__ == "__main__":
    main()

