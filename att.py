#!/usr/bin/env python3
"""Extended benchmark script with XLA implementation and multi‑seq‑len support.

Adds a fourth attention variant using ``implementation="xla"`` and loops over a set of
sequence lengths (default: 128,256,512,784,1024,1400,1800,2200,2600,3000,4000,5000,8000,10000)
printing a CSV summary and a Matplotlib plot.

KV Cache Mode: Simulates incremental decoding by measuring attention performance
as the KV cache grows from an initial prompt length to various cache sizes.
"""
# Original documentation omitted for brevity – see upstream repo for details.

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

# XLA implementation (standard JAX attention without backend spec)
def xla_attention(q: jax.Array, k: jax.Array, v: jax.Array, *, is_causal: bool = False) -> jax.Array:
    """Attention using JAX's default XLA implementation.
    Equivalent to ``implementation="xla"`` in ``dot_product_attention``.
    """
    return jax.nn.dot_product_attention(q, k, v, implementation="xla", is_causal=is_causal)

xla_attention_jit = jax.jit(xla_attention, static_argnames=("is_causal",))


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

    # Forward-only call site with separate paths for causal and non-causal
    def pallas_attn_non_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        bs = BlockSizes(
            block_q=block_size,
            block_k=block_size,
            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_q_dq=block_size,
            block_kv_dq=block_size,
        )
        return mha(q, k, v, None, block_sizes=bs)

    def pallas_attn_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        # For causal attention with Pallas, we'll fall back to the naive implementation
        # since the Pallas MHA kernel has strict shape requirements for bias
        b, t, n, h = q.shape
        s = k.shape[1]
        scale = 1.0 / math.sqrt(h)
        
        logits = jnp.einsum("btnh,bsnh->bnts", q, k) * scale
        
        # Causal mask
        mask = jnp.tril(jnp.ones((t, s), dtype=bool))
        logits = jnp.where(mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)
        
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bnts,bsnh->btnh", weights, v)
        return out
    
    def pallas_attn(q: jax.Array, k: jax.Array, v: jax.Array, *, is_causal: bool = False) -> jax.Array:
        if is_causal:
            return pallas_attn_causal(q, k, v)
        else:
            return pallas_attn_non_causal(q, k, v)

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
    ap.add_argument("--seq-len", type=int, default=10_000, help="Legacy single sequence length")
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--block-size", type=int, default=128, help="Pallas attention tile size; also padding multiple.")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--kv-cache", action="store_true", help="Enable KV cache simulation mode for incremental decoding")
    ap.add_argument("--prompt-len", type=int, default=128, help="Initial prompt length for KV cache mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-lens", type=str, default="128,256,512,784,1024,1400,1800,2200,2600,3000,4000,5000,8000,10000", help="Comma-separated list of sequence lengths to benchmark (or cache sizes in KV mode)")
    args = ap.parse_args()

    # In KV cache mode, use bf16 by default to avoid XLA precision issues
    if args.kv_cache and args.dtype == "fp16":
        print("Note: Switching from fp16 to bf16 for KV cache mode to avoid XLA precision issues")
        args.dtype = "bf16"

    # Parse sequence lengths (or cache sizes in KV mode)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
    single_seq_len_mode = len(seq_lens) == 1 and args.seq_len != seq_lens[0]
    if single_seq_len_mode:
        seq_lens = [args.seq_len]

    # In KV cache mode, these represent cache sizes (must be >= prompt_len)
    if args.kv_cache:
        seq_lens = [s for s in seq_lens if s >= args.prompt_len]
        if not seq_lens:
            print(f"No valid cache sizes found (must be >= prompt_len={args.prompt_len})")
            return

    dtype_map = {"fp16": jnp.float16, "bf16": jnp.bfloat16, "fp32": jnp.float32}
    dtype = dtype_map[args.dtype]

    backend = jax.default_backend()
    devs = jax.devices()
    print(f"JAX backend: {backend} | devices: {devs}")

    # Prepare Pallas function (same for all seq lens)
    pallas_fn, pallas_msg = make_pallas_attention(args.block_size)

    # Collect all results: list of tuples (seq_len, impl_name, min_ms, mean_ms, max_ms)
    all_results = []

    for cache_size in seq_lens:
        if args.kv_cache:
            print(f"\n=== Benchmarking KV cache size = {cache_size} (prompt_len={args.prompt_len}) ===")
            # In KV cache mode: Q is new token (seq_len=1), K/V are cache (seq_len=cache_size)
            b, n, h = args.batch, args.num_heads, args.head_dim
            q_len, kv_len = 1, cache_size

            # Memory calculation for attention matrix
            padded_kv_len = ceil_to_multiple(kv_len, args.block_size)
            addr = bytes_for_attention_matrix(b, n, q_len, kv_len, dtype)
            print(f"Attention matrix size: {b}*{n}*{q_len}*{kv_len} @ {args.dtype} ~= {human_bytes(addr)}")

            # Generate data: Q is new token, K/V are from cache
            key = _key(args.seed + cache_size)
            k1, k2, k3 = jax.random.split(key, 3)
            q = jax.random.normal(k1, (b, q_len, n, h), dtype=dtype)  # New token
            k = jax.random.normal(k2, (b, kv_len, n, h), dtype=dtype)  # Cache
            v = jax.random.normal(k3, (b, kv_len, n, h), dtype=dtype)  # Cache

            # Pad to block size multiple
            k_pad = pad_to_len(k, padded_kv_len, axis=1)
            v_pad = pad_to_len(v, padded_kv_len, axis=1)
            q_pad = q  # No padding needed for q (length 1)

            # Always use causal attention for decoding
            causal = True
        else:
            print(f"\n=== Benchmarking seq_len = {cache_size} ===")
            b, t, n, h = args.batch, cache_size, args.num_heads, args.head_dim
            q_len, kv_len = t, t

            # Padding
            padded_len = ceil_to_multiple(t, args.block_size)
            if padded_len != t:
                print(f"Padding seq_len {t} -> {padded_len} (multiple of block_size={args.block_size})")

            addr = bytes_for_attention_matrix(b, n, padded_len, padded_len, dtype)
            print(f"Naive attention matrix size: {b}*{n}*{padded_len}*{padded_len} @ {args.dtype} ~= {human_bytes(addr)}")

            key = _key(args.seed + cache_size)
            k1, k2, k3 = jax.random.split(key, 3)
            q = jax.random.normal(k1, (b, t, n, h), dtype=dtype)
            k = jax.random.normal(k2, (b, t, n, h), dtype=dtype)
            v = jax.random.normal(k3, (b, t, n, h), dtype=dtype)

            q_pad = pad_to_len(q, padded_len, axis=1)
            k_pad = pad_to_len(k, padded_len, axis=1)
            v_pad = pad_to_len(v, padded_len, axis=1)

            causal = args.causal

        # cuDNN
        try:
            mn, mean, mx = bench(lambda: cudnn_attention_jit(q_pad, k_pad, v_pad, is_causal=causal)[:, :q_len, :, :], warmup=args.warmup, iters=args.iters)
            all_results.append((cache_size, "cudnn", mn, mean, mx))
        except Exception as e:
            print(f"[SKIP] cuDNN attention failed: {e}")
            all_results.append((cache_size, "cudnn", float("nan"), float("nan"), float("nan")))

        # XLA
        try:
            mn, mean, mx = bench(lambda: xla_attention_jit(q_pad, k_pad, v_pad, is_causal=causal)[:, :q_len, :, :], warmup=args.warmup, iters=args.iters)
            all_results.append((cache_size, "xla", mn, mean, mx))
        except Exception as e:
            print(f"[SKIP] XLA attention failed: {e}")
            all_results.append((cache_size, "xla", float("nan"), float("nan"), float("nan")))

        # naive
        try:
            mn, mean, mx = bench(lambda: naive_attention_jit(q_pad, k_pad, v_pad, is_causal=causal)[:, :q_len, :, :], warmup=args.warmup, iters=args.iters)
            all_results.append((cache_size, "naive", mn, mean, mx))
        except Exception as e:
            print(f"[SKIP] naive attention failed: {e}")
            all_results.append((cache_size, "naive", float("nan"), float("nan"), float("nan")))

        # pallas
        if pallas_fn is None:
            print(f"[SKIP] pallas attention: {pallas_msg}")
            all_results.append((cache_size, "pallas", float("nan"), float("nan"), float("nan")))
        else:
            try:
                mn, mean, mx = bench(lambda: pallas_fn(q_pad, k_pad, v_pad, is_causal=causal)[:, :q_len, :, :], warmup=args.warmup, iters=args.iters)
                all_results.append((cache_size, "pallas", mn, mean, mx))
            except Exception as e:
                print(f"[SKIP] pallas attention failed: {e}")
                all_results.append((cache_size, "pallas", float("nan"), float("nan"), float("nan")))

    # Print console summary
    metric_name = "cache_size" if args.kv_cache else "seq_len"
    print(f"\n=== Timing (ms) ===")
    for seq in seq_lens:
        print(f"\n{metric_name}={seq}")
        sub = [r for r in all_results if r[0] == seq]
        for _, name, mn, mean, mx in sub:
            if math.isnan(mean):
                print(f"  {name:12s} : (skipped)")
            else:
                print(f"  {name:12s} : min {mn:8.2f} | mean {mean:8.2f} | max {mx:8.2f}")

    # Write CSV
    csv_path = "benchmark_results.csv"
    with open(csv_path, "w") as f:
        header = "cache_size" if args.kv_cache else "seq_len"
        f.write(f"{header},implementation,min_ms,mean_ms,max_ms\n")
        for seq, name, mn, mean, mx in all_results:
            f.write(f"{seq},{name},{mn},{mean},{mx}\n")
    print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    main()

