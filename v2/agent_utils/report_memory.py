#!/usr/bin/env python
"""
Compute parameter and KV-cache size breakdown for the Gemma-2/BgGPT 2.6B model
using the fp32 NPZ checkpoint.

Outputs:
  - Parameter group sizes (MB), sorted by size.
  - KV cache size for a given context length and batch size.
  - Approximate total runtime memory footprint at the given sequence length.

Assumptions:
  - Checkpoint is a flat NPZ with keys like "TinyTransformerBlock_X/..."
  - Compute dtype for params is float32 (4 bytes).
  - KV cache dtype is assumed to match compute dtype (float32 here).
  - Biases are absent (as in current model). Scale params are small.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def group_params(params: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Group param sizes into high-level buckets."""
    totals = {
        "embedding": 0,
        "attention_qkv": 0,
        "attention_out": 0,
        "mlp_fc1": 0,
        "mlp_fc2": 0,
        "norm_scales": 0,
        "other": 0,
    }
    for k, v in params.items():
        size = v.nbytes
        if k.startswith("Embed_0/embedding"):
            totals["embedding"] += size
        elif "qkv_proj/kernel" in k:
            totals["attention_qkv"] += size
        elif "o_proj/kernel" in k:
            totals["attention_out"] += size
        elif "/fc1/kernel" in k:
            totals["mlp_fc1"] += size
        elif "/fc2/kernel" in k:
            totals["mlp_fc2"] += size
        elif "rms" in k or "final_norm" in k:
            totals["norm_scales"] += size
        else:
            totals["other"] += size
    return totals


def kv_cache_bytes(
    batch_size: int,
    n_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_len: int,
    dtype_bytes: int,
) -> int:
    # KV cache stores keys and values: shape (batch, layers, 2, num_kv_heads, context, head_dim)
    return batch_size * n_layers * 2 * num_kv_heads * context_len * head_dim * dtype_bytes


def activation_buffer_bytes(
    batch_size: int,
    context_len: int,
    hidden_size: int,
    dtype_bytes: int,
) -> int:
    # Rough buffer for a couple of activations (input + intermediate). This is a heuristic.
    return batch_size * context_len * hidden_size * dtype_bytes * 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to fp32 NPZ checkpoint.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=2048,
        help="Sequence length to estimate KV cache for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for KV cache estimate.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=26,
        help="Number of transformer layers.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=4,
        help="Number of KV heads (GQA).",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=256,
        help="Head dimension.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=2304,
        help="Model hidden size.",
    )
    parser.add_argument(
        "--dtype_bytes",
        type=int,
        default=4,
        help="Bytes per element (4 for fp32).",
    )
    args = parser.parse_args()

    params = load_npz(args.checkpoint)
    grouped = group_params(params)

    print("Parameter group sizes (MB):")
    for name, size_bytes in sorted(grouped.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {bytes_to_mb(size_bytes):,.2f} MB")

    kv_bytes = kv_cache_bytes(
        batch_size=args.batch_size,
        n_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        context_len=args.context_len,
        dtype_bytes=args.dtype_bytes,
    )
    act_bytes = activation_buffer_bytes(
        batch_size=args.batch_size,
        context_len=args.context_len,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
    )

    total_params_mb = bytes_to_mb(sum(grouped.values()))
    total_kv_mb = bytes_to_mb(kv_bytes)
    total_act_mb = bytes_to_mb(act_bytes)
    total_runtime_mb = total_params_mb + total_kv_mb + total_act_mb

    print("\nKV cache (MB):")
    print(f"  context_len={args.context_len}, batch_size={args.batch_size}")
    print(f"  kv_cache: {total_kv_mb:,.2f} MB")
    print("\nHeuristic activations buffer (MB):")
    print(f"  activations: {total_act_mb:,.2f} MB")

    print("\nTotals:")
    print(f"  params:       {total_params_mb:,.2f} MB")
    print(f"  kv_cache:     {total_kv_mb:,.2f} MB")
    print(f"  activations:  {total_act_mb:,.2f} MB")
    print(f"  approx total: {total_runtime_mb:,.2f} MB")


if __name__ == "__main__":
    main()
