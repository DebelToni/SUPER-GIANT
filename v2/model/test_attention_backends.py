"""
Quick check of dot_product_attention backend selection on GPU vs CPU.
- Constructs random qkv tensors for a variety of sequence lengths.
- Calls jax.nn.dot_product_attention in both full-seq and KV-cache modes.
- Logs which implementation was used (cudnn/xla) and whether any error occurred.

Run on CUDA box:  python test_attention_backends.py
"""
from __future__ import annotations

import itertools
import os
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random


def _run_attention(
    key: jax.Array,
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    use_bias: bool,
    use_cache: bool,
) -> Tuple[str, str]:
    """
    Returns (impl, status) where impl is the backend used,
    status is "ok" or the exception type.
    """
    qkey, kkey, vkey, bkey = random.split(key, 4)
    q = random.normal(qkey, (batch, seqlen, heads, head_dim), dtype=jnp.float16)
    k = random.normal(kkey, (batch, seqlen, heads, head_dim), dtype=jnp.float16)
    v = random.normal(vkey, (batch, seqlen, heads, head_dim), dtype=jnp.float16)

    bias = None
    if use_bias:
        # Causal mask as bias: 0 on diag/lower, -1e10 above.
        mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=jnp.float16))
        bias = (mask - 1.0) * 1e10
        bias = bias.reshape(1, 1, seqlen, seqlen)

    if use_cache:
        # Simulate cache by padding K/V and biasing future positions.
        ctx = seqlen * 2
        k_full = jnp.pad(k, ((0, 0), (0, ctx - seqlen), (0, 0), (0, 0)))
        v_full = jnp.pad(v, ((0, 0), (0, ctx - seqlen), (0, 0), (0, 0)))
        k = k_full
        v = v_full
        seqlen_q = seqlen
        seqlen_k = ctx
    else:
        seqlen_q = seqlen_k = seqlen

    # Pick impl=auto; capture which backend was picked by inspecting the compiled HLO.
    def call():
        return jax.nn.dot_product_attention(q, k, v, bias=bias, is_causal=use_cache)

    try:
        out = call()
        # Probe which backend was used by inspecting the compiled executable.
        hlo = jax.xla_computation(call)().as_hlo_text()
        impl = "cudnn" if "cudnn" in hlo.lower() else "xla"
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        impl = "error"
        status = f"{type(exc).__name__}: {exc}"
    return impl, status


def main():
    is_gpu = any(dev.platform == "gpu" for dev in jax.local_devices())
    print(f"Detected GPU: {is_gpu}")
    configs = []
    # Test a variety of sequence lengths around cudnn constraints.
    for seqlen in (64, 96, 127, 128, 256):
        for use_bias in (False, True):
            for use_cache in (False, True):
                configs.append((seqlen, use_bias, use_cache))

    key = random.PRNGKey(0)
    results = []
    for idx, (seqlen, use_bias, use_cache) in enumerate(configs):
        impl, status = _run_attention(
            random.fold_in(key, idx),
            batch=1,
            seqlen=seqlen,
            heads=8,
            head_dim=64,
            use_bias=use_bias,
            use_cache=use_cache,
        )
        results.append((seqlen, use_bias, use_cache, impl, status))
        print(
            f"L={seqlen:>3} bias={use_bias} cache={use_cache} -> impl={impl} status={status}"
        )

    # Summary
    errors = [r for r in results if r[4] != "ok"]
    print("\nSummary:")
    print(f"  Total configs: {len(results)}")
    print(f"  Errors: {len(errors)}")
    for e in errors:
        print(f"    L={e[0]} bias={e[1]} cache={e[2]} impl={e[3]} status={e[4]}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
