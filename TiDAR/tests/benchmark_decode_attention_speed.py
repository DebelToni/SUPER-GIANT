#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from TiDAR.model.Transformer_block import NativeJaxSelfAttention
from TiDAR.model.tidar_core import (
    build_decode_bias_template_step_prefix,
    build_decode_position_template,
)


def _parse_dtype(name: str) -> jnp.dtype:
    table = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from: {', '.join(table)}")
    return table[name]


def _make_kernel_bench_fns(implementation: Literal["xla", "cudnn"]):
    @jax.jit
    def run_structured(
        q_bank: jnp.ndarray,
        k_full: jnp.ndarray,
        v_full: jnp.ndarray,
        attn_bias: jnp.ndarray,
    ) -> jnp.ndarray:
        n_iters = q_bank.shape[0]

        def body_fn(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            out = jax.nn.dot_product_attention(
                q_bank[i],
                k_full,
                v_full,
                bias=attn_bias,
                is_causal=False,
                implementation=implementation,
            )
            return acc + jnp.sum(out.astype(jnp.float32))

        return jax.lax.fori_loop(0, n_iters, body_fn, jnp.array(0.0, dtype=jnp.float32))

    @jax.jit
    def run_dense(
        q_bank: jnp.ndarray,
        k_full: jnp.ndarray,
        v_full: jnp.ndarray,
    ) -> jnp.ndarray:
        n_iters = q_bank.shape[0]

        def body_fn(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            out = jax.nn.dot_product_attention(
                q_bank[i],
                k_full,
                v_full,
                bias=None,
                is_causal=False,
                implementation=implementation,
            )
            return acc + jnp.sum(out.astype(jnp.float32))

        return jax.lax.fori_loop(0, n_iters, body_fn, jnp.array(0.0, dtype=jnp.float32))

    @jax.jit
    def run_dense_jax_zero_masked(
        q_bank: jnp.ndarray,
        k_full: jnp.ndarray,
        v_full: jnp.ndarray,
        keep_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        n_iters = q_bank.shape[0]
        scale = jnp.asarray(1.0 / (q_bank.shape[-1] ** 0.5), dtype=jnp.float32)
        keep_mask_f = keep_mask.astype(jnp.float32)

        def body_fn(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            q = q_bank[i]
            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_full).astype(jnp.float32) * scale
            probs = jax.nn.softmax(logits, axis=-1)
            probs = probs * keep_mask_f
            denom = jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-9)
            probs = probs / denom
            out = jnp.einsum("bhqk,bkhd->bqhd", probs.astype(q.dtype), v_full)
            return acc + jnp.sum(out.astype(jnp.float32))

        return jax.lax.fori_loop(0, n_iters, body_fn, jnp.array(0.0, dtype=jnp.float32))

    return run_structured, run_dense, run_dense_jax_zero_masked


def _make_pallas_masked_kernel_fn(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
):
    try:
        from jax.experimental.pallas.ops.gpu import attention as pallas_attention
    except Exception:
        return None

    kv_padded = ((kv_len + 15) // 16) * 16
    kv_pad = kv_padded - kv_len
    block_sizes = pallas_attention.BlockSizes(block_q=1, block_k=16)
    sm_scale = float(1.0 / (head_dim ** 0.5))

    @jax.jit
    def run_pallas_masked(
        q_bank: jnp.ndarray,
        k_full: jnp.ndarray,
        v_full: jnp.ndarray,
        keep_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        n_iters = q_bank.shape[0]
        keep_mask_b = keep_mask.astype(jnp.bool_)
        if kv_pad > 0:
            k_pad = jnp.pad(k_full, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
            v_pad = jnp.pad(v_full, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
        else:
            k_pad = k_full
            v_pad = v_full

        def body_fn(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            q_iter = q_bank[i]

            def q_row_body(qi: int, out_acc: jnp.ndarray) -> jnp.ndarray:
                q_row = jax.lax.dynamic_slice(q_iter, (0, qi, 0, 0), (batch_size, 1, num_heads, head_dim))
                mask_row = jax.lax.dynamic_slice(keep_mask_b, (0, 0, qi, 0), (1, 1, 1, kv_len)).squeeze((0, 1, 2))
                if kv_pad > 0:
                    mask_row = jnp.pad(mask_row, ((0, kv_pad),), constant_values=False)
                seg_ids = jnp.where(mask_row[None, :], 0, 1).astype(jnp.int32)
                seg_ids = seg_ids.at[:, 0].set(0)
                out_row = pallas_attention.mha(
                    q_row,
                    k_pad,
                    v_pad,
                    segment_ids=seg_ids,
                    sm_scale=sm_scale,
                    causal=False,
                    block_sizes=block_sizes,
                    backward_pass_impl="triton",
                    num_warps=None,
                    num_stages=2,
                    grid=None,
                    interpret=False,
                    debug=False,
                    return_residuals=False,
                )
                return jax.lax.dynamic_update_slice(out_acc, out_row, (0, qi, 0, 0))

            out0 = jnp.zeros((batch_size, q_len, num_heads, head_dim), dtype=dtype)
            out = jax.lax.fori_loop(0, q_len, q_row_body, out0)
            return acc + jnp.sum(out.astype(jnp.float32))

        return jax.lax.fori_loop(0, n_iters, body_fn, jnp.array(0.0, dtype=jnp.float32))

    return run_pallas_masked


def _make_native_bench_fn(
    module: NativeJaxSelfAttention,
    *,
    prefix_len: int,
    cache_write_len: int,
    kv_cache_len: int,
):
    prefix_len_arr = jnp.asarray(prefix_len, dtype=jnp.int32)

    @jax.jit
    def run_native(
        params: Any,
        cache_vars: Any,
        x_bank: jnp.ndarray,
        pos_ids: jnp.ndarray,
        attn_bias: jnp.ndarray,
    ) -> jnp.ndarray:
        n_iters = x_bank.shape[0]

        def body_fn(i: int, carry: tuple[jnp.ndarray, Any]) -> tuple[jnp.ndarray, Any]:
            acc, cache = carry
            y, mutated = module.apply(
                {"params": params, "cache": cache},
                x_bank[i],
                deterministic=True,
                use_kv_cache=True,
                write_to_cache=False,
                prefix_len=prefix_len_arr,
                cache_write_len=cache_write_len,
                attn_bias=attn_bias,
                position_ids=pos_ids,
                kv_cache_len=kv_cache_len,
                mutable=["cache"],
            )
            return (acc + jnp.sum(y.astype(jnp.float32)), mutated["cache"])

        acc0 = jnp.array(0.0, dtype=jnp.float32)
        acc_final, _ = jax.lax.fori_loop(0, n_iters, body_fn, (acc0, cache_vars))
        return acc_final

    return run_native


def _time_fn(fn, *args, warmup: int, repeats: int) -> tuple[list[float], float]:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        out.block_until_ready()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    return times, float(jax.device_get(out))


def _avg_s(times: list[float]) -> float:
    return sum(times) / len(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TiDAR decode attention at L x (L + prefix), L = K + K^2. "
            "Runs both kernel-level (dot_product_attention) and NativeJaxSelfAttention "
            "structured-vs-dense comparisons, plus a dense+JAX-zero-masked kernel variant."
        )
    )
    parser.add_argument("--draft_len", type=int, default=8, help="K in TiDAR decode.")
    parser.add_argument("--prefix_len", type=int, default=1024, help="Simulated committed KV prefix length.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--rope_dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--iterations", type=int, default=100, help="How many attention calls per benchmark run.")
    parser.add_argument("--repeats", type=int, default=5, help="How many timed repeats to average.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--implementation", type=str, default="cudnn", choices=["xla", "cudnn"])
    parser.add_argument("--bias_value", type=float, default=-1.0e10)
    parser.add_argument(
        "--enable_pallas_masked",
        action="store_true",
        help="Enable optional pallas masked attention kernel benchmark term.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.draft_len <= 1:
        raise ValueError("--draft_len must be > 1")
    if args.prefix_len < 0:
        raise ValueError("--prefix_len must be >= 0")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.num_heads <= 0:
        raise ValueError("--num_heads must be > 0")
    if args.num_kv_heads <= 0:
        raise ValueError("--num_kv_heads must be > 0")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num_heads must be divisible by --num_kv_heads")
    if args.rope_dim <= 0 or args.rope_dim > args.head_dim or args.rope_dim % 2 != 0:
        raise ValueError("--rope_dim must be even and in [2, head_dim]")

    backend = jax.default_backend()
    if args.implementation == "cudnn" and backend != "gpu":
        raise ValueError(
            "--implementation cudnn requires GPU backend. "
            "On CPU, run with --implementation xla."
        )

    impl: Literal["xla", "cudnn"] = "cudnn" if args.implementation == "cudnn" else "xla"
    dtype = _parse_dtype(args.dtype)
    k = int(args.draft_len)
    l = k + (k * k)
    key_len = l + int(args.prefix_len)
    d_model = args.num_heads * args.head_dim

    q_shape = (args.iterations, args.batch_size, l, args.num_heads, args.head_dim)
    kv_shape = (args.batch_size, key_len, args.num_heads, args.head_dim)
    x_shape = (args.iterations, args.batch_size, l, d_model)

    key = jax.random.PRNGKey(args.seed)
    key, q_key, k_key, v_key, x_key, init_key, cache_key = jax.random.split(key, 7)

    q_bank = jax.random.normal(q_key, q_shape, dtype=dtype)
    k_full = jax.random.normal(k_key, kv_shape, dtype=dtype)
    v_full = jax.random.normal(v_key, kv_shape, dtype=dtype)
    x_bank = jax.random.normal(x_key, x_shape, dtype=dtype)

    structured_bias = build_decode_bias_template_step_prefix(
        cache_len=int(args.prefix_len),
        draft_len=k,
        bias_value=float(args.bias_value),
    ).astype(dtype)
    keep_mask = structured_bias >= jnp.asarray(0.0, dtype=structured_bias.dtype)
    dense_bias = jnp.zeros_like(structured_bias)

    decode_pos = (int(args.prefix_len) + build_decode_position_template(k)).astype(jnp.int32)
    pos_ids = jnp.broadcast_to(decode_pos[None, :], (args.batch_size, l))

    run_structured_kernel, run_dense_kernel, run_dense_zero_masked_kernel = _make_kernel_bench_fns(impl)
    run_pallas_masked_kernel = None
    if args.enable_pallas_masked:
        run_pallas_masked_kernel = _make_pallas_masked_kernel_fn(
            batch_size=args.batch_size,
            q_len=l,
            kv_len=key_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dtype=dtype,
        )

    native_context_len = int(args.prefix_len)

    module = NativeJaxSelfAttention(
        num_heads=args.num_heads,
        qkv_features=d_model,
        context_length=native_context_len,
        dropout_rate=0.0,
        num_kv=args.num_kv_heads,
        dtype=dtype,
        param_dtype=jnp.float32,
        rotary_dim=args.rope_dim,
        draft_len=k,
    )

    dummy_x = jnp.zeros((args.batch_size, l, d_model), dtype=dtype)
    init_vars = module.init(
        {"params": init_key},
        dummy_x,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
        position_ids=pos_ids,
        kv_cache_len=native_context_len,
    )
    native_params = init_vars["params"]
    native_cache = init_vars["cache"]

    cache_k_key, cache_v_key = jax.random.split(cache_key)
    cache_k = jnp.asarray(native_cache["k"])
    cache_v = jnp.asarray(native_cache["v"])
    native_cache = {
        "k": jax.random.normal(cache_k_key, cache_k.shape, dtype=dtype),
        "v": jax.random.normal(cache_v_key, cache_v.shape, dtype=dtype),
    }

    run_native = _make_native_bench_fn(
        module,
        prefix_len=int(args.prefix_len),
        cache_write_len=0,
        kv_cache_len=native_context_len,
    )

    kernel_struct_times, kernel_struct_sum = _time_fn(
        run_structured_kernel,
        q_bank,
        k_full,
        v_full,
        structured_bias,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    kernel_dense_times, kernel_dense_sum = _time_fn(
        run_dense_kernel,
        q_bank,
        k_full,
        v_full,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    kernel_dense_zero_times, kernel_dense_zero_sum = _time_fn(
        run_dense_zero_masked_kernel,
        q_bank,
        k_full,
        v_full,
        keep_mask,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    pallas_masked_times = None
    pallas_masked_sum = None
    pallas_masked_error: str | None = None
    if run_pallas_masked_kernel is not None:
        try:
            pallas_masked_times, pallas_masked_sum = _time_fn(
                run_pallas_masked_kernel,
                q_bank,
                k_full,
                v_full,
                keep_mask,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except Exception as exc:
            pallas_masked_error = f"{type(exc).__name__}: {exc}"

    native_struct_times, native_struct_sum = _time_fn(
        run_native,
        native_params,
        native_cache,
        x_bank,
        pos_ids,
        structured_bias,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    native_dense_times, native_dense_sum = _time_fn(
        run_native,
        native_params,
        native_cache,
        x_bank,
        pos_ids,
        dense_bias,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    kernel_struct_avg = _avg_s(kernel_struct_times)
    kernel_dense_avg = _avg_s(kernel_dense_times)
    kernel_dense_zero_avg = _avg_s(kernel_dense_zero_times)
    native_struct_avg = _avg_s(native_struct_times)
    native_dense_avg = _avg_s(native_dense_times)

    kernel_struct_ms = (kernel_struct_avg / args.iterations) * 1000.0
    kernel_dense_ms = (kernel_dense_avg / args.iterations) * 1000.0
    kernel_dense_zero_ms = (kernel_dense_zero_avg / args.iterations) * 1000.0
    native_struct_ms = (native_struct_avg / args.iterations) * 1000.0
    native_dense_ms = (native_dense_avg / args.iterations) * 1000.0

    kernel_ratio = kernel_dense_avg / kernel_struct_avg if kernel_struct_avg > 0 else float("inf")
    kernel_dense_zero_ratio = kernel_dense_zero_avg / kernel_struct_avg if kernel_struct_avg > 0 else float("inf")
    kernel_dense_zero_vs_dense = kernel_dense_zero_avg / kernel_dense_avg if kernel_dense_avg > 0 else float("inf")
    pallas_masked_avg = _avg_s(pallas_masked_times) if pallas_masked_times is not None else None
    pallas_masked_ms = ((pallas_masked_avg / args.iterations) * 1000.0) if pallas_masked_avg is not None else None
    pallas_masked_over_struct = (pallas_masked_avg / kernel_struct_avg) if pallas_masked_avg is not None and kernel_struct_avg > 0 else None
    native_ratio = native_dense_avg / native_struct_avg if native_struct_avg > 0 else float("inf")
    native_vs_kernel_struct = native_struct_avg / kernel_struct_avg if kernel_struct_avg > 0 else float("inf")
    native_vs_kernel_dense = native_dense_avg / kernel_dense_avg if kernel_dense_avg > 0 else float("inf")

    print("=" * 90)
    print("TiDAR Decode Attention Benchmark")
    print("=" * 90)
    print(f"backend:                        {backend}")
    print(f"kernel implementation:          {impl}")
    print(f"dtype:                          {args.dtype}")
    print(f"batch_size:                     {args.batch_size}")
    print(f"num_heads / num_kv_heads:       {args.num_heads} / {args.num_kv_heads}")
    print(f"head_dim / rope_dim:            {args.head_dim} / {args.rope_dim}")
    print(f"draft_len (K):                  {k}")
    print(f"decode_q_len (L):               {l}  (L = K + K^2)")
    print(f"prefix_len:                     {args.prefix_len}")
    print(f"key_len (L + prefix):           {key_len}")
    print(f"iterations per run:             {args.iterations}")
    print(f"timed repeats:                  {args.repeats}")
    print("-" * 90)
    print("Kernel-level (jax.nn.dot_product_attention)")
    print(f"  structured avg total s:       {kernel_struct_avg:.6f}")
    print(f"  structured ms / attn:         {kernel_struct_ms:.6f}")
    print(f"  dense avg total s:            {kernel_dense_avg:.6f}")
    print(f"  dense ms / attn:              {kernel_dense_ms:.6f}")
    print(f"  dense/structured ratio:       {kernel_ratio:.4f}")
    print(f"  dense+jax-zero avg total s:   {kernel_dense_zero_avg:.6f}")
    print(f"  dense+jax-zero ms / attn:     {kernel_dense_zero_ms:.6f}")
    print(f"  dense+jax-zero/structured:    {kernel_dense_zero_ratio:.4f}")
    print(f"  dense+jax-zero/dense:         {kernel_dense_zero_vs_dense:.4f}")
    if pallas_masked_avg is not None and pallas_masked_ms is not None and pallas_masked_over_struct is not None:
        print(f"  pallas masked avg total s:    {pallas_masked_avg:.6f}")
        print(f"  pallas masked ms / attn:      {pallas_masked_ms:.6f}")
        print(f"  pallas masked/structured:     {pallas_masked_over_struct:.4f}")
    else:
        if args.enable_pallas_masked:
            if pallas_masked_error is None:
                print("  pallas masked:                unavailable for this JAX/backend")
            else:
                print(f"  pallas masked:                failed ({pallas_masked_error})")
        else:
            print("  pallas masked:                skipped (enable via --enable_pallas_masked)")
    print("-" * 90)
    print("NativeJaxSelfAttention decode path (with KV cache + projections + RoPE)")
    print(f"  native structured avg total s:{native_struct_avg:.6f}")
    print(f"  native structured ms / attn:  {native_struct_ms:.6f}")
    print(f"  native dense avg total s:     {native_dense_avg:.6f}")
    print(f"  native dense ms / attn:       {native_dense_ms:.6f}")
    print(f"  native dense/structured ratio:{native_ratio:.4f}")
    print("-" * 90)
    print("Cross-comparison")
    print(f"  native_struct / kernel_struct:{native_vs_kernel_struct:.4f}")
    print(f"  native_dense / kernel_dense:  {native_vs_kernel_dense:.4f}")
    print("-" * 90)
    print("Checksums")
    print(f"  kernel structured checksum:   {kernel_struct_sum:.6f}")
    print(f"  kernel dense checksum:        {kernel_dense_sum:.6f}")
    print(f"  kernel dense+jax-zero sum:    {kernel_dense_zero_sum:.6f}")
    if pallas_masked_sum is not None:
        print(f"  kernel pallas-masked sum:     {pallas_masked_sum:.6f}")
    print(f"  native structured checksum:   {native_struct_sum:.6f}")
    print(f"  native dense checksum:        {native_dense_sum:.6f}")
    print("=" * 90)
    print("Note: native dense uses zero attn_bias in decode path (no structured mask).")


if __name__ == "__main__":
    main()
