#!/usr/bin/env python3
"""Mock Anchor-TiDAR acceptance/iter speedup benchmark.

This benchmark uses the real TiDAR model forward paths and KV-cache code, but
removes model-quality dependence by forcing a deterministic acceptance schedule.
It needs no dataset, tokenizer, or checkpoint.

Measured paths:
- AR baseline: one KV-cache decode token per model forward.
- Anchor-TiDAR: one structured decode forward over K + K^2 query positions,
  with optimistic KV writes and pointer commit; acceptance is forced to a target
  average such as 2.0 or 1.5 tokens/iteration.

For target_accept=1.5 the deterministic schedule alternates 1,2,1,2,..., so
half of the iterations are the Anchor worst case (+1 token only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TiDAR.model.GiantTiDAR import TiDAR
from TiDAR.model.tidar_core import (
    build_decode_bias_template,
    build_decode_position_template,
    prefill_prompt_with_draft,
    sample_tokens,
)


@dataclass(frozen=True)
class ModelProfile:
    vocab_size: int
    d_model: int
    n_heads: int
    num_kv_heads: int
    rope_dim: int
    d_ff: int
    n_layers: int
    context_length: int
    draft_len: int


PROFILES: dict[str, ModelProfile] = {
    "tiny": ModelProfile(
        vocab_size=4096,
        d_model=256,
        n_heads=4,
        num_kv_heads=4,
        rope_dim=64,
        d_ff=768,
        n_layers=4,
        context_length=1024,
        draft_len=8,
    ),
    "small": ModelProfile(
        vocab_size=8192,
        d_model=512,
        n_heads=8,
        num_kv_heads=4,
        rope_dim=64,
        d_ff=1536,
        n_layers=12,
        context_length=1536,
        draft_len=8,
    ),
    "smollm135": ModelProfile(
        vocab_size=49152,
        d_model=576,
        n_heads=9,
        num_kv_heads=3,
        rope_dim=64,
        d_ff=1536,
        n_layers=30,
        context_length=2048,
        draft_len=8,
    ),
    "smollm360": ModelProfile(
        vocab_size=49152,
        d_model=960,
        n_heads=15,
        num_kv_heads=5,
        rope_dim=64,
        d_ff=2560,
        n_layers=32,
        context_length=4096,
        draft_len=8,
    ),
}


@dataclass
class RunMetrics:
    label: str
    generated_tokens: int
    iterations: int
    mean_s: float
    std_s: float
    tokens_per_s: float
    ms_per_token: float | None
    ms_per_iter: float | None
    avg_accept_per_iter: float
    max_accept_per_iter: int


def _parse_targets(value: str) -> list[float]:
    out: list[float] = []
    for part in value.replace(",", " ").split():
        if part.strip():
            out.append(float(part))
    if not out:
        raise ValueError("At least one target accept/iter value is required")
    return out


def _tree_block_until_ready(x: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a,
        x,
    )


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _build_model(profile: ModelProfile, *, param_dtype: str, compute_dtype: str) -> TiDAR:
    return TiDAR(
        vocab_size=profile.vocab_size,
        context_length=profile.context_length,
        d_model=profile.d_model,
        n_heads=profile.n_heads,
        num_kv_heads=profile.num_kv_heads,
        rope_dim=profile.rope_dim,
        d_ff=profile.d_ff,
        n_layers=profile.n_layers,
        dropout_rate=0.0,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        use_remat=False,
        draft_len=profile.draft_len,
    )


def _init_params_and_cache(
    model: TiDAR,
    *,
    key: jax.Array,
    pad_token_id: int,
    kv_cache_len: int,
):
    dummy = jnp.full((1, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": key},
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
        position_ids=jnp.zeros((1, 1), dtype=jnp.int32),
        kv_cache_len=kv_cache_len,
    )
    return variables["params"], variables["cache"]


def make_ar_generate_fn(
    model: TiDAR,
    *,
    cache_len: int,
    temperature: float,
    top_k: int,
):
    @jax.jit
    def generate_ar(
        params,
        cache_vars,
        out_ids: jnp.ndarray,
        prefix_len: jnp.ndarray,
        max_steps: jnp.ndarray,
        prev_logit: jnp.ndarray,
        rng_key: jax.Array,
    ):
        prefix_len = prefix_len.astype(jnp.int32)
        generated = jnp.asarray(0, dtype=jnp.int32)

        def cond_fn(state):
            _, _, _, _, generated, _, _ = state
            return generated < max_steps

        def body_fn(state):
            rng, cache, out, prefix_len, generated, prev_logit, max_token = state
            rng, token = sample_tokens(rng, prev_logit, temperature, top_k)
            token = token.astype(jnp.int32)
            out = jax.lax.dynamic_update_slice(out, token.reshape((1,)), (prefix_len,))

            logits, mutated = model.apply(
                {"params": params, "cache": cache},
                token.reshape((1, 1)),
                deterministic=True,
                use_kv_cache=True,
                write_to_cache=True,
                cur_index=prefix_len,
                position_ids=prefix_len.reshape((1, 1)),
                kv_cache_len=cache_len,
                mutable=["cache"],
            )
            prefix_len2 = prefix_len + 1
            generated2 = generated + 1
            max_token2 = jnp.maximum(max_token, token)
            return (rng, mutated["cache"], out, prefix_len2, generated2, logits[0, -1], max_token2)

        state0 = (
            rng_key,
            cache_vars,
            out_ids,
            prefix_len,
            generated,
            prev_logit,
            jnp.asarray(0, dtype=jnp.int32),
        )
        _, _, out, prefix_len, generated, _, max_token = jax.lax.while_loop(cond_fn, body_fn, state0)
        stats = {
            "total_accepts": generated,
            "n_iterations": generated,
            "avg_accept_per_iter": jnp.asarray(1.0, dtype=jnp.float32),
            "max_accept_per_iter": jnp.asarray(1, dtype=jnp.int32),
            "max_token": max_token,
        }
        return out, prefix_len, generated, stats

    return generate_ar


def make_tidar_forced_generate_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    pad_token_id: int,
    temperature: float,
    top_k: int,
    bias_value: float,
):
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))

    @jax.jit
    def generate_tidar(
        params,
        cache_vars,
        out_ids: jnp.ndarray,
        prefix_len: jnp.ndarray,
        max_steps: jnp.ndarray,
        prev_logit: jnp.ndarray,
        rng_key: jax.Array,
        initial_draft_logits: jnp.ndarray,
        target_accept_per_iter: jnp.ndarray,
    ):
        prefix_len = prefix_len.astype(jnp.int32)
        generated = jnp.asarray(0, dtype=jnp.int32)
        total_accepts = jnp.asarray(0, dtype=jnp.int32)
        n_iterations = jnp.asarray(0, dtype=jnp.int32)
        max_accepts = jnp.asarray(0, dtype=jnp.int32)

        target_accept_per_iter = jnp.clip(
            target_accept_per_iter.astype(jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(float(draft_len), dtype=jnp.float32),
        )

        rng_key, anchor = sample_tokens(rng_key, prev_logit, temperature, top_k)
        rng_key, init_draft = sample_tokens(rng_key, initial_draft_logits, temperature, top_k)
        current_draft = init_draft.astype(jnp.int32).at[0].set(anchor.astype(jnp.int32))
        current_draft_logits = initial_draft_logits

        def cond_fn(state):
            _, _, _, _, _, generated, _, _, _, _ = state
            return generated < max_steps

        def body_fn(state):
            (
                rng,
                cache,
                out,
                prefix_len,
                current_draft,
                generated,
                current_draft_logits,
                total_accepts,
                n_iters,
                max_acc,
            ) = state

            step_tokens = jnp.concatenate([current_draft, predraft_masks])
            step_pos_ids = (prefix_len + position_template).astype(jnp.int32)
            logits, mutated = model.apply(
                {"params": params, "cache": cache},
                step_tokens[None, :],
                deterministic=True,
                use_kv_cache=True,
                write_to_cache=False,
                prefix_len=prefix_len,
                cache_write_len=draft_len,
                attn_bias=decode_bias,
                position_ids=step_pos_ids[None, :],
                kv_cache_len=cache_len,
                mutable=["cache"],
            )
            cache = mutated["cache"]
            logits = logits[0]
            verify_logits = logits[:draft_len]
            predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)

            # Deterministic schedule whose cumulative accepted tokens track
            # floor(n_iters * target_accept_per_iter). For target=1.5 this is
            # 1,2,1,2,...; for target=2.0 this is 2,2,2,...
            desired_before = jnp.floor(n_iters.astype(jnp.float32) * target_accept_per_iter).astype(jnp.int32)
            desired_after = jnp.floor((n_iters.astype(jnp.float32) + 1.0) * target_accept_per_iter).astype(jnp.int32)
            forced_accept = jnp.clip(desired_after - desired_before, 1, draft_len)

            remaining = max_steps - generated
            eff_accept = jnp.minimum(forced_accept, remaining)

            proposal_idx = jnp.clip(eff_accept - 1, 0, draft_len - 1).astype(jnp.int32)
            next_anchor_logits = verify_logits[proposal_idx]
            rng, next_anchor = sample_tokens(rng, next_anchor_logits, temperature, top_k)
            next_anchor = next_anchor.astype(jnp.int32)

            proposal_logits = predraft_logits[proposal_idx]
            rng, next_draft = sample_tokens(rng, proposal_logits, temperature, top_k)
            next_draft = next_draft.astype(jnp.int32).at[0].set(next_anchor)

            commit_padded = jnp.where(idx_k < eff_accept, current_draft, pad_token_id).astype(jnp.int32)
            out = jax.lax.dynamic_update_slice(out, commit_padded, (prefix_len,))

            prefix_len2 = prefix_len + eff_accept
            generated2 = generated + eff_accept
            total_accepts2 = total_accepts + eff_accept
            n_iters2 = n_iters + 1
            max_acc2 = jnp.maximum(max_acc, eff_accept)
            return (
                rng,
                cache,
                out,
                prefix_len2,
                next_draft,
                generated2,
                proposal_logits,
                total_accepts2,
                n_iters2,
                max_acc2,
            )

        state0 = (
            rng_key,
            cache_vars,
            out_ids,
            prefix_len,
            current_draft,
            generated,
            current_draft_logits,
            total_accepts,
            n_iterations,
            max_accepts,
        )
        (
            _,
            _,
            out,
            prefix_len,
            _,
            generated,
            _,
            total_accepts,
            n_iterations,
            max_accepts,
        ) = jax.lax.while_loop(cond_fn, body_fn, state0)
        stats = {
            "total_accepts": total_accepts,
            "n_iterations": n_iterations,
            "avg_accept_per_iter": total_accepts.astype(jnp.float32) / jnp.maximum(n_iterations, 1),
            "max_accept_per_iter": max_accepts,
        }
        return out, prefix_len, generated, stats

    return generate_tidar


def make_no_anchor_zero_recovery_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    temperature: float,
    top_k: int,
    bias_value: float,
):
    """No-anchor TiDAR worst-case mock: 0 accepts, then a second structured forward.

    Each loop commits 1 token but charges 2 full TiDAR structured decode forwards.
    This approximates the paper-style no-progress edge case that Anchor-TiDAR avoids.
    """
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))

    def structured_forward(params, cache, current_draft, prefix_len):
        step_tokens = jnp.concatenate([current_draft, predraft_masks])
        step_pos_ids = (prefix_len + position_template).astype(jnp.int32)
        logits, mutated = model.apply(
            {"params": params, "cache": cache},
            step_tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            cache_write_len=draft_len,
            attn_bias=decode_bias,
            position_ids=step_pos_ids[None, :],
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]

    @jax.jit
    def generate_no_anchor_zero(
        params,
        cache_vars,
        out_ids: jnp.ndarray,
        prefix_len: jnp.ndarray,
        max_steps: jnp.ndarray,
        rng_key: jax.Array,
        initial_draft_logits: jnp.ndarray,
    ):
        prefix_len = prefix_len.astype(jnp.int32)
        generated = jnp.asarray(0, dtype=jnp.int32)
        n_iterations = jnp.asarray(0, dtype=jnp.int32)

        rng_key, current_draft = sample_tokens(rng_key, initial_draft_logits, temperature, top_k)
        current_draft = current_draft.astype(jnp.int32)

        def cond_fn(state):
            _, _, _, _, generated, _, _ = state
            return generated < max_steps

        def body_fn(state):
            rng, cache, out, prefix_len, generated, current_draft, n_iters = state

            # First no-anchor attempt: full rejection, no committed tokens.
            _, cache = structured_forward(params, cache, current_draft, prefix_len)

            # Recovery attempt: charge another full structured forward and commit 1 token.
            logits, cache = structured_forward(params, cache, current_draft, prefix_len)
            verify_logits = logits[:draft_len]
            predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)

            rng, token = sample_tokens(rng, verify_logits[0], temperature, top_k)
            token = token.astype(jnp.int32)
            out = jax.lax.dynamic_update_slice(out, token.reshape((1,)), (prefix_len,))

            proposal_logits = predraft_logits[0]
            rng, next_draft = sample_tokens(rng, proposal_logits, temperature, top_k)
            next_draft = next_draft.astype(jnp.int32)

            return (
                rng,
                cache,
                out,
                prefix_len + 1,
                generated + 1,
                next_draft,
                n_iters + 1,
            )

        state0 = (rng_key, cache_vars, out_ids, prefix_len, generated, current_draft, n_iterations)
        _, _, out, prefix_len, generated, _, n_iterations = jax.lax.while_loop(cond_fn, body_fn, state0)
        stats = {
            "total_accepts": jnp.asarray(0, dtype=jnp.int32),
            "n_iterations": n_iterations,
            "avg_accept_per_iter": jnp.asarray(0.0, dtype=jnp.float32),
            "max_accept_per_iter": jnp.asarray(0, dtype=jnp.int32),
        }
        return out, prefix_len, generated, stats

    return generate_no_anchor_zero


def _time_call(fn, args: tuple[Any, ...], *, warmup: int, repeats: int) -> tuple[list[float], Any]:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        _tree_block_until_ready(out)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        _tree_block_until_ready(out)
        times.append(time.perf_counter() - t0)
    assert out is not None
    return times, out


def _metrics_from_output(label: str, times: list[float], out: Any, *, per_iter_is_token: bool) -> RunMetrics:
    _, _, generated, stats = out
    generated_i = int(np.asarray(generated))
    iterations_i = int(np.asarray(stats["n_iterations"]))
    avg_accept = float(np.asarray(stats["avg_accept_per_iter"]))
    max_accept = int(np.asarray(stats["max_accept_per_iter"]))
    mean_s, std_s = _mean_std(times)
    return RunMetrics(
        label=label,
        generated_tokens=generated_i,
        iterations=iterations_i,
        mean_s=mean_s,
        std_s=std_s,
        tokens_per_s=generated_i / mean_s if mean_s > 0 else float("inf"),
        ms_per_token=(mean_s / generated_i) * 1000.0 if generated_i > 0 else None,
        ms_per_iter=(mean_s / iterations_i) * 1000.0 if iterations_i > 0 else None,
        avg_accept_per_iter=avg_accept,
        max_accept_per_iter=max_accept,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Mock Anchor-TiDAR accept/iter speedup benchmark")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smollm135")
    parser.add_argument("--steps", type=int, default=300, help="Generated tokens per timed run.")
    parser.add_argument("--prefix_len", type=int, default=1024)
    parser.add_argument("--targets", type=str, default="2,1.5", help="Comma/space separated target accept/iter values.")
    parser.add_argument("--draft_len", type=int, default=None, help="Override profile K.")
    parser.add_argument("--context_length", type=int, default=None, help="Override profile context length.")
    parser.add_argument("--vocab_size", type=int, default=None, help="Override profile vocab size.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--param_dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--compute_dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--bias_value", type=float, default=-1.0e10)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.prefix_len <= 0:
        raise ValueError("--prefix_len must be > 0")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("Need --warmup >= 0 and --repeats > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_k < 0:
        raise ValueError("--top_k must be >= 0")

    base_profile = PROFILES[args.profile]
    profile = ModelProfile(
        vocab_size=args.vocab_size or base_profile.vocab_size,
        d_model=base_profile.d_model,
        n_heads=base_profile.n_heads,
        num_kv_heads=base_profile.num_kv_heads,
        rope_dim=base_profile.rope_dim,
        d_ff=base_profile.d_ff,
        n_layers=base_profile.n_layers,
        context_length=args.context_length or base_profile.context_length,
        draft_len=args.draft_len or base_profile.draft_len,
    )
    targets = _parse_targets(args.targets)
    for target in targets:
        if not (1.0 <= target <= float(profile.draft_len)):
            raise ValueError(f"Target accept/iter {target} must be in [1, K={profile.draft_len}]")

    required_len = args.prefix_len + args.steps + profile.draft_len + 1
    if required_len > profile.context_length:
        raise ValueError(
            f"Required length {required_len} exceeds context_length {profile.context_length}. "
            "Increase --context_length or reduce --prefix_len/--steps."
        )

    print("=" * 88)
    print("Mock Anchor-TiDAR acceptance/iter speedup benchmark")
    print("=" * 88)
    print(f"backend:             {jax.default_backend()}")
    print(f"devices:             {[str(d) for d in jax.devices()]}")
    print(f"profile:             {args.profile}")
    print(f"profile_params:      {asdict(profile)}")
    print(f"prefix_len:          {args.prefix_len}")
    print(f"steps:               {args.steps}")
    print(f"targets:             {targets}")
    print(f"temperature/top_k:   {args.temperature} / {args.top_k}")
    print(f"warmup/repeats:      {args.warmup} / {args.repeats}")
    print("-" * 88)

    rng = jax.random.PRNGKey(args.seed)
    model = _build_model(profile, param_dtype=args.param_dtype, compute_dtype=args.compute_dtype)
    pad_token_id = 0
    mask_id = 1

    rng, init_key = jax.random.split(rng)
    params, empty_cache = _init_params_and_cache(
        model,
        key=init_key,
        pad_token_id=pad_token_id,
        kv_cache_len=required_len,
    )
    params = jax.device_put(params)
    empty_cache = jax.device_put(empty_cache)

    prompt_ids = jnp.full((args.prefix_len,), 2, dtype=jnp.int32)
    cache_vars, prefix_len, prev_logit, initial_draft_logits = prefill_prompt_with_draft(
        model,
        params,
        empty_cache,
        prompt_ids,
        draft_len=profile.draft_len,
        mask_id=mask_id,
        kv_cache_len=required_len,
        bias_value=args.bias_value,
    )
    _tree_block_until_ready((cache_vars, prev_logit, initial_draft_logits))

    out_host = np.full((required_len,), pad_token_id, dtype=np.int32)
    out_host[: args.prefix_len] = 2
    out_ids = jax.device_put(jnp.asarray(out_host))

    ar_fn = make_ar_generate_fn(
        model,
        cache_len=required_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    tidar_fn = make_tidar_forced_generate_fn(
        model,
        cache_len=required_len,
        draft_len=profile.draft_len,
        mask_id=mask_id,
        pad_token_id=pad_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        bias_value=args.bias_value,
    )
    no_anchor_zero_fn = make_no_anchor_zero_recovery_fn(
        model,
        cache_len=required_len,
        draft_len=profile.draft_len,
        mask_id=mask_id,
        temperature=args.temperature,
        top_k=args.top_k,
        bias_value=args.bias_value,
    )

    ar_call_args = (
        params,
        cache_vars,
        out_ids,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(args.steps, dtype=jnp.int32),
        prev_logit,
        jax.random.PRNGKey(args.seed + 1000),
    )
    print("Compiling/running AR baseline...")
    ar_times, ar_out = _time_call(ar_fn, ar_call_args, warmup=args.warmup, repeats=args.repeats)
    ar_metrics = _metrics_from_output("AR", ar_times, ar_out, per_iter_is_token=True)

    print(
        f"AR: mean={ar_metrics.mean_s:.6f}s std={ar_metrics.std_s:.6f}s "
        f"TPS={ar_metrics.tokens_per_s:.3f} ms/token={ar_metrics.ms_per_token:.4f}"
    )

    no_anchor_call_args = (
        params,
        cache_vars,
        out_ids,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(args.steps, dtype=jnp.int32),
        jax.random.PRNGKey(args.seed + 1500),
        initial_draft_logits,
    )
    print("Compiling/running no-anchor TiDAR worst-case 0-accept recovery...")
    no_anchor_times, no_anchor_out = _time_call(
        no_anchor_zero_fn,
        no_anchor_call_args,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    no_anchor_metrics = _metrics_from_output(
        "NoAnchor@0x2",
        no_anchor_times,
        no_anchor_out,
        per_iter_is_token=False,
    )
    assert ar_metrics.ms_per_token is not None
    assert no_anchor_metrics.ms_per_iter is not None
    no_anchor_break_even = no_anchor_metrics.ms_per_iter / ar_metrics.ms_per_token
    no_anchor_speedup = no_anchor_metrics.tokens_per_s / ar_metrics.tokens_per_s
    print(
        f"NoAnchor@0x2: mean={no_anchor_metrics.mean_s:.6f}s std={no_anchor_metrics.std_s:.6f}s "
        f"TPS={no_anchor_metrics.tokens_per_s:.3f} iters={no_anchor_metrics.iterations} "
        f"ms/2-forward-cycle={no_anchor_metrics.ms_per_iter:.4f} "
        f"break_even_accept={no_anchor_break_even:.4f} speedup={no_anchor_speedup:.4f}x"
    )

    tidar_metrics: list[RunMetrics] = []
    for i, target in enumerate(targets):
        tidar_call_args = (
            params,
            cache_vars,
            out_ids,
            jnp.asarray(prefix_len, dtype=jnp.int32),
            jnp.asarray(args.steps, dtype=jnp.int32),
            prev_logit,
            jax.random.PRNGKey(args.seed + 2000 + i),
            initial_draft_logits,
            jnp.asarray(target, dtype=jnp.float32),
        )
        print(f"Compiling/running Anchor-TiDAR forced target_accept/iter={target}...")
        tidar_times, tidar_out = _time_call(tidar_fn, tidar_call_args, warmup=args.warmup, repeats=args.repeats)
        metrics = _metrics_from_output(f"TiDAR@{target:g}", tidar_times, tidar_out, per_iter_is_token=False)
        tidar_metrics.append(metrics)

        assert ar_metrics.ms_per_token is not None
        assert metrics.ms_per_iter is not None
        break_even_accept = metrics.ms_per_iter / ar_metrics.ms_per_token
        speedup = metrics.tokens_per_s / ar_metrics.tokens_per_s
        print(
            f"TiDAR target={target:g}: mean={metrics.mean_s:.6f}s std={metrics.std_s:.6f}s "
            f"TPS={metrics.tokens_per_s:.3f} avg_accept={metrics.avg_accept_per_iter:.4f} "
            f"iters={metrics.iterations} ms/iter={metrics.ms_per_iter:.4f} "
            f"break_even_accept={break_even_accept:.4f} speedup={speedup:.4f}x"
        )

    print("-" * 88)
    print("Summary")
    print(
        "label                 gen  iters  avg_acc  mean_s     TPS       "
        "ms/token  ms/iter   speedup  break_even_acc"
    )
    print(
        f"{ar_metrics.label:<20} {ar_metrics.generated_tokens:4d} {ar_metrics.iterations:6d} "
        f"{ar_metrics.avg_accept_per_iter:7.3f} {ar_metrics.mean_s:8.5f} "
        f"{ar_metrics.tokens_per_s:9.2f} {ar_metrics.ms_per_token or 0:8.4f} "
        f"{ar_metrics.ms_per_iter or 0:8.4f} {1.0:7.3f} {1.0:15.3f}"
    )
    rows = [
        {
            **asdict(no_anchor_metrics),
            "speedup_vs_ar": no_anchor_speedup,
            "break_even_accept_per_iter": no_anchor_break_even,
        }
    ]
    print(
        f"{no_anchor_metrics.label:<20} {no_anchor_metrics.generated_tokens:4d} {no_anchor_metrics.iterations:6d} "
        f"{no_anchor_metrics.avg_accept_per_iter:7.3f} {no_anchor_metrics.mean_s:8.5f} "
        f"{no_anchor_metrics.tokens_per_s:9.2f} {no_anchor_metrics.ms_per_token or 0:8.4f} "
        f"{no_anchor_metrics.ms_per_iter or 0:8.4f} {no_anchor_speedup:7.3f} {no_anchor_break_even:15.3f}"
    )
    for metrics in tidar_metrics:
        assert ar_metrics.ms_per_token is not None
        assert metrics.ms_per_iter is not None
        speedup = metrics.tokens_per_s / ar_metrics.tokens_per_s
        break_even_accept = metrics.ms_per_iter / ar_metrics.ms_per_token
        rows.append(
            {
                **asdict(metrics),
                "speedup_vs_ar": speedup,
                "break_even_accept_per_iter": break_even_accept,
            }
        )
        print(
            f"{metrics.label:<20} {metrics.generated_tokens:4d} {metrics.iterations:6d} "
            f"{metrics.avg_accept_per_iter:7.3f} {metrics.mean_s:8.5f} "
            f"{metrics.tokens_per_s:9.2f} {metrics.ms_per_token or 0:8.4f} "
            f"{metrics.ms_per_iter or 0:8.4f} {speedup:7.3f} {break_even_accept:15.3f}"
        )

    result = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "profile_name": args.profile,
        "profile": asdict(profile),
        "prefix_len": args.prefix_len,
        "steps": args.steps,
        "targets": targets,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ar": asdict(ar_metrics),
        "tidar": rows,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote_json: {out_path}")


if __name__ == "__main__":
    main()
