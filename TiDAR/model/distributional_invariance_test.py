"""Distributional invariance test for Anchor-TiDAR.

This script checks that non-greedy sampling from Anchor-TiDAR matches pure
autoregressive (AR) sampling for the same prompt/temperature/top-k settings.
It draws multiple samples for AR and each draft length, builds histograms for
the first generated positions, and reports L1/KL distances versus the AR
baseline.

Example (N=2000, once a trained TiDAR checkpoint exists):

  /opt/venv/bin/python TiDAR/model/distributional_invariance_test.py \
    --checkpoint /proj/giant-data/TiDAR/smol/smollm-135m.npz \
    --prompt "Hello" \
    --temperature 0.7 \
    --top_k 50 \
    --draft_lens 2,8,20 \
    --num_samples 2000 \
    --num_tokens 1 \
    --output_json /proj/giant-data/TiDAR/distributional_invariance/hello_t0p7_k50_n2000.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TiDAR.model.inference import (
    build_model,
    load_configs,
    load_params,
    load_tokenizer,
    make_anchor_tidar_generate_fn,
    resolve_checkpoint_path,
    tokenize_prompt,
)
from TiDAR.model.tidar_core import (
    init_kv_cache,
    prefill_prompt,
    sample_tokens,
)
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params


def parse_draft_lens(value: str) -> List[int]:
    parts = [p for p in value.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("draft_lens must not be empty")
    lens = [int(p) for p in parts]
    if any(dl <= 1 for dl in lens):
        raise ValueError("draft_lens values must be > 1")
    return lens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Distributional invariance test")
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--draft_lens", type=str, default="2,8,20")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--num_tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def normalize(counter: Counter) -> Dict[int, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in counter.items()}


def l1_distance(p: Dict[int, float], q: Dict[int, float]) -> float:
    keys = set(p) | set(q)
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def kl_divergence(p: Dict[int, float], q: Dict[int, float], eps: float = 1e-8) -> float:
    total = 0.0
    for token, p_val in p.items():
        if p_val <= 0.0:
            continue
        q_val = q.get(token, 0.0)
        total += p_val * np.log(p_val / (q_val + eps))
    return float(total)


def top_tokens(dist: Dict[int, float], tokenizer, top_n: int = 5) -> List[Tuple[int, float, str]]:
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(tok, prob, tokenizer.decode([tok])) for tok, prob in sorted_items]


def top_differences(
    baseline: Dict[int, float],
    candidate: Dict[int, float],
    tokenizer,
    top_n: int = 5,
) -> List[Tuple[int, float, float, str]]:
    keys = set(baseline) | set(candidate)
    diffs = []
    for tok in keys:
        base_prob = baseline.get(tok, 0.0)
        cand_prob = candidate.get(tok, 0.0)
        diffs.append((tok, cand_prob - base_prob))
    diffs.sort(key=lambda item: abs(item[1]), reverse=True)
    top = diffs[:top_n]
    return [(tok, baseline.get(tok, 0.0), candidate.get(tok, 0.0), tokenizer.decode([tok])) for tok, _ in top]


def collect_histograms(
    *,
    generate_fn,
    params,
    cache_vars,
    out_ids,
    prefix_len,
    prev_logit,
    prompt_len: int,
    num_tokens: int,
    seed: int,
    num_samples: int,
) -> List[Counter]:
    counters = [Counter() for _ in range(num_tokens)]
    for i in range(num_samples):
        rng = jax.random.PRNGKey(seed + i)
        out_ids_final, _, generated, _ = generate_fn(
            params,
            cache_vars,
            out_ids,
            jnp.asarray(prefix_len, dtype=jnp.int32),
            jnp.asarray(num_tokens, dtype=jnp.int32),
            prev_logit,
            rng,
        )
        out_ids_final.block_until_ready()
        generated_count = int(np.asarray(generated))
        if generated_count < num_tokens:
            raise RuntimeError(
                f"Generated fewer tokens than requested ({generated_count} < {num_tokens})."
            )
        tokens = np.asarray(out_ids_final[prompt_len : prompt_len + num_tokens], dtype=np.int32)
        for pos, tok in enumerate(tokens):
            counters[pos][int(tok)] += 1
    return counters


def collect_histograms_ar(
    *,
    model,
    params,
    cache_vars,
    prefix_len,
    prev_logit,
    prompt_len: int,
    num_tokens: int,
    seed: int,
    num_samples: int,
    temperature: float,
    top_k: int,
    eos_id: int,
    stop_on_eos: bool,
    kv_cache_len: int,
) -> List[Counter]:
    counters = [Counter() for _ in range(num_tokens)]
    for i in range(num_samples):
        rng = jax.random.PRNGKey(seed + i)
        cache = cache_vars
        cur_prefix_len = int(prefix_len)
        cur_logit = prev_logit
        tokens = []

        for _ in range(num_tokens):
            rng, token = sample_tokens(rng, cur_logit, temperature, top_k)
            token = token.astype(jnp.int32)
            tokens.append(int(token))

            if stop_on_eos and eos_id >= 0 and int(token) == eos_id:
                break

            token_pos = jnp.array([cur_prefix_len], dtype=jnp.int32)
            logits, mutated = model.apply(
                {"params": params, "cache": cache},
                token[None, None],
                deterministic=True,
                use_kv_cache=True,
                write_to_cache=True,
                cur_index=cur_prefix_len,
                position_ids=token_pos[None, :],
                kv_cache_len=kv_cache_len,
                mutable=["cache"],
            )
            cache = mutated["cache"]
            cur_prefix_len += 1
            cur_logit = logits[0, -1]

        if len(tokens) < num_tokens:
            raise RuntimeError(
                f"Generated fewer tokens than requested ({len(tokens)} < {num_tokens})."
            )

        for pos, tok in enumerate(tokens):
            counters[pos][int(tok)] += 1

    return counters


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if args.num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0 for non-greedy sampling")
    if args.top_k < 0:
        raise ValueError("top_k must be >= 0")

    draft_lens = parse_draft_lens(args.draft_lens)
    baseline_label = "ar"

    cfg = load_configs()
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    tokenizer = load_tokenizer(cfg)

    context_length = args.context_length if args.context_length is not None else int(cfg.model.context_length)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=False)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens")

    prompt_len = int(prompt_ids.shape[0])
    max_draft_len = max(draft_lens)
    required_len = prompt_len + args.num_tokens + max_draft_len + 1
    if required_len > context_length:
        raise ValueError(
            f"Required length {required_len} exceeds context_length {context_length}"
        )

    # Setup mask token (match inference.py behavior)
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"Added mask token '{mask_token}' (id={mask_id})")
    else:
        print(f"Using mask token '{mask_token}' (id={mask_id})")

    model = build_model(cfg, len(tokenizer), context_length)
    params = load_params(checkpoint_path)

    rng = jax.random.PRNGKey(args.seed)
    rng, resize_key = jax.random.split(rng)
    params, added_rows = resize_embedding_params(params, len(tokenizer), key=resize_key)
    if added_rows:
        print(f"Expanded embeddings by {added_rows} rows")

    params = jax.device_put(params)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eos_id = tokenizer.eos_token_id
    eos_id_for_jit = int(eos_id) if eos_id is not None else -1

    # Prefill prompt once and reuse across samples.
    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)
    cache_vars, prefix_len, prev_logit = prefill_prompt(
        model,
        params,
        cache_vars,
        jnp.asarray(prompt_ids),
        kv_cache_len=context_length,
    )

    buffer_len = required_len
    out_host = np.full((buffer_len,), pad_token_id, dtype=np.int32)
    out_host[:prompt_len] = prompt_ids
    out_ids = jax.device_put(jnp.asarray(out_host))

    results = {}
    distributions = {}

    ar_counters = collect_histograms_ar(
        model=model,
        params=params,
        cache_vars=cache_vars,
        prefix_len=prefix_len,
        prev_logit=prev_logit,
        prompt_len=prompt_len,
        num_tokens=args.num_tokens,
        seed=args.seed,
        num_samples=args.num_samples,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        eos_id=eos_id_for_jit,
        stop_on_eos=args.stop_on_eos,
        kv_cache_len=context_length,
    )
    distributions[baseline_label] = [normalize(counter) for counter in ar_counters]

    for draft_len in draft_lens:
        generate_fn = make_anchor_tidar_generate_fn(
            model,
            cache_len=context_length,
            draft_len=draft_len,
            mask_id=int(mask_id),
            pad_token_id=int(pad_token_id),
            eos_id=eos_id_for_jit,
            stop_on_eos=args.stop_on_eos,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            bias_value=float(cfg.tidar.attn_bias_value),
            verbose_stats=False,
        )

        counters = collect_histograms(
            generate_fn=generate_fn,
            params=params,
            cache_vars=cache_vars,
            out_ids=out_ids,
            prefix_len=prefix_len,
            prev_logit=prev_logit,
            prompt_len=prompt_len,
            num_tokens=args.num_tokens,
            seed=args.seed,
            num_samples=args.num_samples,
        )

        dist = [normalize(counter) for counter in counters]
        distributions[draft_len] = dist

    baseline_dist = distributions[baseline_label]

    print("\n=== Distributional Invariance Test ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Draft lengths: {draft_lens}")
    print(f"Samples per draft_len: {args.num_samples}")
    print(f"Tokens analyzed: {args.num_tokens}")
    print(f"Baseline: {baseline_label}")

    summary = {}
    for draft_len in draft_lens:
        summary[draft_len] = {"l1": [], "kl": []}

    for pos in range(args.num_tokens):
        print(f"\n[Position {pos}] Baseline top tokens:")
        for tok, prob, text in top_tokens(baseline_dist[pos], tokenizer):
            print(f"  {tok:>6}  p={prob:.4f}  {text!r}")

        for draft_len in draft_lens:
            l1 = l1_distance(baseline_dist[pos], distributions[draft_len][pos])
            kl = kl_divergence(baseline_dist[pos], distributions[draft_len][pos])
            summary[draft_len]["l1"].append(l1)
            summary[draft_len]["kl"].append(kl)
            print(f"  vs draft_len={draft_len}: L1={l1:.6f}, KL={kl:.6f}")
            diff_tokens = top_differences(baseline_dist[pos], distributions[draft_len][pos], tokenizer)
            for tok, base_prob, cand_prob, text in diff_tokens:
                delta = cand_prob - base_prob
                print(f"    Δ {tok:>6}  {base_prob:.4f} → {cand_prob:.4f} ({delta:+.4f})  {text!r}")

    print("\n=== Summary (vs AR baseline) ===")
    for draft_len in draft_lens:
        l1_vals = summary[draft_len]["l1"]
        kl_vals = summary[draft_len]["kl"]
        l1_mean = float(np.mean(l1_vals))
        l1_max = float(np.max(l1_vals))
        kl_mean = float(np.mean(kl_vals))
        kl_max = float(np.max(kl_vals))
        print(
            f"draft_len={draft_len}: "
            f"L1 mean={l1_mean:.6f}, L1 max={l1_max:.6f}, "
            f"KL mean={kl_mean:.6f}, KL max={kl_max:.6f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = Path(cfg.paths.data_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt": args.prompt,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "draft_lens": draft_lens,
            "num_samples": args.num_samples,
            "num_tokens": args.num_tokens,
            "baseline": baseline_label,
            "distributions": distributions,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
