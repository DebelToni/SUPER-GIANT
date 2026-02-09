"""
Acceptance-rate controlled throughput test for Anchor-TiDAR inference.

This script reuses the regular TiDAR inference stack (config/model/tokenizer/KV
cache/prefill), but replaces rejection sampling with a deterministic PRNG-driven
acceptance controller:

- With probability `--accept-rate`, an iteration accepts all K draft tokens.
- Otherwise, it accepts exactly 1 token after the anchor.

It can run from a real checkpoint or from fully random initialized weights when
`--checkpoint` is omitted.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from TiDAR.model.inference import (
    build_model,
    load_configs,
    load_params,
    load_tokenizer,
    resolve_checkpoint_path,
    tokenize_prompt,
)
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from TiDAR.model.tidar_core import (
    build_decode_bias_template_step_prefix,
    build_decode_position_template,
    init_kv_cache,
    prefill_prompt_with_draft,
    sample_tokens,
)


def controlled_accept(
    key: jax.Array,
    *,
    anchor_token: jnp.ndarray,
    draft_tokens: jnp.ndarray,     # [K-1]
    verify_logits: jnp.ndarray,    # [K, V]
    draft_logits: jnp.ndarray,     # [K, V]
    predraft_tokens: jnp.ndarray,  # [K, K]
    temperature: float,
    top_k: int,
    accept_rate: jnp.ndarray,      # scalar float32 in [0, 1]
):
    """Deterministic PRNG-driven acceptance controller."""
    del anchor_token, draft_logits
    k = draft_tokens.shape[0] + 1
    k_minus_1 = draft_tokens.shape[0]

    key, key_accept, key_first, key_bonus = jax.random.split(key, 4)
    accept_all = jax.random.bernoulli(key_accept, p=jnp.clip(accept_rate, 0.0, 1.0))

    # If rejecting the iteration, we still need one committed token.
    first_logits = verify_logits[0:1]
    _, first_token = sample_tokens(key_first, first_logits, temperature, top_k)
    first_token = first_token[0]

    # Bonus token used when all K are accepted (and as filler for shape stability).
    bonus_logits = verify_logits[-1:]
    _, bonus_token = sample_tokens(key_bonus, bonus_logits, temperature, top_k)
    bonus_token = bonus_token[0]

    committed_accept = jnp.concatenate([draft_tokens, bonus_token[None]], axis=0)  # [K]
    reject_tail = jnp.concatenate([draft_tokens[1:], bonus_token[None]], axis=0)   # [K-1]
    committed_reject = jnp.concatenate([first_token[None], reject_tail], axis=0)   # [K]
    committed = jnp.where(accept_all, committed_accept, committed_reject)

    accepted_count = jnp.where(accept_all, jnp.asarray(k, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    proposal_idx = jnp.where(accept_all, jnp.asarray(k_minus_1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    selected_proposal = predraft_tokens[proposal_idx]

    next_anchor = jnp.where(accept_all, bonus_token, first_token)
    selected_proposal = selected_proposal.at[0].set(next_anchor)

    return key, accepted_count, committed, selected_proposal


def make_generate_fn(
    model,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    pad_token_id: int,
    eos_id: int,
    stop_on_eos: bool,
    temperature: float,
    top_k: int,
    bias_value: float,
):
    """Build JIT decode function that uses an externally supplied acceptance fn."""
    decode_bias = jax.device_put(build_decode_bias_template_step_prefix(cache_len, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))

    def decode_apply(params, cache_vars, step_tokens, pos_ids, prefix_len):
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            cache_write_len=draft_len,
            attn_bias=decode_bias,
            position_ids=pos_ids[None, :],
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]

    @jax.jit
    def generate(
        params,
        cache_vars,
        out_ids: jnp.ndarray,
        prefix_len: jnp.ndarray,
        max_steps: jnp.ndarray,
        prev_logit: jnp.ndarray,
        rng_key: jax.Array,
        initial_draft_logits: jnp.ndarray,
        accept_rate: jnp.ndarray,
    ):
        prefix_len = prefix_len.astype(jnp.int32)
        generated = jnp.array(0, dtype=jnp.int32)
        done = jnp.array(False)

        total_accepts = jnp.array(0, dtype=jnp.int32)
        n_iterations = jnp.array(0, dtype=jnp.int32)
        max_accepts = jnp.array(0, dtype=jnp.int32)

        rng_key, anchor = sample_tokens(rng_key, prev_logit, temperature, top_k)
        anchor = anchor.astype(jnp.int32)
        rng_key, init_draft = sample_tokens(rng_key, initial_draft_logits, temperature, top_k)
        init_draft = init_draft.astype(jnp.int32)
        current_draft = init_draft.at[0].set(anchor)
        current_draft_logits = initial_draft_logits

        def cond_fn(state):
            _, _, _, _, _, generated, _, done, _, _, _ = state
            return (generated < max_steps) & (~done)

        def body_fn(state):
            (rng, cache, out, prefix_len, current_draft, generated,
             current_draft_logits, done, total_accepts, n_iters, max_acc) = state

            step_tokens = jnp.concatenate([current_draft, predraft_masks])
            step_pos_ids = (prefix_len + position_template).astype(jnp.int32)
            logits, cache = decode_apply(params, cache, step_tokens, step_pos_ids, prefix_len)
            verify_logits = logits[:draft_len]
            predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)

            rng, predraft_flat_tokens = sample_tokens(
                rng,
                predraft_logits.reshape(-1, predraft_logits.shape[-1]),
                temperature,
                top_k,
            )
            predraft_tokens = predraft_flat_tokens.reshape(draft_len, draft_len).astype(jnp.int32)

            rng, accept_count, _, next_draft = controlled_accept(
                rng,
                anchor_token=current_draft[0],
                draft_tokens=current_draft[1:],
                verify_logits=verify_logits,
                draft_logits=current_draft_logits,
                predraft_tokens=predraft_tokens,
                temperature=temperature,
                top_k=top_k,
                accept_rate=accept_rate,
            )

            remaining = max_steps - generated
            eff_accept = jnp.minimum(accept_count, remaining)

            has_eos = jnp.array(False)
            if stop_on_eos and eos_id >= 0:
                eos_mask = (current_draft == eos_id) & (idx_k < eff_accept)
                first_eos = jnp.where(
                    jnp.any(eos_mask),
                    jnp.argmax(eos_mask.astype(jnp.int32)),
                    draft_len,
                ).astype(jnp.int32)
                has_eos = first_eos < eff_accept
                eff_accept = jnp.where(has_eos, first_eos + 1, eff_accept)

            commit_padded = jnp.where(idx_k < eff_accept, current_draft, pad_token_id).astype(jnp.int32)
            out = jax.lax.dynamic_update_slice(out, commit_padded[:draft_len], (prefix_len,))

            prefix_len2 = prefix_len + eff_accept
            generated2 = generated + eff_accept
            done2 = done | (generated2 >= max_steps) | has_eos

            k_minus_1 = draft_len - 1
            stopped = accept_count < draft_len
            proposal_idx = jnp.where(stopped, accept_count - 1, k_minus_1)
            proposal_idx = jnp.clip(proposal_idx, 0, k_minus_1)
            next_draft_logits = predraft_logits[proposal_idx]

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
                next_draft_logits,
                done2,
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
            done,
            total_accepts,
            n_iterations,
            max_accepts,
        )
        state_final = jax.lax.while_loop(cond_fn, body_fn, state0)

        (_, _, out_final, prefix_len_final, _, generated_final, _, _, total_accepts_final, n_iters_final, max_accepts_final) = state_final
        stats = {
            "total_accepts": total_accepts_final,
            "n_iterations": n_iters_final,
            "avg_accept_per_iter": total_accepts_final.astype(jnp.float32) / jnp.maximum(n_iters_final, 1),
            "max_accept_per_iter": max_accepts_final,
        }
        return out_final, prefix_len_final, generated_final, stats

    return generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Anchor-TiDAR throughput test with controlled accept rate")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="If omitted, random params are used.")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument(
        "--prefill_tokens",
        type=int,
        default=None,
        help="If set, override prompt token length with an exact token count.",
    )
    parser.add_argument(
        "--prefill_token_id",
        type=int,
        default=None,
        help="Token id used when --prefill_tokens is set (defaults to tokenizer EOS or 0).",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    parser.add_argument("--accept-rate", type=float, default=0.4, help="Iteration acceptance probability in [0, 1].")
    parser.add_argument("--silent", action="store_true", help="Do not print decoded text.")
    return parser.parse_args()


def _resolve_stop_on_eos(arg_val: Optional[str], default_val: bool) -> bool:
    if arg_val is None:
        return bool(default_val)
    return arg_val.lower() in ("true", "1", "yes")


def _build_prompt_ids(
    tokenizer,
    *,
    prompt: str,
    context_length: int,
    strip_eos: bool,
    prefill_tokens: Optional[int],
    prefill_token_id: Optional[int],
) -> np.ndarray:
    if prefill_tokens is None:
        return tokenize_prompt(tokenizer, prompt, context_length, strip_eos=strip_eos)

    if prefill_tokens < 0:
        raise ValueError("--prefill_tokens must be >= 0")
    if prefill_tokens > context_length:
        raise ValueError(f"--prefill_tokens {prefill_tokens} exceeds context_length {context_length}")

    if prefill_token_id is not None:
        token_id = int(prefill_token_id)
    elif tokenizer.eos_token_id is not None:
        token_id = int(tokenizer.eos_token_id)
    else:
        token_id = 0

    if prefill_tokens == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.full((prefill_tokens,), token_id, dtype=np.int32)


def main():
    args = parse_args()
    if not (0.0 <= args.accept_rate <= 1.0):
        raise ValueError("--accept-rate must be in [0, 1]")

    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)
    stop_on_eos = _resolve_stop_on_eos(args.stop_on_eos, bool(cfg.inference.stop_on_eos))

    model_context_length = int(cfg.model.context_length)
    context_length = args.context_length if args.context_length is not None else model_context_length
    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if draft_len <= 1:
        raise ValueError("draft_len must be > 1")
    if context_length > model_context_length:
        raise ValueError(f"context_length {context_length} exceeds model max {model_context_length}")

    tokenizer = load_tokenizer(cfg)
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"Added mask token '{mask_token}' (id={mask_id})")

    model = build_model(cfg, len(tokenizer), context_length, draft_len)
    rng = jax.random.PRNGKey(args.seed)

    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
        print(f"Checkpoint mode: {checkpoint_path}")
        params = load_params(Path(checkpoint_path))
        rng, resize_key = jax.random.split(rng)
        params, added_rows = resize_embedding_params(params, len(tokenizer), key=resize_key)
        if added_rows:
            print(f"Expanded embeddings by {added_rows} rows")
    else:
        print("Checkpoint mode: random initialization")
        rng, init_key = jax.random.split(rng)
        dummy_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
        init_vars = model.init({"params": init_key}, dummy_tokens, deterministic=True, use_kv_cache=False)
        params = init_vars["params"]

    params = jax.device_put(params)

    prompt_ids = _build_prompt_ids(
        tokenizer,
        prompt=args.prompt,
        context_length=context_length,
        strip_eos=args.strip_eos,
        prefill_tokens=args.prefill_tokens,
        prefill_token_id=args.prefill_token_id,
    )
    prompt_len = int(prompt_ids.shape[0])
    required_len = prompt_len + max_steps + draft_len + 1
    if required_len > context_length:
        raise ValueError(f"Required length {required_len} exceeds context_length {context_length}")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id
    eos_id_for_jit = int(eos_id) if eos_id is not None else -1

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)

    cache_vars, prefix_len, prev_logit, initial_draft_logits = prefill_prompt_with_draft(
        model,
        params,
        cache_vars,
        jnp.asarray(prompt_ids),
        draft_len=draft_len,
        mask_id=int(mask_id),
        kv_cache_len=context_length,
        bias_value=bias_value,
    )

    out_host = np.full((required_len,), pad_token_id, dtype=np.int32)
    out_host[:prompt_len] = prompt_ids
    out_ids = jax.device_put(jnp.asarray(out_host))

    generate_fn = make_generate_fn(
        model,
        cache_len=context_length,
        draft_len=draft_len,
        mask_id=int(mask_id),
        pad_token_id=int(pad_token_id),
        eos_id=eos_id_for_jit,
        stop_on_eos=stop_on_eos,
        temperature=float(temperature),
        top_k=int(top_k),
        bias_value=bias_value,
    )

    decode_start = time.perf_counter()
    out_ids_final, final_len, generated, stats = generate_fn(
        params,
        cache_vars,
        out_ids,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(max_steps, dtype=jnp.int32),
        prev_logit,
        rng,
        initial_draft_logits,
        jnp.asarray(args.accept_rate, dtype=jnp.float32),
    )
    out_ids_final.block_until_ready()
    decode_time = time.perf_counter() - decode_start

    final_len = int(np.asarray(final_len))
    out_tokens = np.asarray(out_ids_final[:final_len], dtype=np.int32)
    text = tokenizer.decode(out_tokens, skip_special_tokens=True)

    generated_count = int(np.asarray(generated))
    n_iters = int(np.asarray(stats["n_iterations"]))
    avg_accept = float(np.asarray(stats["avg_accept_per_iter"]))
    max_accept = int(np.asarray(stats["max_accept_per_iter"]))
    toks_per_s = generated_count / decode_time if decode_time > 0 else float("inf")
    accept_prob_proxy = (avg_accept - 1.0) / max(draft_len - 1, 1)

    if not args.silent:
        print("\n==================== RESULT ====================")
        print(text)
        print("================================================")

    print("\n[metrics: decode only]")
    print(f"  prompt_tokens:                 {prompt_len}")
    print(f"  generated_tokens:              {generated_count}")
    print(f"  n_iterations:                  {n_iters}")
    print(f"  target_accept_rate:            {args.accept_rate:.6f}")
    print(f"  observed_avg_accept_per_iter:  {avg_accept:.6f}")
    print(f"  observed_max_accept_per_iter:  {max_accept}")
    print(f"  observed_accept_prob_proxy:    {accept_prob_proxy:.6f}")
    print(f"  decode_time_s:                 {decode_time:.6f}")
    print(f"  tokens_per_second_decode_only: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
