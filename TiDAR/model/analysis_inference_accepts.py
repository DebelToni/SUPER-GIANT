"""Anchor-TiDAR debug inference: per-iteration accept counts and drafts.

This script is intentionally slow and un-jitted for interpretability.
It prints, for each iteration, how many tokens were accepted and the draft tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from TiDAR.model.inference import (
    build_model,
    load_configs,
    load_params,
    load_tokenizer,
    resolve_checkpoint_path,
    tokenize_prompt,
)
from TiDAR.model.tidar_core import (
    anchor_rejection_sample,
    build_decode_bias_template,
    build_decode_position_template,
    init_kv_cache,
    prefill_prompt_with_draft,
    sample_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Anchor-TiDAR debug inference (accept trace)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    return parser.parse_args()


def format_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    stripped = text.lstrip()
    if stripped != text:
        text = " " + stripped
    return text


def format_tokens(tokenizer, token_ids) -> list[str]:
    return [format_token(tokenizer, t) for t in token_ids]


def format_tokens_padded(tokens: list[str], width: int) -> str:
    return " | ".join(t.ljust(width) for t in tokens)


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    # Settings
    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)

    if args.stop_on_eos is not None:
        stop_on_eos = args.stop_on_eos.lower() in ("true", "1", "yes")
    else:
        stop_on_eos = bool(cfg.inference.stop_on_eos)

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

    # Load checkpoint + tokenizer
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    tokenizer = load_tokenizer(cfg)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens")

    prompt_len = int(prompt_ids.shape[0])
    required_len = prompt_len + max_steps + draft_len + 1
    if required_len > context_length:
        raise ValueError(f"Required length {required_len} exceeds context_length {context_length}")

    # Ensure mask token
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"Added mask token '{mask_token}' (id={mask_id})")

    # Build model + params
    model = build_model(cfg, len(tokenizer), context_length, draft_len)
    params = load_params(Path(checkpoint_path))
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

    # Init cache
    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)

    # Prefill prompt + initial draft
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

    # Pre-build templates
    decode_bias = jax.device_put(build_decode_bias_template(context_length, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))

    def decode_apply(params, cache_vars, step_tokens, pos_ids, prefix_len_val):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len_val,
            attn_bias=decode_bias,
            position_ids=pos_ids[None, :],
            kv_cache_len=context_length,
        )
        return logits[0]

    def cache_write_apply(params, cache_vars, tokens, pos_ids, cur_index):
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=cur_index,
            position_ids=pos_ids[None, :],
            kv_cache_len=context_length,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]

    # Track output tokens
    output_ids = [int(t) for t in prompt_ids.tolist()]

    # Sample first anchor and commit it
    rng, anchor = sample_tokens(rng, prev_logit, temperature, top_k)
    anchor = anchor.astype(jnp.int32)
    anchor_pos = jnp.array([prefix_len], dtype=jnp.int32)
    _, cache_vars = cache_write_apply(params, cache_vars, anchor[None], anchor_pos, prefix_len)
    prefix_len = prefix_len + 1
    generated = 1
    output_ids.append(int(anchor))
    done = generated >= max_steps
    if stop_on_eos and eos_id_for_jit >= 0:
        done = done | (anchor == eos_id_for_jit)

    # Sample initial draft
    rng, init_draft = sample_tokens(rng, initial_draft_logits, temperature, top_k)
    init_draft = init_draft.astype(jnp.int32)
    current_draft = init_draft.at[0].set(anchor)
    current_draft_logits = initial_draft_logits

    trace_records = []

    iteration = 0
    while (generated < max_steps) and (not bool(done)):
        iteration += 1

        # Build decode input
        step_tokens = jnp.concatenate([current_draft, predraft_masks])
        step_pos_ids = (prefix_len - 1 + position_template).astype(jnp.int32)

        # Forward pass
        logits = decode_apply(params, cache_vars, step_tokens, step_pos_ids, prefix_len - 1)
        verify_logits = logits[:draft_len]
        predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)

        # Sample predraft groups
        rng, predraft_flat_tokens = sample_tokens(
            rng,
            predraft_logits.reshape(-1, predraft_logits.shape[-1]),
            temperature,
            top_k,
        )
        predraft_tokens = predraft_flat_tokens.reshape(draft_len, draft_len).astype(jnp.int32)

        # Rejection sampling
        anchor_tok = current_draft[0]
        draft_toks = current_draft[1:]
        rng, accept_count, committed, next_draft = anchor_rejection_sample(
            rng,
            anchor_token=anchor_tok,
            draft_tokens=draft_toks,
            verify_logits=verify_logits,
            draft_logits=current_draft_logits,
            predraft_tokens=predraft_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        remaining = max_steps - generated
        eff_accept = int(jnp.minimum(accept_count, remaining))

        # EOS check
        has_eos = False
        if stop_on_eos and eos_id_for_jit >= 0:
            eos_mask = (committed == eos_id_for_jit) & (idx_k < eff_accept)
            if bool(jnp.any(eos_mask)):
                first_eos = int(jnp.argmax(eos_mask.astype(jnp.int32)))
                has_eos = first_eos < eff_accept
                eff_accept = first_eos + 1

        # Commit tokens
        commit_padded = jnp.where(idx_k < eff_accept, committed, pad_token_id).astype(jnp.int32)
        commit_pos_ids = (prefix_len + idx_k).astype(jnp.int32)
        _, cache_vars = cache_write_apply(params, cache_vars, commit_padded, commit_pos_ids, prefix_len)

        # Update counters
        prefix_len = prefix_len + eff_accept
        generated = generated + eff_accept
        done = (generated >= max_steps) or has_eos
        output_ids.extend([int(t) for t in committed[:eff_accept].tolist()])

        # Update next draft logits
        k_minus_1 = draft_len - 1
        n_verified = accept_count - 1
        stopped = accept_count < draft_len
        proposal_idx = jnp.where(stopped, n_verified - 1, k_minus_1)
        proposal_idx = jnp.clip(proposal_idx, 0, k_minus_1)
        next_draft_logits = predraft_logits[proposal_idx]

        # Pretty print
        anchor_id = int(anchor_tok)
        draft_ids = np.asarray(draft_toks)
        committed_ids = np.asarray(committed)[:eff_accept]
        anchor_text = format_token(tokenizer, anchor_id)
        draft_texts = format_tokens(tokenizer, draft_ids)
        commit_texts = format_tokens(tokenizer, committed_ids)
        next_draft_ids = np.asarray(next_draft)
        next_anchor_text = format_token(tokenizer, int(next_draft_ids[0]))
        next_draft_texts = format_tokens(tokenizer, next_draft_ids)
        committed_full_texts = format_tokens(tokenizer, np.asarray(committed))
        accepted_cols = [""] * draft_len
        for idx, tok in enumerate(commit_texts):
            if idx < draft_len:
                accepted_cols[idx] = tok
        accept_count_int = int(np.asarray(accept_count))
        proposal_idx_label = min(max(accept_count_int - 1, 0), draft_len - 1)
        trace_records.append(
            {
                "accept": eff_accept,
                "anchor": anchor_text,
                "next_anchor": next_anchor_text,
                "draft": [f"({anchor_text})"] + draft_texts,
                "next_draft": next_draft_texts,
                "committed_full": committed_full_texts,
                "accepted_cols": accepted_cols,
                "proposal_idx": proposal_idx_label,
            }
        )

        # Advance draft for next iteration
        current_draft = next_draft
        current_draft_logits = next_draft_logits

    if trace_records:
        max_len = 0
        for rec in trace_records:
            max_len = max(max_len, len(rec["anchor"]))
            max_len = max(max_len, len(rec["next_anchor"]))
            for tok in rec["draft"]:
                max_len = max(max_len, len(tok))
            for tok in rec["next_draft"]:
                max_len = max(max_len, len(tok))
            for tok in rec["committed_full"]:
                max_len = max(max_len, len(tok))
            for tok in rec["accepted_cols"]:
                max_len = max(max_len, len(tok))
    else:
        max_len = 1

    print(f"prompt: {args.prompt}")
    print(f"prompt tokens: {prompt_len}")
    print(f"draft len: {draft_len}; max steps: {max_steps}\n")

    accept_label = f"accept {draft_len} of {draft_len}: "
    select_label_template = f"select {draft_len}-th draft for next: "
    label_width = max(
        len("currently verified draft: "),
        len("sample from verified draft: "),
        len(accept_label),
        len(select_label_template),
    )

    for rec in trace_records:
        accept_label = f"accept {rec['accept']} of {draft_len}: "
        accept_line_prefix = accept_label.ljust(label_width)
        accept_tokens = rec["accepted_cols"][: rec["accept"]]

        print(f"anchor: {rec['anchor'].ljust(max_len)} -> {rec['next_anchor'].ljust(max_len)}")
        print(
            f"{'currently verified draft: '.ljust(label_width)}"
            f"{format_tokens_padded(rec['draft'], max_len)}"
        )
        print(
            f"{'sample from verified draft: '.ljust(label_width)}"
            f"{format_tokens_padded(rec['committed_full'], max_len)}"
        )
        print(
            f"{accept_line_prefix}"
            f"{format_tokens_padded(accept_tokens, max_len)}"
        )
        select_label = f"select {rec['proposal_idx'] + 1}-th draft for next: "
        print(
            f"{select_label.ljust(label_width)}"
            f"{format_tokens_padded(rec['next_draft'], max_len)}\n"
        )

    final_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("final output")
    print(final_text)

    iteration_count = len(trace_records)
    total_accepted = sum(int(rec["accept"]) for rec in trace_records)
    avg_accepted = total_accepted / iteration_count if iteration_count else 0.0
    accept_utilization = avg_accepted / draft_len if draft_len else 0.0
    print("\naccept metrics")
    print(f"iterations: {iteration_count}")
    print(f"accepted tokens in iterations: {total_accepted}")
    print(f"avg accepted per iteration: {avg_accepted:.3f} / {draft_len} ({accept_utilization:.2%})")


if __name__ == "__main__":
    main()
