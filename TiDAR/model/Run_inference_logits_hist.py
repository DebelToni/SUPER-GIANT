from __future__ import annotations

import argparse
from pathlib import Path

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
    anchor_rejection_sample,
    build_decode_bias_template,
    build_decode_position_template,
    init_kv_cache,
    prefill_prompt_with_draft,
    sample_tokens,
)


def _sanitize_token(token: str, max_len: int = 6) -> str:
    token = token.replace("\n", " ").replace("\t", " ")
    token = token.encode("ascii", errors="replace").decode("ascii")
    if not token:
        token = "<empty>"
    if len(token) > max_len:
        token = token[: max_len - 1] + "~"
    return token


def _typst_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "\\\"")


def _topk_probs(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    probs = np.asarray(probs)
    idx = np.argsort(-probs)[:k]
    return idx, probs[idx]


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _is_sentence_end(token: str) -> bool:
    return ("\n" in token) or any(ch in token for ch in (".", "?", "!"))


def _extract_sentence(token_texts: list[str], target_index: int) -> tuple[str, int, int]:
    if not token_texts:
        return "", 0, -1
    safe_target = min(max(target_index, 0), len(token_texts) - 1)
    start_idx = 0
    for i in range(safe_target, -1, -1):
        if _is_sentence_end(token_texts[i]):
            start_idx = i + 1
            break
    end_idx = len(token_texts) - 1
    for i in range(safe_target, len(token_texts)):
        if _is_sentence_end(token_texts[i]):
            end_idx = i
            break
    sentence = "".join(token_texts[start_idx : end_idx + 1])
    sentence = sentence.replace("\n", " ")
    sentence = sentence.lstrip()
    return sentence, start_idx, end_idx


def _format_labels(tokens: list[str]) -> str:
    items = []
    for token in tokens:
        label = _typst_escape(token)
        items.append(f'"{label}"')
    return "(" + ", ".join(items) + ")"


def _format_values(probs: np.ndarray) -> str:
    items = [f"{prob:.3f}" for prob in probs]
    return "(" + ", ".join(items) + ")"


def _write_typst_data(
    output_path: Path,
    *,
    meta: dict,
    hist_rows: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("#let hist_meta = (")
    lines.append(f'  prompt: "{_typst_escape(meta["prompt"])}",')
    lines.append(f'  checkpoint_name: "{_typst_escape(meta["checkpoint_name"])}",')
    lines.append(f'  checkpoint_path: "{_typst_escape(meta["checkpoint"])}",')
    lines.append(f"  target_position: {meta["target_position"]},")
    lines.append(f"  block_start: {meta["block_start"]},")
    lines.append(f"  block_end: {meta["block_end"]},")
    lines.append(f"  draft_len: {meta["draft_len"]},")
    lines.append(f"  temperature: {meta["temperature"]},")
    lines.append(f"  top_k: {meta["top_k"]},")
    lines.append(f'  context_text: "{_typst_escape(meta["context_text"])}",')
    lines.append(f"  context_start: {meta["context_start"]},")
    lines.append(f"  context_end: {meta["context_end"]},")
    lines.append(")")
    lines.append("")
    lines.append("#let hist_data = (")
    for row in hist_rows:
        labels = _format_labels(row["tokens"])
        ar_values = _format_values(row["ar_probs"])
        diff_values = _format_values(row["diff_probs"])
        title = _typst_escape(f"pos {row['abs_pos']}")
        diff_top = _typst_escape(row["diff_top"])
        accept = row["accept"]
        if accept is None:
            accept_literal = "none"
        else:
            accept_literal = "true" if accept else "false"
        lines.append(
            f"  (title: \"{title}\", labels: {labels}, ar: {ar_values}, diff: {diff_values}, diff_top: \"{diff_top}\", accept: {accept_literal}),"
        )
    lines.append(")")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and dump logits histograms.")
    parser.add_argument("--config", type=str, default="TiDAR/model/training_configs/TinyStories_TiDAR/TiDAR_post_training.yml")
    parser.add_argument("--global_config", type=str, default="TiDAR/Global_Config.yml")
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/proj/giant-data/TiDAR/checkpoints/TinyStories_exp/tidar_stable_bigger_beta",
    )
    parser.add_argument("--prompt", type=str, default="Once upon")
    parser.add_argument("--target_position", type=int, default=30, help="1-based position in full sequence")
    parser.add_argument("--draft_len", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--context_extra", type=int, default=32)
    parser.add_argument(
        "--output_data",
        type=str,
        default="TiDAR/Docs/Training_logs/once_upon_logits_hist_data.typ",
    )
    args = parser.parse_args()

    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    tokenizer = load_tokenizer(cfg)

    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    _, mask_id, _ = ensure_tidar_mask_token(tokenizer, base_token=base_token)

    context_length = int(cfg.model.context_length)
    draft_len = int(args.draft_len)
    bias_value = float(cfg.tidar.attn_bias_value)

    model = build_model(cfg, len(tokenizer), context_length, draft_len)
    params = load_params(Path(checkpoint_path))

    rng = jax.random.PRNGKey(0)
    rng, resize_key = jax.random.split(rng)
    params, _ = resize_embedding_params(params, len(tokenizer), key=resize_key)
    params = jax.device_put(params)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=False)
    prompt_len = int(prompt_ids.shape[0])
    prompt_ids_list = prompt_ids.tolist()
    required_len = prompt_len + args.max_steps + draft_len + 1
    if required_len > context_length:
        raise ValueError(
            f"Prompt too long: required_len {required_len} exceeds context_length {context_length}"
        )

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

    decode_bias = jax.device_put(build_decode_bias_template(context_length, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))

    def decode_apply(params, cache_vars, step_tokens, pos_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
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

    rng, anchor = sample_tokens(rng, prev_logit, args.temperature, args.top_k)
    anchor = anchor.astype(jnp.int32)
    anchor_id = int(np.asarray(anchor))
    generated_tokens: list[int] = [anchor_id]

    anchor_pos = jnp.array([prefix_len], dtype=jnp.int32)
    _, cache_vars = cache_write_apply(params, cache_vars, anchor[None], anchor_pos, prefix_len)
    prefix_len = prefix_len + 1
    generated = 1
    done = generated >= args.max_steps
    if args.temperature <= 0.0 and eos_id >= 0:
        done = done | (anchor == eos_id)

    rng, init_draft = sample_tokens(rng, initial_draft_logits, args.temperature, args.top_k)
    init_draft = init_draft.astype(jnp.int32)
    current_draft = init_draft.at[0].set(anchor)
    current_draft_logits = initial_draft_logits

    target_index = args.target_position - 1
    captured = None
    context_limit = None

    while (generated < args.max_steps) and (not bool(np.asarray(done))):
        prefix_len_int = int(np.asarray(prefix_len))
        block_start = prefix_len_int
        block_end = prefix_len_int + draft_len - 1

        step_tokens = jnp.concatenate([current_draft, predraft_masks])
        step_pos_ids = (prefix_len - 1 + position_template).astype(jnp.int32)
        decode_prefix_len = prefix_len - 1
        logits = decode_apply(params, cache_vars, step_tokens, step_pos_ids, decode_prefix_len)
        verify_logits = logits[:draft_len]
        predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)

        if block_start <= target_index <= block_end and captured is None:
            captured = {
                "block_start": block_start,
                "block_end": block_end,
                "verify_logits": np.asarray(verify_logits),
                "draft_logits": np.asarray(current_draft_logits),
                "bonus_diff_logits": np.asarray(predraft_logits[-1, 0]),
                "draft_tokens": np.asarray(current_draft),
            }
            max_full_index = prompt_len + args.max_steps - 1
            context_limit = min(target_index + args.context_extra, max_full_index)

        rng, predraft_flat = sample_tokens(
            rng,
            predraft_logits.reshape(-1, predraft_logits.shape[-1]),
            args.temperature,
            args.top_k,
        )
        predraft_tokens = predraft_flat.reshape(draft_len, draft_len).astype(jnp.int32)

        rng, accept_count, committed, next_draft = anchor_rejection_sample(
            rng,
            anchor_token=current_draft[0],
            draft_tokens=current_draft[1:],
            verify_logits=verify_logits,
            draft_logits=current_draft_logits,
            predraft_tokens=predraft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        remaining = jnp.asarray(args.max_steps - int(np.asarray(generated)), dtype=jnp.int32)
        eff_accept = jnp.minimum(accept_count, remaining)
        has_eos = jnp.array(False)
        if args.temperature <= 0.0 and eos_id >= 0:
            eos_mask = (committed == eos_id) & (idx_k < eff_accept)
            first_eos = jnp.where(
                jnp.any(eos_mask),
                jnp.argmax(eos_mask.astype(jnp.int32)),
                draft_len,
            ).astype(jnp.int32)
            has_eos = first_eos < eff_accept
            eff_accept = jnp.where(has_eos, first_eos + 1, eff_accept)

        commit_padded = jnp.where(
            idx_k < eff_accept,
            committed,
            pad_token_id,
        ).astype(jnp.int32)
        eff_accept_int = int(np.asarray(eff_accept))
        if eff_accept_int > 0:
            committed_np = np.asarray(committed).astype(np.int32)
            generated_tokens.extend(committed_np[:eff_accept_int].tolist())
        commit_pos_ids = (prefix_len + idx_k).astype(jnp.int32)
        _, cache_vars = cache_write_apply(params, cache_vars, commit_padded, commit_pos_ids, prefix_len)

        prefix_len = prefix_len + eff_accept
        generated = generated + eff_accept
        done = done | (generated >= args.max_steps) | has_eos

        k_minus_1 = draft_len - 1
        stopped = accept_count < draft_len
        proposal_idx = jnp.where(stopped, accept_count - 1, k_minus_1)
        proposal_idx = jnp.clip(proposal_idx, 0, k_minus_1)
        current_draft_logits = predraft_logits[proposal_idx]
        current_draft = next_draft

        if captured is not None and context_limit is not None:
            current_last_index = int(np.asarray(prefix_len)) - 1
            if current_last_index >= context_limit:
                break

    if captured is None:
        raise RuntimeError("Target position not reached before max_steps.")

    hist_rows = []
    block_start_pos = captured["block_start"] + 1
    block_end_pos = captured["block_end"] + 1
    draft_tokens = captured["draft_tokens"]
    greedy_accept = args.temperature <= 0.0
    if greedy_accept:
        verify_argmax = np.argmax(captured["verify_logits"][:-1], axis=-1)
        matches = draft_tokens[1:] == verify_argmax
        accept_flags = []
        accepted_so_far = True
        for match in matches:
            match_bool = bool(match)
            if accepted_so_far and match_bool:
                accept_flags.append(True)
            else:
                accept_flags.append(False)
                accepted_so_far = False
    else:
        accept_flags = [None] * (draft_len - 1)
    for i in range(draft_len):
        ar_logits = captured["verify_logits"][i]
        if i < draft_len - 1:
            diff_logits = captured["draft_logits"][i + 1]
        else:
            diff_logits = captured["bonus_diff_logits"]
        ar_idx, ar_probs = _topk_probs(ar_logits, 5)
        diff_probs = jax.nn.softmax(jnp.asarray(diff_logits), axis=-1)
        diff_probs = np.asarray(diff_probs)[ar_idx]
        diff_top_id = int(np.argmax(diff_logits))
        diff_top = _sanitize_token(_decode_token(tokenizer, diff_top_id))
        if i < draft_len - 1:
            accept = accept_flags[i]
        else:
            accept = None

        tokens = []
        for tok_id in ar_idx:
            tok = _decode_token(tokenizer, int(tok_id))
            tok = _sanitize_token(tok)
            tokens.append(tok)

        hist_rows.append(
            {
                "abs_pos": block_start_pos + i,
                "tokens": tokens,
                "ar_probs": ar_probs,
                "diff_probs": diff_probs,
                "ar_top": tokens[0],
                "diff_top": diff_top,
                "accept": accept,
            }
        )

    meta = {
        "prompt": args.prompt,
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": Path(checkpoint_path).name,
        "target_position": args.target_position,
        "block_start": block_start_pos,
        "block_end": block_end_pos,
        "draft_len": draft_len,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    full_ids = prompt_ids_list + generated_tokens
    token_texts = [_decode_token(tokenizer, tok_id) for tok_id in full_ids]
    context_text, context_start, context_end = _extract_sentence(token_texts, target_index)
    meta["context_text"] = context_text
    meta["context_start"] = context_start + 1
    meta["context_end"] = context_end + 1

    _write_typst_data(Path(args.output_data), meta=meta, hist_rows=hist_rows)
    print(f"Wrote histogram data to {args.output_data}")


if __name__ == "__main__":
    main()
