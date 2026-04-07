from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from GIANT.v3.Long.longdsl import render_messages
from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.Run_training import _to_dtype, load_configs, load_tokenizer
from GIANT.v3.model.checkpoint_manager import latest as latest_ckpt, load_npz


def _assistant_token_positions(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    text, spans = render_messages(messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(x) for x in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    if not spans:
        return input_ids, []
    span_start, span_end = spans[0]
    positions: list[int] = []
    for idx, (start, end) in enumerate(offsets):
        if end > span_start and start < span_end:
            positions.append(idx)
    return input_ids, positions


def _load_model(cfg: OmegaConf, vocab_size: int) -> GiantGPT:
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=int(cfg.model.context_length),
        d_model=int(cfg.model.embedding_size),
        n_heads=int(cfg.model.num_heads),
        d_ff=int(cfg.model.feed_forward_size),
        n_layers=int(cfg.model.num_layers),
        dropout_rate=float(cfg.model.dropout_rate),
        num_kv_heads=int(cfg.model.get("num_kv_heads", cfg.model.num_heads)),
        rotary_dim=int(cfg.model.get("rope_dim", 64)),
        param_dtype=_to_dtype(cfg.model.param_dtype),
        compute_dtype=_to_dtype(cfg.model.compute_dtype),
        use_remat=bool(cfg.model.get("use_remat", False)),
        enable_xsa=bool(cfg.model.get("enable_xsa", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced Long hidden-world exact-match evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--global_config", default=None)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    cfg = load_configs(config_path=args.config, global_config_path=args.global_config)
    tokenizer = load_tokenizer(cfg)
    model = _load_model(cfg, len(tokenizer))

    ckpt_path = args.checkpoint
    if ckpt_path == "latest":
        ckpt_dir = Path(cfg.paths.checkpoints_root)
        resolved = latest_ckpt(str(ckpt_dir))
        if resolved is None:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
        ckpt_path = resolved

    params = jax.tree_util.tree_map(jnp.asarray, load_npz(ckpt_path, print_name=False))
    records = []
    with Path(args.jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= args.limit:
                break

    ctx = int(cfg.model.context_length)
    correct = 0
    total = 0
    for row in records:
        if "text" in row and "answer_token_index" in row:
            encoded = tokenizer(row["text"], add_special_tokens=False)
            input_ids = [int(x) for x in encoded["input_ids"]]
            answer_positions = [int(row["answer_token_index"])]
        else:
            input_ids, answer_positions = _assistant_token_positions(tokenizer, row["messages"])
            if len(answer_positions) <= 1:
                continue
            answer_positions = answer_positions[1:]
        if len(input_ids) > ctx:
            continue
        x = input_ids + [tokenizer.pad_token_id] * (ctx - len(input_ids))
        x_arr = jnp.asarray([x], dtype=jnp.int32)
        logits = model.apply({"params": params}, x_arr, deterministic=True)

        ok = True
        for pos in answer_positions:
            if pos <= 0:
                ok = False
                break
            pred = int(jnp.argmax(logits[0, pos - 1]))
            if pred != input_ids[pos]:
                ok = False
                break
        correct += int(ok)
        total += 1

    accuracy = (correct / total) if total else 0.0
    print(f"[long-eval] checkpoint={ckpt_path}")
    print(f"[long-eval] total={total} correct={correct} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
