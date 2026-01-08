from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

import Run_inference_sudoku as inference
from config_utils import load_config, resolve_data_path
from tokenizer_utils import build_custom_tokenizer, load_tokenizer


def _find_latest_checkpoint(root: Path) -> Optional[Path]:
    checkpoints = list(root.glob("step_*.npz"))
    if not checkpoints:
        return None

    def step_num(path: Path) -> int:
        match = re.search(r"step_(\\d+)", path.stem)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=step_num)


def _resolve_prompt_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _load_prompts(prompts_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    base = prompts_path.parent
    loaded = []
    for prompt in prompts:
        prompt_file = _resolve_prompt_path(base, prompt["file"])
        loaded.append(
            {
                "id": prompt.get("id", prompt_file.stem),
                "label": prompt.get("label", prompt.get("id", prompt_file.stem)),
                "file": prompt_file,
                "text": prompt_file.read_text(encoding="utf-8").strip(),
            }
        )
    return loaded


def _prepare_prompt_ids(cfg, tokenizer, prompt_text: str, puzzle_str: str) -> np.ndarray:
    prompt = inference._build_prompt(cfg, puzzle_str, prompt_text)
    max_seq_len = max(stage.seq_len for stage in cfg.stages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1) :]
    return np.asarray(prompt_ids, dtype=np.int32)


def _run_single(
    cfg,
    tokenizer,
    model,
    params,
    prefill_fn,
    decode_fn,
    prompt_text: str,
    seed: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    puzzle, solution = inference._generate_puzzle(seed)
    puzzle_str = inference._format_puzzle(puzzle)
    prompt_ids = _prepare_prompt_ids(cfg, tokenizer, prompt_text, puzzle_str)

    key_params, key_dropout = jax.random.split(jax.random.PRNGKey(seed), 2)
    _, nonparam = inference.init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=tokenizer.pad_token_id or 0,
        use_kv_cache=True,
    )

    prompt_ids_b = prompt_ids[None, :]
    nonparam, last_pos, last_tok = prefill_fn(params, nonparam, prompt_ids_b)
    new_ids, _ = decode_fn(
        params,
        nonparam,
        last_tok,
        last_pos,
        steps=int(max_new_tokens),
        do_sample=False,
        top_k=0,
        temperature=1.0,
        rng_key=None,
    )

    full_ids = np.concatenate([prompt_ids, np.array(new_ids[0])], axis=0)
    extracted, _segment_text, open_found, close_found = inference._extract_trm_digits(
        tokenizer, full_ids, len(prompt_ids)
    )
    matches_input = (
        extracted is not None and bool(np.all(extracted.reshape(-1) == puzzle.reshape(-1)))
    )

    solved_match = False
    if extracted is not None:
        solved = inference.solve_sudoku(extracted)
        if solved is not None and solution is not None:
            solved_match = bool(np.all(solved.reshape(-1) == solution.reshape(-1)))

    ok = bool(open_found and close_found and matches_input and solved_match)
    return {
        "seed": seed,
        "open_found": bool(open_found),
        "close_found": bool(close_found),
        "matches_input": bool(matches_input),
        "solved_match": bool(solved_match),
        "ok": ok,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate OOD Sudoku prompts (LLM-only).")
    ap.add_argument("--checkpoint", default=None, help="Path to .npz checkpoint.")
    ap.add_argument("--prompts_json", default=None, help="Path to prompts.json.")
    ap.add_argument("--output_json", default=None, help="Where to write results.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--num_runs", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    build_custom_tokenizer(force=False)
    tokenizer = load_tokenizer()
    inference._apply_compute_dtype_override(cfg)

    prompts_path = (
        Path(args.prompts_json)
        if args.prompts_json
        else resolve_data_path(cfg, "trm_token_tool/ood_eval/prompts.json")
    )
    if prompts_path is None or not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    output_path = (
        Path(args.output_json)
        if args.output_json
        else resolve_data_path(cfg, "trm_token_tool/ood_eval/results.json")
    )
    if output_path is None:
        raise FileNotFoundError("Failed to resolve output path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(Path(cfg.paths.checkpoint_root))
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found.")

    max_seq_len = max(stage.seq_len for stage in cfg.stages)
    model = inference.GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=0.0,
    )
    params = inference.load_npz(str(ckpt_path))
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
    prefill_fn, decode_fn = inference.make_prefill_and_decode_fns(model)

    prompts = _load_prompts(prompts_path)
    results = []
    total_ok = 0
    total_runs = 0
    for prompt in prompts:
        runs = []
        for seed in range(args.num_runs):
            run = _run_single(
                cfg,
                tokenizer,
                model,
                params,
                prefill_fn,
                decode_fn,
                prompt["text"],
                seed,
                args.max_new_tokens,
            )
            runs.append(run)
        success = sum(int(r["ok"]) for r in runs)
        total_ok += success
        total_runs += args.num_runs
        results.append(
            {
                "id": prompt["id"],
                "label": prompt["label"],
                "file": str(prompt["file"]),
                "success": success,
                "total": args.num_runs,
                "rate": success / float(args.num_runs),
                "runs": runs,
            }
        )

    output = {
        "checkpoint": str(ckpt_path),
        "max_new_tokens": args.max_new_tokens,
        "num_runs": args.num_runs,
        "total_prompts": len(results),
        "aggregate_success": total_ok,
        "aggregate_total": total_runs,
        "aggregate_rate": (total_ok / float(total_runs)) if total_runs else 0.0,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for row in results:
        print(f"{row['label']}: {row['success']}/{row['total']}")


if __name__ == "__main__":
    main()
