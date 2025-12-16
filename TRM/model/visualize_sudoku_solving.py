from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as flax_core
from omegaconf import OmegaConf

from TRM import TRM
from checkpoint_manager import latest as latest_ckpt
from checkpoint_manager import load as load_ckpt
from sudoku_dataset import SIDE, format_grid, load_or_generate_dataset


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDER = "\033[4m"
FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_CYAN = "\033[36m"
FG_GRAY = "\033[90m"


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def parse_args():
    p = argparse.ArgumentParser("Visualize TRM solving a Sudoku (step-by-step)")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints/trm_sudoku")
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (overrides --checkpoint_dir).")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--index", type=int, default=None, help="Puzzle index in split.")
    p.add_argument("--auto", action="store_true", help="Pick best puzzle (solved if any, else highest token acc).")
    p.add_argument("--auto_limit", type=int, default=200, help="Max puzzles to consider for --auto.")
    p.add_argument("--steps", type=int, default=16, help="How many refinement steps to visualize.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between steps (seconds).")
    p.add_argument("--clear", action="store_true", help="Clear terminal between frames.")
    p.add_argument("--no_color", action="store_true")
    p.add_argument("--min_clues", type=int, default=None)
    p.add_argument("--regen_dataset", action="store_true")
    return p.parse_args()


def load_cfg(config_path: str | None):
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(config_path or (model_dir / "Config.yml")),
    )
    return cfg


def build_model(cfg: OmegaConf) -> TRM:
    m = cfg.model
    return TRM(
        vocab_size=int(m.vocab_size),
        context_length=int(m.context_length),
        d_model=int(m.embedding_size),
        tiny_layers=int(m.tiny_layers),
        variant=str(m.variant),
        num_heads=int(m.num_heads),
        d_ff=int(m.feed_forward_size),
        mixer_hidden=int(m.mixer_hidden),
        dropout_rate=float(m.dropout_rate),
        activation=str(m.activation),
        add_positional_embedding=bool(getattr(m, "add_positional_embedding", True)),
        L_cycles=int(m.recursion.L_cycles),
        H_cycles=int(m.recursion.H_cycles),
        max_supervision_steps=int(m.recursion.max_supervision_steps),
        enable_early_stop=bool(m.recursion.enable_early_stop),
        halt_threshold_logit=float(m.recursion.halt_threshold_logit),
        aug_enabled=bool(m.augmentation.enabled),
        aug_num_embeddings=int(m.augmentation.num_embeddings),
        aug_default_id=int(m.augmentation.default_id),
    )


def load_params(ckpt_path: str):
    params_np, step = load_ckpt(ckpt_path)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params_np)
    if isinstance(params, dict):
        params = flax_core.freeze(params)
    return params, step


def render_overlay(
    puzzle: np.ndarray,
    pred: np.ndarray,
    solution: np.ndarray,
    *,
    prev_pred: Optional[np.ndarray] = None,
    use_color: bool = True,
) -> str:
    puzzle = puzzle.reshape(SIDE, SIDE)
    pred = pred.reshape(SIDE, SIDE)
    solution = solution.reshape(SIDE, SIDE)
    prev = None if prev_pred is None else prev_pred.reshape(SIDE, SIDE)

    def style(text: str, codes: str) -> str:
        if not use_color:
            return text
        return f"{codes}{text}{RESET}"

    lines = []
    for r in range(SIDE):
        row_chunks = []
        for c in range(SIDE):
            is_clue = int(puzzle[r, c]) != 0
            sol = int(solution[r, c])
            v = int(puzzle[r, c]) if is_clue else int(pred[r, c])
            ch = "." if v == 0 else str(v)

            codes = ""
            if is_clue:
                codes = BOLD + FG_GRAY
            else:
                if v == 0:
                    codes = DIM + FG_GRAY
                elif v == sol:
                    codes = FG_GREEN
                else:
                    codes = FG_RED
                if prev is not None and int(prev[r, c]) != v:
                    codes = UNDER + codes
            row_chunks.append(style(ch, codes) if use_color else ch)

            if c in (2, 5):
                row_chunks.append(style("|", FG_CYAN) if use_color else "|")

        lines.append(" ".join(row_chunks))
        if r in (2, 5):
            lines.append(style("-" * 21, FG_CYAN) if use_color else ("-" * 21))
    return "\n".join(lines)


def make_jitted_refine_step(model: TRM):
    @jax.jit
    def refine(params, x, y, z):
        y, z, logits, q_logit, pred = model.apply(
            {"params": params},
            x,
            y,
            z,
            deterministic=True,
            method=model.step_from_x,
        )
        return y, z, logits, q_logit, pred

    return refine


def make_jitted_infer_fixed(model: TRM, steps: int):
    steps = int(steps)

    @jax.jit
    def infer_fixed(params, tokens, aug_ids):
        x = model.apply(
            {"params": params},
            tokens,
            deterministic=True,
            aug_ids=aug_ids,
            method=model.encode,
        )
        y, z = model.apply({"params": params}, tokens, method=model.initial_state)

        def body(carry, _):
            y, z = carry
            y, z, _logits, q_logit, pred = model.apply(
                {"params": params},
                x,
                y,
                z,
                deterministic=True,
                method=model.step_from_x,
            )
            return (y, z), (pred, q_logit)

        (_y, _z), (preds, qs) = jax.lax.scan(body, (y, z), xs=None, length=steps)
        return preds[-1], qs[-1]

    return infer_fixed


def pick_best_puzzle(
    *,
    params,
    model: TRM,
    puzzles: np.ndarray,
    solutions: np.ndarray,
    limit: int,
    steps: int,
    batch_size: int = 32,
) -> int:
    limit = min(int(limit), puzzles.shape[0])
    infer_fixed = make_jitted_infer_fixed(model, steps)

    best_idx = 0
    best_solved = -1
    best_acc = -1.0

    for start in range(0, limit, batch_size):
        end = min(limit, start + batch_size)
        tokens = jnp.asarray(puzzles[start:end], dtype=jnp.int32)
        aug = jnp.zeros((end - start,), dtype=jnp.int32)
        pred, _q = infer_fixed(params, tokens, aug)
        pred_np = np.array(pred, dtype=np.int32)
        sol_np = solutions[start:end]

        solved = np.all(pred_np == sol_np, axis=1)
        tok_acc = (pred_np == sol_np).mean(axis=1)

        for i in range(end - start):
            s = int(solved[i])
            a = float(tok_acc[i])
            if s > best_solved or (s == best_solved and a > best_acc):
                best_solved = s
                best_acc = a
                best_idx = start + i

    return best_idx


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    model = build_model(cfg)

    base_root = Path(cfg.paths.data_root) if "paths" in cfg and cfg.paths.get("data_root") else Path.cwd()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = (base_root / ckpt_dir).resolve()
        ckpt_path = latest_ckpt(str(ckpt_dir))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    else:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (base_root / ckpt_path).resolve()
        ckpt_path = str(ckpt_path)

    params, step = load_params(ckpt_path)
    print(f"[ckpt] {ckpt_path} (step {step})")

    sudoku_cfg = cfg.sudoku
    min_clues = int(args.min_clues or sudoku_cfg.min_clues)
    cache_path = Path(str(sudoku_cfg.cache_path))
    if not cache_path.is_absolute():
        cache_path = (base_root / cache_path).resolve()
    ds = load_or_generate_dataset(
        cache_path=str(cache_path),
        train_samples=int(sudoku_cfg.train_samples),
        val_samples=int(sudoku_cfg.val_samples),
        min_clues=min_clues,
        seed=int(cfg.training.seed),
        regen=bool(args.regen_dataset),
        max_remove_attempts=250,
    )

    if args.split == "train":
        puzzles = ds.train_puzzle
        solutions = ds.train_solution
    else:
        puzzles = ds.val_puzzle
        solutions = ds.val_solution

    if args.auto:
        idx = pick_best_puzzle(
            params=params,
            model=model,
            puzzles=puzzles,
            solutions=solutions,
            limit=args.auto_limit,
            steps=int(args.steps),
        )
    else:
        idx = int(args.index or 0)
        if idx < 0 or idx >= puzzles.shape[0]:
            raise ValueError(f"index out of range: {idx} (split size {puzzles.shape[0]})")

    puzzle = puzzles[idx].astype(np.int32)
    solution = solutions[idx].astype(np.int32)
    clue_count = int(np.count_nonzero(puzzle))

    print(f"[puzzle] split={args.split} idx={idx} clues={clue_count}")
    print(format_grid(puzzle))

    tokens = jnp.asarray(puzzle[None, :], dtype=jnp.int32)
    aug = jnp.zeros((1,), dtype=jnp.int32)
    x = model.apply({"params": params}, tokens, deterministic=True, aug_ids=aug, method=model.encode)
    y, z = model.apply({"params": params}, tokens, method=model.initial_state)

    refine = make_jitted_refine_step(model)

    prev_pred = None
    for s in range(int(args.steps)):
        if args.clear:
            os.system("clear")

        y, z, _logits, q_logit, pred = refine(params, x, y, z)
        pred_np = np.array(pred[0], dtype=np.int32)
        q = float(np.array(q_logit[0], dtype=np.float32))
        tok_acc = float((pred_np == solution).mean())
        solved = bool(np.all(pred_np == solution))
        correct = int((pred_np == solution).sum())

        header = (
            f"step {s+1:02d}/{int(args.steps)}  "
            f"tok_acc={tok_acc*100:5.1f}% ({correct}/81)  "
            f"solved={solved}  "
            f"q_logit={q:+.3f}  q_prob={sigmoid(q):.3f}"
        )
        print(header)
        print(render_overlay(puzzle, pred_np, solution, prev_pred=prev_pred, use_color=not args.no_color))

        prev_pred = pred_np
        if args.sleep > 0:
            time.sleep(float(args.sleep))

    print("\n[solution]")
    print(format_grid(solution))


if __name__ == "__main__":
    main()
