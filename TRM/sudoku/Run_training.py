from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import core as flax_core
from flax import serialization
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRM import TRM
from Training_step import eval_step, train_step
from common.checkpoint_manager import latest as latest_ckpt
from common.checkpoint_manager import load as load_ckpt
from common.checkpoint_manager import load_opt_state, save as save_ckpt, save_opt_state
from common.optimizer_utils import create_weight_decay_mask
from sudoku_dataset import format_grid, load_or_generate_dataset


IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())


def load_configs(config_path: str | None = None) -> OmegaConf:
    sudoku_dir = Path(__file__).resolve().parent
    global_cfg = OmegaConf.load(PROJECT_ROOT / "Global_Config.yml")
    model_cfg = OmegaConf.load(PROJECT_ROOT / "model" / "Config.yml")
    local_cfg = OmegaConf.load(config_path or (sudoku_dir / "Config.yml"))
    return OmegaConf.merge(global_cfg, model_cfg, local_cfg)


def build_model(cfg: OmegaConf) -> TRM:
    m = cfg.model
    return TRM(
        vocab_size=int(m.vocab_size),
        context_length=int(m.context_length),
        d_model=int(m.embedding_size),
        tiny_layers=int(m.tiny_layers),
        variant=str(m.variant),
        num_heads=int(m.num_heads),
        rope_dim=int(m.rope_dim),
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


def build_optimizer(cfg: OmegaConf, total_steps: int, params) -> optax.GradientTransformation:
    warmup_steps = int(cfg.optimizer.warmup_steps)
    base_lr = float(cfg.optimizer.base_learning_rate)
    min_lr = float(cfg.optimizer.min_learning_rate)

    if total_steps > warmup_steps and warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=min_lr,
        )
    else:
        schedule = base_lr

    exclusions = cfg.optimizer.get("weight_decay_exclusions", [])
    mask = create_weight_decay_mask(params, exclusions) if exclusions else None

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.optimizer.gradient_clip_norm)),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=float(cfg.optimizer.weight_decay),
            mask=mask,
        ),
    )
    return optimizer


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("TRM Sudoku training")
    cli.add_argument("--config", default=None, help="Override config path (defaults to sudoku/Config.yml).")
    cli.add_argument("--data_root", default=None, help="Override cfg.paths.data_root (dataset/cache/checkpoints).")
    cli.add_argument("--checkpoint_dir", default="checkpoints/trm_sudoku")
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    cli.add_argument("--regen_dataset", action="store_true", help="Regenerate cached Sudoku dataset.")
    cli.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps.")
    cli.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size.")
    cli.add_argument("--microbatch_size", type=int, default=None, help="Microbatch size for gradient accumulation.")
    cli.add_argument("--supervision_steps", type=int, default=None, help="Override training.supervision_steps.")
    cli.add_argument("--train_samples", type=int, default=None)
    cli.add_argument("--val_samples", type=int, default=None)
    cli.add_argument("--min_clues", type=int, default=None)
    return cli.parse_args()


def _make_batch(rng: np.random.Generator, puzzles: np.ndarray, solutions: np.ndarray, batch_size: int):
    idx = rng.integers(0, puzzles.shape[0], size=(batch_size,))
    return {
        "puzzle": jnp.asarray(puzzles[idx], dtype=jnp.int32),
        "solution": jnp.asarray(solutions[idx], dtype=jnp.int32),
        "aug_ids": jnp.zeros((batch_size,), dtype=jnp.int32),
    }


def _eval_dataset(
    params, ds, *, model: TRM, batch_size: int, supervision_steps: int
) -> tuple[float, float, float]:
    puzzles = ds["puzzle"]
    solutions = ds["solution"]
    n = puzzles.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    total_solved = 0.0
    total_token = 0.0
    total_ce = 0.0

    for i in range(n_batches):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        batch = {
            "puzzle": jnp.asarray(puzzles[sl], dtype=jnp.int32),
            "solution": jnp.asarray(solutions[sl], dtype=jnp.int32),
            "aug_ids": jnp.zeros((puzzles[sl].shape[0],), dtype=jnp.int32),
        }
        solved_acc, token_acc, ce = eval_step(params, batch, model=model, supervision_steps=supervision_steps)
        total_solved += float(solved_acc)
        total_token += float(token_acc)
        total_ce += float(ce)

    return total_solved / n_batches, total_token / n_batches, total_ce / n_batches


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config)

    if args.data_root is not None:
        base_root = Path(str(args.data_root)).resolve()
        if "paths" not in cfg:
            cfg.paths = OmegaConf.create({})
        cfg.paths.data_root = str(base_root)
    else:
        base_root = Path(cfg.paths.data_root) if "paths" in cfg and cfg.paths.get("data_root") else Path.cwd()

    seed = int(cfg.training.seed)
    rng_np = np.random.default_rng(seed)
    rng = jax.random.PRNGKey(seed)

    sudoku_cfg = cfg.sudoku
    train_samples = int(args.train_samples or sudoku_cfg.train_samples)
    val_samples = int(args.val_samples or sudoku_cfg.val_samples)
    min_clues = int(args.min_clues or sudoku_cfg.min_clues)
    cache_path = Path(str(sudoku_cfg.cache_path))
    if not cache_path.is_absolute():
        cache_path = (base_root / cache_path).resolve()

    ds = load_or_generate_dataset(
        cache_path=str(cache_path),
        train_samples=train_samples,
        val_samples=val_samples,
        min_clues=min_clues,
        seed=seed,
        regen=bool(args.regen_dataset),
        max_remove_attempts=250,
    )

    print("[dataset] train:", ds.train_puzzle.shape, "val:", ds.val_puzzle.shape)
    print("[dataset] example puzzle (0=blank):\n" + format_grid(ds.train_puzzle[0]))
    print("[dataset] example solution:\n" + format_grid(ds.train_solution[0]))

    model = build_model(cfg)

    batch_size = int(args.batch_size or cfg.training.batch_size)
    microbatch_size = args.microbatch_size
    if microbatch_size is None:
        microbatch_size = cfg.training.get("microbatch_size", None)
    microbatch_size = None if microbatch_size is None else int(microbatch_size)
    if microbatch_size is not None:
        if microbatch_size <= 0:
            raise ValueError("--microbatch_size must be > 0")
        if batch_size % microbatch_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by microbatch_size ({microbatch_size})")

    supervision_steps = int(args.supervision_steps or cfg.training.supervision_steps)
    max_steps = int(args.max_steps or cfg.training.max_steps)
    total_steps = max_steps

    ckpt_path = Path(args.checkpoint_dir)
    if not ckpt_path.is_absolute():
        ckpt_path = (base_root / ckpt_path).resolve()
    ckpt_dir = str(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_every = int(args.checkpoint_every or cfg.training.checkpoint_every)

    init_tokens = jnp.zeros((batch_size, model.context_length), dtype=jnp.int32)
    init_aug = jnp.zeros((batch_size,), dtype=jnp.int32)
    rng, key_params, key_dropout = jax.random.split(rng, 3)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        init_tokens,
        deterministic=False,
        aug_ids=init_aug,
    )
    params = variables["params"]
    if isinstance(params, dict):
        params = flax_core.freeze(params)

    optimizer = build_optimizer(cfg, total_steps, params)
    opt_state = optimizer.init(params)

    global_step = 0
    if args.resume:
        if args.resume == "latest":
            ckpt_path = latest_ckpt(ckpt_dir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = (base_root / resume_path).resolve()
            ckpt_path = str(resume_path)

        loaded_params, global_step = load_ckpt(ckpt_path)
        loaded_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), loaded_params)
        if isinstance(loaded_params, dict):
            loaded_params = flax_core.freeze(loaded_params)
        params = loaded_params

        opt_state = optimizer.init(params)
        opt_bytes = load_opt_state(global_step, ckpt_dir)
        if opt_bytes is not None:
            opt_state = serialization.from_bytes(opt_state, opt_bytes)
        print(f"↩ Resumed from {ckpt_path} (step {global_step})")

    log_every = int(cfg.training.log_every)
    eval_every = int(cfg.training.eval_every)

    train_arrays = {"puzzle": ds.train_puzzle, "solution": ds.train_solution}
    val_arrays = {"puzzle": ds.val_puzzle, "solution": ds.val_solution}

    pbar = tqdm(range(global_step, max_steps), desc="train", dynamic_ncols=True)
    t_compile = None

    for step in pbar:
        batch = _make_batch(rng_np, train_arrays["puzzle"], train_arrays["solution"], batch_size)
        rng, key = jax.random.split(rng)

        t0 = time.time()
        params, opt_state, (loss, ce, halt, solved_acc, token_acc) = train_step(
            params,
            opt_state,
            batch,
            model=model,
            optimizer=optimizer,
            dropout_rng=key,
            supervision_steps=supervision_steps,
            microbatch_size=microbatch_size,
        )
        jax.block_until_ready(loss)
        t1 = time.time()

        if t_compile is None:
            t_compile = t1 - t0

        if (step + 1) % log_every == 0 or step == global_step:
            pbar.set_postfix(
                loss=float(loss),
                ce=float(ce),
                halt=float(halt),
                solved=float(solved_acc),
                tok=float(token_acc),
                step_s=f"{(t1 - t0):.3f}",
            )

        if (step + 1) % eval_every == 0:
            val_solved, val_tok, val_ce = _eval_dataset(
                params, val_arrays, model=model, batch_size=batch_size, supervision_steps=supervision_steps
            )
            pbar.write(
                f"[eval] step={step+1} val_solved={val_solved:.3f} val_tok={val_tok:.3f} val_ce={val_ce:.3f}"
            )

        if (step + 1) % checkpoint_every == 0:
            ckpt_path = save_ckpt(params, step + 1, ckpt_dir)
            save_opt_state(opt_state, step + 1, ckpt_dir)
            pbar.write(f"[ckpt] saved {ckpt_path}")

    ckpt_path = save_ckpt(params, max_steps, ckpt_dir)
    save_opt_state(opt_state, max_steps, ckpt_dir)
    print(f"[ckpt] saved final {ckpt_path}")

    if t_compile is not None:
        print(f"[timing] first step (compile+run): {t_compile:.3f}s")


if __name__ == "__main__":
    main()
