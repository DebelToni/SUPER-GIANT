from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from GiantGPT import GiantGPT
from Training_step import train_step
from arrow_data_loader import (
    ShardedArrowDataset,
    StageDataLoader,
    load_dataloader_state,
    save_dataloader_state,
)
from checkpoint_manager import latest as latest_ckpt
from checkpoint_manager import load as load_ckpt
from checkpoint_manager import save as save_ckpt
from checkpoint_manager import save_opt_state, load_opt_state
from optimizer_utils import create_weight_decay_mask
from flax import core as flax_core
from flax import serialization

jax.config.update("jax_default_matmul_precision", "tensorfloat32")


_stop_requested = False


def _signal_handler(signum, frame):
    """Mark that we should checkpoint+exit after finishing the current step."""
    global _stop_requested
    if _stop_requested:
        return
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(f"\n[signal] Caught {name}; will checkpoint and exit after this step...", flush=True)
    _stop_requested = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@dataclass
class StageConfig:
    name: str
    dataset: str
    seq_len: int
    epochs: int
    end_ratio: float
    shuffle: bool = True
    fraction: float = 1.0


@dataclass
class StageRuntime:
    config: StageConfig
    loader: StageDataLoader
    total_steps: int


def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    global_cfg = OmegaConf.load(project_root / "Global_Config.yml")
    local_cfg = OmegaConf.load(model_dir / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str(base_prefix / path)

    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)

    for key in ("processed_data_root", "dataloader_state_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "answers_arrow" in cfg.qa_finetune and cfg.qa_finetune.answers_arrow is not None:
        resolved = resolve_path(cfg.qa_finetune.answers_arrow)
        if resolved is not None:
            cfg.qa_finetune.answers_arrow = resolved

    if "checkpoint_dir" in cfg.qa_finetune and cfg.qa_finetune.checkpoint_dir is not None:
        resolved = resolve_path(cfg.qa_finetune.checkpoint_dir)
        if resolved is not None:
            cfg.qa_finetune.checkpoint_dir = resolved

    return cfg


def load_tokenizer(cfg: OmegaConf):
    from transformers import AutoTokenizer

    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def parse_stage_configs(cfg: OmegaConf) -> List[StageConfig]:
    stages_raw = OmegaConf.to_container(cfg.stages, resolve=True)
    stage_cfgs = []
    for stage in stages_raw:
        stage_cfgs.append(
            StageConfig(
                name=stage["name"],
                dataset=stage["dataset"],
                seq_len=int(stage["seq_len"]),
                epochs=int(stage["epochs"]),
                end_ratio=float(stage["end_ratio"]),
                shuffle=bool(stage.get("shuffle", True)),
                fraction=float(stage.get("fraction", 1.0)),
            )
        )
    return stage_cfgs


def build_stage_runtimes(
    stage_cfgs: List[StageConfig],
    *,
    dataset_root: Path,
    batch_size: int,
    seed: int,
    pad_token_id: int,
) -> List[StageRuntime]:
    dataset_cache: Dict[Path, ShardedArrowDataset] = {}
    runtimes: List[StageRuntime] = []
    for stage in stage_cfgs:
        data_path = dataset_root / stage.dataset
        dataset = dataset_cache.get(data_path)
        if dataset is None:
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset directory '{data_path}' missing. Run the data pipeline to generate shards."
                )
            print(f"[loader] loading shards from {data_path}")
            dataset = ShardedArrowDataset(data_path)
            dataset_cache[data_path] = dataset
        fraction = max(0.0, min(stage.fraction, 1.0))
        target_rows = int(dataset.total_rows * fraction)
        if target_rows < batch_size:
            raise ValueError(
                f"Stage {stage.name} fraction too small: {target_rows} rows for batch_size {batch_size}"
            )
        loader = StageDataLoader(
            dataset,
            batch_size=batch_size,
            seq_len=stage.seq_len,
            shuffle=stage.shuffle,
            seed=seed,
            pad_token_id=pad_token_id,
            max_rows=target_rows,
        )
        print(
            f"[loader] stage={stage.name} rows={target_rows}/{dataset.total_rows} "
            f"ctx={stage.seq_len} steps_per_epoch={loader.steps_per_epoch}"
        )
        total_steps = stage.epochs * loader.steps_per_epoch
        if total_steps == 0:
            raise ValueError(f"Stage {stage.name} has zero training steps. Adjust dataset or batch size.")
        runtimes.append(StageRuntime(config=stage, loader=loader, total_steps=total_steps))
    return runtimes


def validate_milestones(stage_cfgs: List[StageConfig], cfg: OmegaConf) -> None:
    milestones = list(cfg.training_defaults.lr_milestones)
    expected = [stage.end_ratio for stage in stage_cfgs]
    if not expected or expected[-1] != 1.0:
        raise ValueError("Stage configuration must end with end_ratio == 1.0")
    if sorted(expected) != expected:
        raise ValueError("Stage end_ratio values must be non-decreasing")
    if milestones and milestones[-1] != 1.0:
        milestones.append(1.0)
    if milestones and len(milestones) != len(stage_cfgs):
        print("⚠ lr_milestones count does not match number of stages; proceeding regardless.")
    else:
        for m, e in zip(milestones, expected):
            if abs(m - e) > 1e-3:
                print("⚠ lr milestone", m, "differs from stage end_ratio", e)


def build_optimizer(cfg: OmegaConf, total_steps: int, params) -> optax.GradientTransformation:
    warmup_steps = int(cfg.optimizer.warmup_steps)
    if total_steps <= warmup_steps:
        raise ValueError("Total steps must exceed warmup steps for cosine decay")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimizer.base_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=cfg.optimizer.min_learning_rate,
    )
    exclusions = cfg.optimizer.get("weight_decay_exclusions", [])
    mask = create_weight_decay_mask(params, exclusions) if exclusions else None

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.gradient_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=cfg.optimizer.weight_decay,
            mask=mask,
        ),
    )
    return optimizer


def dataloader_state_path(cfg: OmegaConf, step: int) -> Path:
    root = Path(cfg.paths.dataloader_state_root)
    return root / f"state_{step:07d}.json"


def save_training_state(
    *,
    cfg: OmegaConf,
    params,
    opt_state,
    checkpoint_dir: str,
    global_step: int,
    stage_idx: int,
    completed_in_stage: int,
    stage_states: Dict[str, Dict[str, int]],
    runtime: StageRuntime,
):
    """Persist params, optimizer, and dataloader progress atomically."""
    ckpt_file = save_ckpt(params, global_step, checkpoint_dir)
    save_opt_state(opt_state, global_step, checkpoint_dir)
    stage_states[runtime.config.name] = runtime.loader.state_dict()
    save_dataloader_state(
        dataloader_state_path(cfg, global_step),
        {
            "stage_index": stage_idx,
            "stage_step_total": completed_in_stage,
            "stage_states": stage_states,
        },
    )
    return ckpt_file


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("SUPER-GIANT training")
    cli.add_argument("--checkpoint_dir", default="checkpoints")
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    tokenizer = load_tokenizer(cfg)

    stage_cfgs = parse_stage_configs(cfg)
    validate_milestones(stage_cfgs, cfg)

    base_root = Path(cfg.paths.data_root)

    dataset_root = Path(cfg.paths.processed_data_root)
    if not dataset_root.is_absolute():
        dataset_root = (base_root / dataset_root).resolve()
    batch_size = int(cfg.training.batch_size)
    seed = int(cfg.training.seed)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )
    stage_runtimes = build_stage_runtimes(
        stage_cfgs,
        dataset_root=dataset_root,
        batch_size=batch_size,
        seed=seed,
        pad_token_id=pad_token_id,
    )
    total_steps = sum(stage.total_steps for stage in stage_runtimes)

    max_seq_len = max(stage.config.seq_len for stage in stage_runtimes)
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
    )

    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32))["params"]
    from flax import core as flax_core
    if isinstance(params, dict):
        params = flax_core.freeze(params)

    optimizer = build_optimizer(cfg, total_steps, params)
    opt_state = optimizer.init(params)
    global_step = 0

    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (base_root / checkpoint_path).resolve()
    checkpoint_dir = str(checkpoint_path)
    checkpoint_every = args.checkpoint_every or cfg.training.checkpoint_every

    resume_request = args.resume
    if resume_request and resume_request != "latest":
        resume_path = Path(resume_request)
        if not resume_path.is_absolute():
            resume_request = str((base_root / resume_path).resolve())
    else:
        resume_request = args.resume

    # Resume if requested
    if resume_request:
        if resume_request == "latest":
            ckpt_path = latest_ckpt(checkpoint_dir)
            if ckpt_path is None:
                raise FileNotFoundError("No checkpoints available to resume from.")
        else:
            ckpt_path = resume_request
        params, global_step = load_ckpt(ckpt_path)
        if isinstance(params, dict):
            params = flax_core.freeze(params)
        opt_state = optimizer.init(params)
        resume_dir = str(Path(ckpt_path).parent)
        opt_bytes = load_opt_state(global_step, resume_dir)
        if opt_bytes is not None:
            try:
                opt_state = serialization.from_bytes(opt_state, opt_bytes)
                print(f"▶ Resumed optimizer state from {resume_dir}")
            except Exception as exc:
                print(f"⚠ Failed to restore optimizer state ({exc}); reinitializing.")
        else:
            print("⚠ No optimizer state found; proceeding with fresh AdamW buffers.")
        print(f"▶ Resumed parameters from {ckpt_path} at step {global_step}")

    # Restore dataloader state if available
    loader_state = load_dataloader_state(dataloader_state_path(cfg, global_step)) if global_step else None
    stage_states: Dict[str, Dict[str, int]] = {}
    current_stage_idx = 0
    stage_step_total = 0
    if loader_state:
        current_stage_idx = int(loader_state.get("stage_index", 0))
        stage_step_total = int(loader_state.get("stage_step_total", 0))
        stage_states = loader_state.get("stage_states", {})
        print(f"▶ Restored dataloader state at stage {current_stage_idx} step {stage_step_total}")

    for idx, runtime in enumerate(stage_runtimes):
        state_dict = stage_states.get(runtime.config.name, {"epoch": 0, "step_in_epoch": 0})
        runtime.loader.load_state(state_dict)

    base_rng = jax.random.PRNGKey(seed)

    start = time.time()
    for stage_idx in range(current_stage_idx, len(stage_runtimes)):
        runtime = stage_runtimes[stage_idx]
        stage_steps_target = runtime.total_steps
        completed_in_stage = stage_step_total if stage_idx == current_stage_idx else 0

        print(
            f"→ Stage {runtime.config.name}: seq_len={runtime.config.seq_len} epochs={runtime.config.epochs} "
            f"steps={stage_steps_target} (resume at {completed_in_stage})"
        )

        pbar = tqdm(
            total=stage_steps_target,
            desc=f"stage:{runtime.config.name}",
            initial=completed_in_stage,
            leave=True,
            dynamic_ncols=True,
        )
        last_loss = None

        while completed_in_stage < stage_steps_target:
            batch = next(runtime.loader)
            dropout_rng = jax.random.fold_in(base_rng, global_step)
            params, opt_state, loss = train_step(
                params,
                opt_state,
                batch,
                model=model,
                optimizer=optimizer,
                dropout_rng=dropout_rng,
            )
            global_step += 1
            completed_in_stage += 1
            last_loss = float(loss)
            pbar.update(1)

            if global_step % cfg.training.log_every == 0:
                elapsed = time.time() - start
                ppl = float(np.exp(loss)) if loss < 20 else float("inf")
                print(
                    f"step {global_step:>7}/{total_steps:<7} | stage {runtime.config.name:<18} "
                    f"loss {loss:.4f} ppl {ppl:.2f} ({elapsed:.1f}s)"
                )
                start = time.time()

            if global_step % checkpoint_every == 0:
                ckpt_file = save_training_state(
                    cfg=cfg,
                    params=params,
                    opt_state=opt_state,
                    checkpoint_dir=checkpoint_dir,
                    global_step=global_step,
                    stage_idx=stage_idx,
                    completed_in_stage=completed_in_stage,
                    stage_states=stage_states,
                    runtime=runtime,
                )
                print(f"💾 checkpoint → {ckpt_file}")

            if _stop_requested:
                ckpt_file = save_training_state(
                    cfg=cfg,
                    params=params,
                    opt_state=opt_state,
                    checkpoint_dir=checkpoint_dir,
                    global_step=global_step,
                    stage_idx=stage_idx,
                    completed_in_stage=completed_in_stage,
                    stage_states=stage_states,
                    runtime=runtime,
                )
                print(f"💾 checkpoint (signal) → {ckpt_file}")
                print("[signal] Checkpoint saved after signal; exiting early.")
                pbar.close()
                return

            stage_states[runtime.config.name] = runtime.loader.state_dict()

        pbar.close()
        if last_loss is not None:
            print(
                f"✓ Stage {runtime.config.name} completed (last loss {last_loss:.4f})"
            )
        else:
            print(f"✓ Stage {runtime.config.name} completed (no batches emitted)")
        # Stage finished → reset step tracker
        stage_step_total = 0

    final_ckpt = save_ckpt(params, global_step, checkpoint_dir)
    save_opt_state(opt_state, global_step, checkpoint_dir)
    save_dataloader_state(
        dataloader_state_path(cfg, global_step),
        {
            "stage_index": len(stage_runtimes) - 1,
            "stage_step_total": stage_runtimes[-1].total_steps,
            "stage_states": stage_states,
        },
    )
    print(f"✔ Training complete. Final checkpoint: {final_ckpt}")
    print("→ Use this checkpoint as --init_checkpoint for the QA finetune stage.")


if __name__ == "__main__":
    main()
