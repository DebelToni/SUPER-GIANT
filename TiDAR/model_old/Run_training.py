from __future__ import annotations

import sys

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import core as flax_core
from flax import serialization
from flax import traverse_util
from flax.core import freeze
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.GiantGPT import GiantGPT
from model.Training_step import train_step
from model.arrow_data_loader import (
    ShardedArrowDataset,
    StageDataLoader,
    load_dataloader_state,
    save_dataloader_state,
)
from model.checkpoint_manager import latest as latest_ckpt
from model.checkpoint_manager import load as load_ckpt
from model.checkpoint_manager import save as save_ckpt
from model.checkpoint_manager import save_opt_state, load_opt_state
from model.tokenizer_utils import (
    ensure_tidar_mask_token,
    resize_embedding_params,
    init_mask_embedding_row,
)
from model.tidar_utils import build_train_batch


_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        return
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(f"\n[signal] Caught {name}; will exit after the current chunk...", flush=True)
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
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(model_dir / "Config.yml"),
    )

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

    for key in ("processed_data_root", "dataloader_state_root", "logs_root", "checkpoints_root", "hf_cache_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "tokenizer" in cfg:
        cache_dir = cfg.tokenizer.get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cfg.tokenizer.cache_dir = str(Path(cfg.paths.data_root) / cache_path)
        custom_path = cfg.tokenizer.get("custom_path")
        if custom_path:
            custom_path = Path(str(custom_path))
            if not custom_path.is_absolute():
                cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path)

    return cfg


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


def create_weight_decay_mask(params, exclusions: Iterable[str]):
    """Return a mask tree where True applies weight decay."""
    exclusion_set = {str(name).lower() for name in exclusions}
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {}
    for path in flat_params:
        last = path[-1].lower()
        flat_mask[path] = last not in exclusion_set
    return freeze(traverse_util.unflatten_dict(flat_mask))


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


def ensure_mask_id(tokenizer, cfg: OmegaConf) -> int:
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added <= 0:
        print(f"[mask] using existing token '{mask_token}' (id={mask_id})")
    else:
        print(f"[mask] added token '{mask_token}' (id={mask_id})")
    return mask_id


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
        print("[warn] lr_milestones count does not match number of stages; proceeding regardless.")


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


def build_optimizer(cfg: OmegaConf, total_steps: int, params) -> optax.GradientTransformation:
    warmup_steps = int(cfg.optimizer.warmup_steps)
    if total_steps <= warmup_steps + 1:
        schedule = optax.constant_schedule(cfg.optimizer.base_learning_rate)
    else:
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


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("TiDAR training")
    cli.add_argument("--checkpoint_dir", default=None)
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    cli.add_argument("--init_checkpoint", default=None)
    cli.add_argument("--max_steps", type=int, default=None)
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
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = ensure_mask_id(tokenizer, cfg)
    draft_len = int(cfg.tidar.draft_length)
    bias_value = float(cfg.tidar.attn_bias_value)

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
    params = model.init(
        rng,
        jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
        deterministic=True,
    )["params"]
    if isinstance(params, dict):
        params = flax_core.freeze(params)

    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = (base_root / ckpt_path).resolve()
        print(f"[init] loading init checkpoint from {ckpt_path}")
        params = flax_core.freeze(load_ckpt(str(ckpt_path))[0])
        rng, resize_key = jax.random.split(rng)
        params, added = resize_embedding_params(params, len(tokenizer), key=resize_key)
        if added:
            print(f"[init] expanded embeddings by {added} rows for TiDAR mask token")
    else:
        rng, mask_key = jax.random.split(rng)
        params = init_mask_embedding_row(params, mask_token_id, key=mask_key)

    optimizer = build_optimizer(cfg, total_steps, params)
    opt_state = optimizer.init(params)
    global_step = 0

    checkpoint_dir = args.checkpoint_dir or cfg.paths.checkpoints_root
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (base_root / checkpoint_path).resolve()
    checkpoint_dir = str(checkpoint_path)
    checkpoint_every = args.checkpoint_every or cfg.training.checkpoint_every

    stage_states: Dict[str, Dict[str, int]] = {
        runtime.config.name: runtime.loader.state_dict() for runtime in stage_runtimes
    }
    current_stage_idx = 0
    stage_step_total = 0

    resume_request = args.resume
    if resume_request is not None:
        ckpt_path = None
        if resume_request == "latest":
            ckpt_path = latest_ckpt(checkpoint_dir)
        else:
            ckpt_path = resume_request
        if ckpt_path is None:
            print("[resume] no checkpoint found; starting fresh")
        else:
            params, global_step = load_ckpt(ckpt_path)
            rng, resize_key = jax.random.split(rng)
            params, added = resize_embedding_params(params, len(tokenizer), key=resize_key)
            opt_bytes = load_opt_state(global_step, checkpoint_dir)
            if added:
                print(f"[resume] expanded embeddings by {added} rows; resetting optimizer state")
                opt_state = optimizer.init(params)
            elif opt_bytes is not None:
                opt_state = serialization.from_bytes(opt_state, opt_bytes)
            state_path = dataloader_state_path(cfg, global_step)
            saved_state = load_dataloader_state(state_path)
            if saved_state is not None:
                current_stage_idx = int(saved_state.get("stage_index", 0))
                stage_step_total = int(saved_state.get("stage_step_total", 0))
                stage_states = saved_state.get("stage_states", stage_states)
                for runtime in stage_runtimes:
                    if runtime.config.name in stage_states:
                        runtime.loader.load_state(stage_states[runtime.config.name])
            print(f"[resume] restored step {global_step} from {ckpt_path}")

    max_steps = args.max_steps
    pbar = tqdm(total=total_steps, initial=global_step, desc="training")

    for stage_idx, runtime in enumerate(stage_runtimes):
        if stage_idx < current_stage_idx:
            continue
        for _ in range(runtime.total_steps):
            if _stop_requested:
                break
            batch = next(runtime.loader)
            batch_tokens = jnp.asarray(batch["input"])
            mask = jnp.asarray(batch["mask"])
            lengths = jnp.clip(mask.sum(axis=1).astype(jnp.int32) + 1, 1, runtime.config.seq_len)

            train_batch = build_train_batch(
                batch_tokens,
                lengths,
                mask_id=mask_token_id,
                block_len=draft_len,
                bias_value=bias_value,
            )
            train_batch = {k: jnp.asarray(v) for k, v in train_batch.items()}

            rng, dropout_rng = jax.random.split(rng)
            params, opt_state, loss, ntp_loss, diff_loss = train_step(
                params,
                opt_state,
                train_batch,
                model=model,
                optimizer=optimizer,
                dropout_rng=dropout_rng,
            )

            global_step += 1
            stage_step_total += 1
            pbar.update(1)
            if global_step % cfg.training.log_every == 0:
                pbar.set_postfix(
                    {
                        "loss": float(loss),
                        "ntp": float(ntp_loss),
                        "diff": float(diff_loss),
                    }
                )

            if global_step % checkpoint_every == 0:
                ckpt_file = save_ckpt(params, global_step, checkpoint_dir)
                save_opt_state(opt_state, global_step, checkpoint_dir)
                stage_states[runtime.config.name] = runtime.loader.state_dict()
                save_dataloader_state(
                    dataloader_state_path(cfg, global_step),
                    {
                        "stage_index": stage_idx,
                        "stage_step_total": stage_step_total,
                        "stage_states": stage_states,
                    },
                )
                print(f"[ckpt] saved {ckpt_file}")

            if max_steps is not None and global_step >= max_steps:
                break
        if _stop_requested or (max_steps is not None and global_step >= max_steps):
            break

    pbar.close()
    if global_step % checkpoint_every != 0:
        ckpt_file = save_ckpt(params, global_step, checkpoint_dir)
        save_opt_state(opt_state, global_step, checkpoint_dir)
        stage_states[runtime.config.name] = runtime.loader.state_dict()
        save_dataloader_state(
            dataloader_state_path(cfg, global_step),
            {
                "stage_index": stage_idx,
                "stage_step_total": stage_step_total,
                "stage_states": stage_states,
            },
        )
        print(f"[ckpt] saved {ckpt_file}")

    print("[done] training finished")


if __name__ == "__main__":
    main()
