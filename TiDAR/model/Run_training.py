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
from flax import core as flax_core
from flax import serialization
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from TiDAR.model.GiantTiDAR import TiDAR
from TiDAR.model.Training_step import train_step
from TiDAR.model.tidar_utils import build_train_batch
from TiDAR.model.tokenizer_utils import (
    ensure_tidar_mask_token,
    init_mask_embedding_row,
    resize_embedding_params,
)
from GIANT.v2.model.arrow_data_loader import (
    ShardedArrowDataset,
    StageDataLoader,
    load_dataloader_state,
    save_dataloader_state,
)
from GIANT.v2.model.checkpoint_manager import (
    AsyncMiniCheckpointManager,
    latest as latest_ckpt,
    load as load_ckpt,
    save as save_ckpt,
    save_opt_state,
    load_opt_state,
    set_npz_metadata,
)
from GIANT.v2.model.optimizer_utils import create_weight_decay_mask


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


IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())


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
        return str((base_prefix / path).resolve())

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
        print("WARN: lr_milestones count does not match number of stages; proceeding regardless.")


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
    return optax.apply_if_finite(optimizer, max_consecutive_errors=5)


def dataloader_state_path(cfg: OmegaConf, step: int) -> Path:
    root = Path(cfg.paths.dataloader_state_root)
    return root / f"state_{step:07d}.json"


def _prefetch_to_device(iterator, size: int = 2):
    if size <= 0:
        for batch in iterator:
            yield batch
        return

    it = iter(iterator)
    buf = []
    try:
        for _ in range(size):
            buf.append(jax.device_put(next(it)))
    except StopIteration:
        buf.clear()

    while buf:
        batch = buf.pop(0)
        yield batch
        try:
            buf.append(jax.device_put(next(it)))
        except StopIteration:
            buf.clear()


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("TiDAR training")
    cli.add_argument("--checkpoint_dir", default=None)
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    cli.add_argument("--init_checkpoint", default=None)
    cli.add_argument("--max_steps", type=int, default=None)
    cli.add_argument(
        "--scan_chunk",
        type=int,
        default=None,
        help="Number of steps to fuse with lax.scan inside a single compiled call.",
    )
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
    model = TiDAR(
        vocab_size=len(tokenizer),
        context_length=max_seq_len * 2,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
    )

    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.zeros((batch_size, max_seq_len * 2), dtype=jnp.int32))[
        "params"
    ]
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

    training_cfg = cfg.training
    mini_every = int(getattr(training_cfg, "mini_checkpoint_every", max(1, checkpoint_every // 10)))
    mini_max_to_keep = int(getattr(training_cfg, "mini_max_to_keep", 3))
    mini_ckpt_dir = Path(checkpoint_dir) / "mini"
    mini_ckpt_mgr = AsyncMiniCheckpointManager(
        ckpt_dir=mini_ckpt_dir,
        max_to_keep=mini_max_to_keep,
    )

    stage_states: Dict[str, Dict[str, int]] = {
        runtime.config.name: runtime.loader.state_dict() for runtime in stage_runtimes
    }
    current_stage_idx = 0
    stage_step_total = 0

    resume_request = args.resume
    mini_state_template = {
        "params": params,
        "opt_state": opt_state,
        "global_step": global_step,
        "stage_index": current_stage_idx,
        "stage_step_total": stage_step_total,
        "stage_states": stage_states,
    }
    resumed_from_mini = False

    if resume_request == "latest":
        restored_state, restored_step = mini_ckpt_mgr.restore_latest(mini_state_template)
        if restored_step:
            params = restored_state["params"]
            opt_state = restored_state["opt_state"]
            global_step = int(restored_state.get("global_step", restored_step))
            current_stage_idx = int(restored_state.get("stage_index", 0))
            stage_step_total = int(restored_state.get("stage_step_total", 0))
            stage_states = restored_state.get("stage_states", {})
            resumed_from_mini = True
            print(f"↩ Resumed from mini checkpoint at step {restored_step}")
        else:
            resume_request = "latest_full"
    elif resume_request and resume_request != "latest":
        resume_path = Path(resume_request)
        if not resume_path.is_absolute():
            resume_request = str((base_root / resume_path).resolve())

    if resume_request and not resumed_from_mini:
        if resume_request == "latest_full":
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

    loader_state = None
    if not resumed_from_mini:
        loader_state = load_dataloader_state(dataloader_state_path(cfg, global_step)) if global_step else None
        if loader_state:
            current_stage_idx = int(loader_state.get("stage_index", 0))
            stage_step_total = int(loader_state.get("stage_step_total", 0))
            stage_states = loader_state.get("stage_states", {})
            print(f"▶ Restored dataloader state at stage {current_stage_idx} step {stage_step_total}")

    for idx, runtime in enumerate(stage_runtimes):
        state_dict = stage_states.get(runtime.config.name, {"epoch": 0, "step_in_epoch": 0})
        runtime.loader.load_state(state_dict)

    base_rng = jax.random.PRNGKey(seed)
    cfg_chunk = int(getattr(cfg.training, "scan_chunk", 1))
    chunk_size = max(1, int(args.scan_chunk)) if args.scan_chunk is not None else max(1, cfg_chunk)
    grad_accum = max(1, int(getattr(cfg.training, "gradient_accumulation", 1)))

    def _init_accum_grads(pytree):
        return jax.tree_util.tree_map(jnp.zeros_like, pytree)

    def _stack_batches(batches):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batches)

    @jax.jit
    def _run_chunk(params, opt_state, batch_chunk, start_step, accum_grads, accum_count):
        grad_scale = jnp.asarray(1.0 / grad_accum, dtype=jnp.float32)

        def body(carry, batch):
            params, opt_state, step, accum_grads, accum_count = carry
            dropout_rng = jax.random.fold_in(base_rng, step)

            def loss_fn(p):
                logits = model.apply(
                    {"params": p},
                    batch["input_ids"],
                    rngs={"dropout": dropout_rng},
                    deterministic=False,
                    attn_bias=batch["attn_bias"],
                    position_ids=batch["position_ids"],
                )
                labels = batch["labels"]
                mask_ntp = batch["loss_mask_ntp"]
                mask_diff = batch["loss_mask_diff"]
                active = (mask_ntp + mask_diff) > 0
                labels_safe = jnp.where(active, labels, 0)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_safe)
                ntp_loss = (ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
                diff_loss = (ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)
                return ntp_loss + diff_loss, (ntp_loss, diff_loss)

            (loss, (ntp_loss, diff_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            accum_grads = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
            accum_count = accum_count + 1

            def apply_updates(args):
                params, opt_state, accum_grads = args
                grads = jax.tree_util.tree_map(lambda g: g * grad_scale, accum_grads)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                accum_grads = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
                return params, opt_state, accum_grads

            should_update = accum_count == grad_accum
            params, opt_state, accum_grads = jax.lax.cond(
                should_update,
                apply_updates,
                lambda args: args,
                (params, opt_state, accum_grads),
            )
            accum_count = jnp.where(should_update, 0, accum_count)
            return (params, opt_state, step + 1, accum_grads, accum_count), (loss, ntp_loss, diff_loss)

        (params, opt_state, _, accum_grads, accum_count), losses = jax.lax.scan(
            body, (params, opt_state, start_step, accum_grads, accum_count), batch_chunk
        )
        return params, opt_state, accum_grads, accum_count, losses

    @jax.jit
    def _apply_accum(params, opt_state, accum_grads, denom):
        grads = jax.tree_util.tree_map(lambda g: g / denom, accum_grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    start = time.time()
    accum_grads = _init_accum_grads(params)
    accum_count = jnp.asarray(0, dtype=jnp.int32)
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

        batch_iter = _prefetch_to_device(runtime.loader, size=2 if IS_GPU else 0)
        while completed_in_stage < stage_steps_target:
            batch_list = []
            for _ in range(chunk_size):
                try:
                    batch_list.append(next(batch_iter))
                except StopIteration:
                    break
            if not batch_list:
                break

            prepared_batches = []
            for batch in batch_list:
                batch_tokens = jnp.asarray(batch["input"])
                mask = jnp.asarray(batch["mask"])
                lengths = jnp.clip((batch_tokens != pad_token_id).sum(axis=1), 1, runtime.config.seq_len)
                train_batch = build_train_batch(
                    batch_tokens,
                    lengths,
                    mask_id=mask_token_id,
                    block_len=draft_len,
                    bias_value=bias_value,
                    token_mask=mask,
                )
                prepared_batches.append({k: jnp.asarray(v) for k, v in train_batch.items()})

            chunk = _stack_batches(prepared_batches)
            params, opt_state, accum_grads, accum_count, losses = _run_chunk(
                params, opt_state, chunk, global_step, accum_grads, accum_count
            )
            losses = np.asarray(jax.device_get(losses))
            if losses.ndim == 1:
                losses = losses[:, None]
            chunk_len = losses.shape[0]
            if chunk_len == 0:
                break
            chunk_start = global_step + 1
            global_step += chunk_len
            completed_in_stage += chunk_len
            pbar.update(chunk_len)
            stage_states[runtime.config.name] = runtime.loader.state_dict()

            for offset in range(chunk_len):
                loss_val = losses[offset, 0]
                ntp_val = losses[offset, 1] if losses.shape[1] > 1 else loss_val
                diff_val = losses[offset, 2] if losses.shape[1] > 2 else loss_val
                last_loss = float(loss_val)
                step_val = chunk_start + offset
                if step_val % cfg.training.log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"step {step_val:>7}/{total_steps:<7} | stage {runtime.config.name:<18} "
                        f"loss {float(loss_val):.4f} ntp {float(ntp_val):.4f} diff {float(diff_val):.4f} "
                        f"({elapsed:.1f}s)"
                    )
                    start = time.time()

            if mini_every and (global_step % mini_every == 0):
                mini_state = {
                    "params": params,
                    "opt_state": opt_state,
                    "global_step": global_step,
                    "stage_index": stage_idx,
                    "stage_step_total": completed_in_stage,
                    "stage_states": stage_states,
                }
                mini_ckpt_mgr.save(global_step, mini_state)

            if global_step % checkpoint_every == 0:
                ckpt_file = save_ckpt(params, global_step, checkpoint_dir, train_loss=last_loss)
                if last_loss is not None:
                    print(f"[metadata] Wrote train_loss={last_loss:.6f} to checkpoint {ckpt_file}")
                save_opt_state(opt_state, global_step, checkpoint_dir)
                save_dataloader_state(
                    dataloader_state_path(cfg, global_step),
                    {
                        "stage_index": stage_idx,
                        "stage_step_total": completed_in_stage,
                        "stage_states": stage_states,
                    },
                )
                print(f"💾 checkpoint → {ckpt_file}")

            if _stop_requested:
                accum_count_host = int(jax.device_get(accum_count))
                if accum_count_host > 0:
                    denom = jnp.asarray(accum_count_host, dtype=jnp.float32)
                    params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
                    accum_grads = _init_accum_grads(params)
                    accum_count = jnp.asarray(0, dtype=jnp.int32)
                mini_ckpt_mgr.wait_until_finished(timeout=4.0)
                print("[signal] Stop requested; exiting after current chunk.")
                pbar.close()
                return

        pbar.close()
        accum_count_host = int(jax.device_get(accum_count))
        if accum_count_host > 0:
            denom = jnp.asarray(accum_count_host, dtype=jnp.float32)
            params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
            accum_grads = _init_accum_grads(params)
            accum_count = jnp.asarray(0, dtype=jnp.int32)
        if last_loss is not None:
            print(f"✓ Stage {runtime.config.name} completed (last loss {last_loss:.4f})")
        else:
            print(f"✓ Stage {runtime.config.name} completed (no batches emitted)")
        stage_step_total = 0

    accum_count_host = int(jax.device_get(accum_count))
    if accum_count_host > 0:
        denom = jnp.asarray(accum_count_host, dtype=jnp.float32)
        params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
        accum_grads = _init_accum_grads(params)
        accum_count = jnp.asarray(0, dtype=jnp.int32)
    final_ckpt = save_ckpt(params, global_step, checkpoint_dir, train_loss=last_loss)
    if last_loss is not None:
        print(f"[metadata] Wrote train_loss={last_loss:.6f} to checkpoint {final_ckpt}")
    save_opt_state(opt_state, global_step, checkpoint_dir)
    save_dataloader_state(
        dataloader_state_path(cfg, global_step),
        {
            "stage_index": len(stage_runtimes) - 1,
            "stage_step_total": stage_runtimes[-1].total_steps,
            "stage_states": stage_states,
        },
    )
    mini_ckpt_mgr.wait_until_finished(timeout=10.0)
    print(f"✔ Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
