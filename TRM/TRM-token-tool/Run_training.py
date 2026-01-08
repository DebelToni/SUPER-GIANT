from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax import config as jax_config
import numpy as np
import optax
from flax import core as flax_core
from flax import serialization
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from config_utils import load_config
from tokenizer_utils import build_custom_tokenizer, load_tokenizer


from v2.smol.GiantGPT import GiantGPT
from v2.model.arrow_data_loader import (
    ShardedArrowDataset,
    StageDataLoader,
    load_dataloader_state,
    save_dataloader_state,
)
from v2.model.async_mini_checkpoint import AsyncMiniCheckpointManager
from v2.smol.checkpoint_io import load_npz
from v2.smol.checkpoint_manager import latest as latest_ckpt
from v2.smol.checkpoint_manager import load as load_ckpt
from v2.smol.checkpoint_manager import save as save_ckpt
from v2.smol.checkpoint_manager import save_opt_state, load_opt_state
from v2.model.optimizer_utils import create_weight_decay_mask


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


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


def _apply_compute_dtype_override(cfg: OmegaConf) -> None:
    desired = None
    if "model" in cfg and "compute_dtype" in cfg.model:
        desired = cfg.model.compute_dtype
    if not desired:
        return
    dtype = _to_dtype(str(desired))
    try:
        import v2.smol.GiantGPT as smol_gpt_module
        import v2.smol.Transformer_block as smol_block_module
    except Exception as exc:
        print(f"[dtype] Failed to import smol modules for override: {exc}")
        return

    smol_gpt_module.COMPUTE_DTYPE = dtype
    smol_block_module.COMPUTE_DTYPE = dtype
    try:
        smol_gpt_module.MODEL_CFG.compute_dtype = str(desired)
        smol_block_module.MODEL_CFG.compute_dtype = str(desired)
    except Exception:
        pass
    jax_config.update("jax_default_matmul_precision", str(desired))
    print(f"[dtype] Overriding SmolLM compute dtype to {desired}")


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("TRM SmolLM training")
    cli.add_argument("--checkpoint_dir", default=None, help="Checkpoint dir (default: cfg.paths.checkpoint_root).")
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    cli.add_argument("--init_checkpoint", default=None, help="Init weights (default: cfg.training.init_checkpoint).")
    cli.add_argument("--max_steps", type=int, default=None, help="Stop after N optimizer steps.")
    cli.add_argument(
        "--scan_chunk",
        type=int,
        default=None,
        help="Number of steps to fuse with lax.scan inside a single compiled call.",
    )
    return cli.parse_args()


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
        print("Warning: lr_milestones count does not match number of stages; proceeding regardless.")
    else:
        for m, e in zip(milestones, expected):
            if abs(m - e) > 1e-3:
                print(f"Warning: lr milestone {m} differs from stage end_ratio {e}")


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
                    f"Dataset directory '{data_path}' missing. Run build_sudoku_trm_dataset.py."
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
            raise ValueError(f"Stage {stage.name} has zero training steps.")
        runtimes.append(StageRuntime(config=stage, loader=loader, total_steps=total_steps))
    return runtimes


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


def _resize_embedding(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    new = np.array(target, copy=True)
    src = np.asarray(source)
    rows = min(new.shape[0], src.shape[0])
    new[:rows] = src[:rows]
    return new


def _merge_params(init_params, loaded_params) -> flax_core.FrozenDict:
    merged = flax_core.unfreeze(init_params)
    replaced = 0
    resized = 0

    def assign(dst, src, path=()):
        nonlocal replaced, resized
        for key, val in src.items():
            if key not in dst:
                continue
            if isinstance(val, dict):
                if isinstance(dst[key], dict):
                    assign(dst[key], val, path + (key,))
                continue
            dst_val = dst[key]
            if hasattr(dst_val, "shape") and hasattr(val, "shape"):
                if dst_val.shape == val.shape:
                    dst[key] = val
                    replaced += 1
                elif path + (key,) == ("Embed_0", "embedding"):
                    dst[key] = _resize_embedding(dst_val, val)
                    resized += 1

    assign(merged, loaded_params)
    merged = flax_core.freeze(merged)
    merged = jax.tree_util.tree_map(lambda x: jnp.asarray(x), merged)
    if resized:
        print(f"[init] resized {resized} embedding tensors for new vocab")
    print(f"[init] loaded {replaced} tensors from init checkpoint")
    return merged


def _build_l2_reference(params, ckpt_path: str):
    loaded = load_npz(ckpt_path)
    base_params = _merge_params(params, loaded)

    def build_mask(path, value):
        if path == ("Embed_0", "embedding"):
            base_vocab = loaded.get("Embed_0", {}).get("embedding")
            if base_vocab is None:
                base_rows = value.shape[0]
            else:
                base_rows = np.asarray(base_vocab).shape[0]
            mask = np.ones(value.shape, dtype=np.float32)
            if base_rows < value.shape[0]:
                mask[base_rows:] = 0.0
            return jnp.asarray(mask)
        return jnp.ones_like(value, dtype=jnp.float32)

    flat = flax_core.unfreeze(params)
    mask = jax.tree_util.tree_map_with_path(build_mask, flat)
    mask = flax_core.freeze(mask)
    base_params = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
    mask = jax.tree_util.tree_map(jax.lax.stop_gradient, mask)
    return base_params, mask


def _l2_distance(params, base_params, mask) -> jnp.ndarray:
    def total_fn(p, b, m):
        diff = (p - b) * m
        return jnp.sum(diff * diff)

    def denom_fn(_, __, m):
        return jnp.sum(m)

    total = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(total_fn, params, base_params, mask)))
    denom = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(denom_fn, params, base_params, mask)))
    return total / (denom + 1e-8)


def train_step_with_l2(params, opt_state, batch, *, model, optimizer, dropout_rng, base_params, mask, l2_coeff):
    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        if l2_coeff > 0.0 and base_params is not None and mask is not None:
            loss = loss + l2_coeff * _l2_distance(p, base_params, mask)
        return loss

    (loss, grads) = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


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


def main() -> None:
    args = parse_args()
    cfg = load_config()

    build_custom_tokenizer(force=False)
    tokenizer = load_tokenizer()
    _apply_compute_dtype_override(cfg)

    stage_cfgs = parse_stage_configs(cfg)
    validate_milestones(stage_cfgs, cfg)

    base_root = Path(cfg.paths.data_root)
    dataset_root = Path(cfg.paths.processed_data_root)
    batch_size = int(cfg.training.batch_size)
    seed = int(cfg.training.seed)

    checkpoint_root = Path(args.checkpoint_dir or cfg.paths.get("checkpoint_root", "checkpoints"))
    if not checkpoint_root.is_absolute():
        checkpoint_root = (base_root / checkpoint_root).resolve()
    checkpoint_dir = str(checkpoint_root)

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
    if isinstance(params, dict):
        params = flax_core.freeze(params)

    global_step = 0
    resume_request = args.resume
    init_checkpoint = args.init_checkpoint or cfg.training.get("init_checkpoint")
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
        print(f"[resume] Loaded params from {ckpt_path} at step {global_step}")
    elif init_checkpoint:
        if not Path(init_checkpoint).exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")
        loaded = load_npz(init_checkpoint)
        params = _merge_params(params, loaded)

    optimizer = build_optimizer(cfg, total_steps, params)
    opt_state = optimizer.init(params)

    l2_coeff = float(getattr(cfg.training, "l2_base_coeff", 0.0))
    base_params = None
    base_mask = None
    if l2_coeff > 0.0:
        ref_ckpt = getattr(cfg.training, "l2_base_checkpoint", None) or init_checkpoint
        if ref_ckpt is None:
            raise ValueError("l2_base_coeff is set but no l2_base_checkpoint/init_checkpoint provided.")
        if not Path(ref_ckpt).exists():
            raise FileNotFoundError(f"L2 base checkpoint not found: {ref_ckpt}")
        base_params, base_mask = _build_l2_reference(params, ref_ckpt)
        print(f"[l2] Using base reference from {ref_ckpt} (coeff={l2_coeff})")

    checkpoint_every = args.checkpoint_every or cfg.training.checkpoint_every

    mini_every = int(getattr(cfg.training, "mini_checkpoint_every", max(1, checkpoint_every // 10)))
    mini_max_to_keep = int(getattr(cfg.training, "mini_max_to_keep", 3))
    mini_ckpt_dir = checkpoint_root / "mini"
    mini_ckpt_mgr = AsyncMiniCheckpointManager(
        ckpt_dir=mini_ckpt_dir,
        max_to_keep=mini_max_to_keep,
    )

    stage_states: Dict[str, Dict[str, int]] = {
        runtime.config.name: runtime.loader.state_dict() for runtime in stage_runtimes
    }
    current_stage_idx = 0
    stage_step_total = 0

    if resume_request:
        opt_bytes = load_opt_state(global_step, checkpoint_dir)
        if opt_bytes is not None:
            try:
                opt_state = serialization.from_bytes(opt_state, opt_bytes)
                print(f"[resume] Restored optimizer state from {checkpoint_dir}")
            except Exception as exc:
                print(f"[resume] Failed to restore optimizer state ({exc}); reinitializing.")

    loader_state = load_dataloader_state(dataloader_state_path(cfg, global_step)) if global_step else None
    if loader_state:
        current_stage_idx = int(loader_state.get("stage_index", 0))
        stage_step_total = int(loader_state.get("stage_step_total", 0))
        stage_states = loader_state.get("stage_states", {})
        print(f"[resume] Restored dataloader state at stage {current_stage_idx} step {stage_step_total}")

    for runtime in stage_runtimes:
        state_dict = stage_states.get(runtime.config.name, {"epoch": 0, "step_in_epoch": 0})
        runtime.loader.load_state(state_dict)

    base_rng = jax.random.PRNGKey(seed)
    cfg_chunk = int(getattr(cfg.training, "scan_chunk", 1))
    chunk_size = max(1, int(args.scan_chunk)) if args.scan_chunk is not None else max(1, cfg_chunk)

    def _stack_batches(batches):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batches)

    @jax.jit
    def _run_chunk(params, opt_state, batch_chunk, start_step):
        def body(carry, batch):
            params, opt_state, step = carry
            dropout_rng = jax.random.fold_in(base_rng, step)
            params, opt_state, loss = train_step_with_l2(
                params,
                opt_state,
                batch,
                model=model,
                optimizer=optimizer,
                dropout_rng=dropout_rng,
                base_params=base_params,
                mask=base_mask,
                l2_coeff=l2_coeff,
            )
            return (params, opt_state, step + 1), loss

        (params, opt_state, _), losses = jax.lax.scan(
            body, (params, opt_state, start_step), batch_chunk
        )
        return params, opt_state, losses

    max_steps = args.max_steps
    stop_training = False
    start = time.time()
    for stage_idx in range(current_stage_idx, len(stage_runtimes)):
        runtime = stage_runtimes[stage_idx]
        stage_steps_target = runtime.total_steps
        completed_in_stage = stage_step_total if stage_idx == current_stage_idx else 0

        print(
            f"Stage {runtime.config.name}: seq_len={runtime.config.seq_len} epochs={runtime.config.epochs} "
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

        prefetch_size = int(getattr(cfg.training, "prefetch", 2 if IS_GPU else 0))
        batch_iter = _prefetch_to_device(
            runtime.loader, size=prefetch_size
        )
        while completed_in_stage < stage_steps_target:
            if _stop_requested:
                stop_training = True
                break
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break
            batch_list = []
            for _ in range(chunk_size):
                try:
                    batch_list.append(next(batch_iter))
                except StopIteration:
                    break
            if not batch_list:
                break

            chunk = _stack_batches(batch_list)
            params, opt_state, losses = _run_chunk(params, opt_state, chunk, global_step)
            losses = np.asarray(jax.device_get(losses))

            for loss_val in losses:
                global_step += 1
                completed_in_stage += 1
                last_loss = float(loss_val)
                pbar.update(1)

                stage_states[runtime.config.name] = runtime.loader.state_dict()

                if global_step % cfg.training.log_every == 0:
                    elapsed = time.time() - start
                    ppl = float(np.exp(loss_val)) if loss_val < 20 else float("inf")
                    print(
                        f"step {global_step:>7}/{total_steps:<7} | stage {runtime.config.name:<18} "
                        f"loss {loss_val:.4f} ppl {ppl:.2f} ({elapsed:.1f}s)"
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
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ckpt_file = save_ckpt(params, global_step, checkpoint_dir)
                    save_opt_state(opt_state, global_step, checkpoint_dir)
                    save_dataloader_state(
                        dataloader_state_path(cfg, global_step),
                        {
                            "stage_index": stage_idx,
                            "stage_step_total": completed_in_stage,
                            "stage_states": stage_states,
                        },
                    )
                    print(f"[ckpt] saved {ckpt_file}")

                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break

            if stop_training:
                break

        pbar.close()
        if last_loss is not None:
            print(f"Stage {runtime.config.name} completed. Final loss {last_loss:.4f}")
        if stop_training:
            break

    mini_ckpt_mgr.wait_until_finished()
    if stop_training:
        print("Training stopped early.")


if __name__ == "__main__":
    main()
