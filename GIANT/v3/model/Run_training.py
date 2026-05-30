from __future__ import annotations

import argparse
import copy
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.model_mode import model_mode_is_causal, resolve_model_mode
from GIANT.v3.model.Training_step import loss_and_grad
from GIANT.v3.model.arrow_data_loader import (
    ShardedArrowDataset,
    StageDataLoader,
    load_dataloader_state,
    save_dataloader_state,
)
from GIANT.v3.model.checkpoint_manager import (
    AsyncMiniCheckpointManager,
    latest as latest_ckpt,
    load as load_ckpt,
    save as save_ckpt,
    save_opt_state,
    load_opt_state,
    set_npz_metadata,
)
from GIANT.v3.model.optimizer_utils import create_weight_decay_mask
from GIANT.v3.run_manifest import build_manifest, write_manifest
from GIANT.v3.device_utils import select_default_device
from flax import core as flax_core
from flax import jax_utils as flax_jax_utils
from flax import serialization


_stop_requested = False


def _signal_handler(signum, frame):
    """Mark that we should exit after finishing the current in-flight chunk."""
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


DEFAULT_DEVICE = select_default_device()
LOCAL_DEVICES = tuple(d for d in jax.local_devices() if d.platform == DEFAULT_DEVICE.platform)
if not LOCAL_DEVICES:
    LOCAL_DEVICES = (DEFAULT_DEVICE,)
LOCAL_DEVICE_COUNT = len(LOCAL_DEVICES)


def _configure_nccl_p2p():
    if DEFAULT_DEVICE.platform != "gpu" or LOCAL_DEVICE_COUNT <= 1:
        return None
    if "NCCL_P2P_DISABLE" in os.environ:
        return None

    mode = os.environ.get("GIANT_NCCL_P2P_MODE", "auto").strip().lower()
    if mode in {"0", "false", "off", "enable", "enabled"}:
        return None

    device_kinds = {str(getattr(device, "device_kind", "")) for device in LOCAL_DEVICES}
    should_disable = mode in {"1", "true", "on", "disable", "disabled"}
    if not should_disable and mode == "auto":
        should_disable = bool(device_kinds) and all("A40" in kind for kind in device_kinds)

    if not should_disable:
        return None

    os.environ["NCCL_P2P_DISABLE"] = "1"
    if mode == "auto":
        return "auto-disabled NCCL P2P for multi-GPU A40 setup"
    return "disabled NCCL P2P via GIANT_NCCL_P2P_MODE"


NCCL_P2P_NOTE = _configure_nccl_p2p()
MULTI_GPU_BACKEND = "single"
if LOCAL_DEVICE_COUNT > 1:
    MULTI_GPU_BACKEND = os.environ.get("GIANT_MULTI_GPU_BACKEND", "pmap").strip().lower()
    if MULTI_GPU_BACKEND not in {"shard_map", "pmap"}:
        raise ValueError(
            f"Unsupported GIANT_MULTI_GPU_BACKEND={MULTI_GPU_BACKEND!r}; expected 'shard_map' or 'pmap'"
        )
USE_SHARD_MAP = LOCAL_DEVICE_COUNT > 1 and MULTI_GPU_BACKEND == "shard_map"
USE_PMAP = LOCAL_DEVICE_COUNT > 1 and MULTI_GPU_BACKEND == "pmap"
TRAINING_MESH = Mesh(np.array(LOCAL_DEVICES), ("data",)) if USE_SHARD_MAP else None
REPLICATED_SHARDING = NamedSharding(TRAINING_MESH, P()) if USE_SHARD_MAP else None
IS_GPU = DEFAULT_DEVICE.platform == "gpu"
DEFAULT_ARTIFACT_LOCAL_ROOT = Path("/proj/giant-data")
DEFAULT_ARTIFACT_S3_ROOT = "s3://giant-data"


def _to_dtype(value: jnp.dtype | str) -> jnp.dtype:
    if isinstance(value, jnp.dtype):
        return value
    if isinstance(value, str):
        try:
            return getattr(jnp, value)
        except AttributeError:
            return jnp.dtype(value)
    return jnp.dtype(value)


@dataclass
class StageConfig:
    name: str
    dataset: str
    seq_len: int
    epochs: int
    shuffle: bool = True
    fraction: float = 1.0


@dataclass
class StageRuntime:
    config: StageConfig
    loader: StageDataLoader
    total_steps: int


def load_configs(
    config_path: str | None = None,
    global_config_path: str | None = None,
) -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    
    # Resolve config paths
    if config_path is None:
        config_path = str(model_dir / "Config.yml")
    if global_config_path is None:
        global_config_path = str(project_root / "Global_Config.yml")
    
    # Load configs, with fallback if global config doesn't exist
    local_cfg = OmegaConf.load(config_path)
    if Path(global_config_path).exists():
        global_cfg = OmegaConf.load(global_config_path)
        cfg = OmegaConf.merge(global_cfg, local_cfg)
    else:
        cfg = local_cfg

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

    for key in ("processed_data_root", "dataloader_state_root", "logs_root", "checkpoints_root"):
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


def resolve_warmup_updates(
    cfg: OmegaConf,
    stage_runtimes: List[StageRuntime],
    *,
    global_batch_size: int,
    grad_accum: int,
) -> tuple[int, Optional[int], Optional[int]]:
    warmup_tokens_raw = cfg.optimizer.get("warmup_tokens", None)
    if warmup_tokens_raw is not None:
        warmup_tokens = int(warmup_tokens_raw)
        if warmup_tokens < 0:
            raise ValueError("optimizer warmup_tokens must be >= 0")
        if warmup_tokens == 0:
            return 0, warmup_tokens, 0

        tokens_seen = 0
        micro_steps = 0
        warmup_updates = 0
        for runtime in stage_runtimes:
            tokens_per_microstep = int(global_batch_size * runtime.config.seq_len)
            for _ in range(runtime.total_steps):
                micro_steps += 1
                tokens_seen += tokens_per_microstep
                if micro_steps % grad_accum == 0:
                    warmup_updates += 1
                    if tokens_seen >= warmup_tokens:
                        return warmup_updates, warmup_tokens, tokens_seen

        if micro_steps % grad_accum != 0:
            warmup_updates += 1
        return warmup_updates, warmup_tokens, tokens_seen

    warmup_raw = cfg.optimizer.get("warmup_updates", None)
    if warmup_raw is None:
        warmup_raw = cfg.optimizer.get("warmup_steps", None)
    if warmup_raw is None:
        raise ValueError("optimizer warmup_tokens or warmup_updates or warmup_steps must be set")
    return int(warmup_raw), None, None


def build_optimizer(
    cfg: OmegaConf,
    total_update_steps: int,
    warmup_updates: int,
    params,
) -> optax.GradientTransformation:
    if total_update_steps <= warmup_updates:
        raise ValueError("Total optimizer updates must exceed warmup updates for cosine decay")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimizer.base_learning_rate,
        warmup_steps=warmup_updates,
        decay_steps=total_update_steps,
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
    params_dir: str,
    training_states_dir: str,
    global_step: int,
    stage_idx: int,
    completed_in_stage: int,
    stage_states: Dict[str, Dict[str, int]],
    runtime: StageRuntime,
    train_loss: float | None = None,
    loader_state: Optional[Dict[str, int]] = None,
):
    """Persist params, optimizer, and dataloader progress atomically."""
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(training_states_dir, exist_ok=True)
    host_params = _checkpoint_tree(params)
    host_opt_state = _checkpoint_tree(opt_state)
    ckpt_file = save_ckpt(host_params, global_step, params_dir, train_loss=train_loss)
    if train_loss is not None:
        print(f"[metadata] Wrote train_loss={train_loss:.6f} to checkpoint {ckpt_file}")
    save_opt_state(host_opt_state, global_step, training_states_dir)
    stage_states[runtime.config.name] = (
        copy.deepcopy(loader_state) if loader_state is not None else runtime.loader.state_dict()
    )
    save_dataloader_state(
        dataloader_state_path(cfg, global_step),
        {
            "stage_index": stage_idx,
            "stage_step_total": completed_in_stage,
            "stage_states": stage_states,
        },
    )
    return ckpt_file


def _freeze_if_dict(tree):
    return flax_core.freeze(tree) if isinstance(tree, dict) else tree


def _place_on_training_devices(tree):
    if USE_SHARD_MAP:
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, REPLICATED_SHARDING),
            tree,
        )
    if USE_PMAP:
        return flax_jax_utils.replicate(tree, devices=LOCAL_DEVICES)
    return jax.device_put(tree, DEFAULT_DEVICE)


def _checkpoint_tree(tree):
    if USE_PMAP:
        return jax.device_get(flax_jax_utils.unreplicate(tree))
    return jax.device_get(tree)


def _replicated_scalar_to_int(value) -> int:
    if USE_PMAP:
        return int(np.asarray(jax.device_get(flax_jax_utils.unreplicate(value))))
    host_value = np.asarray(jax.device_get(value))
    if host_value.shape == ():
        return int(host_value)
    return int(host_value.reshape(-1)[0])


def _training_scalar(value, dtype):
    scalar = jnp.asarray(value, dtype=dtype)
    if USE_PMAP:
        return flax_jax_utils.replicate(scalar, devices=LOCAL_DEVICES)
    return _place_on_training_devices(scalar)


def _host_losses(losses) -> np.ndarray:
    if USE_PMAP:
        return np.asarray(jax.device_get(flax_jax_utils.unreplicate(losses)))
    return np.asarray(jax.device_get(losses))


def _reshape_batch_for_pmap(batch, *, per_device_batch: int):
    def reshape_leaf(x):
        arr = np.asarray(x)
        if arr.ndim == 0:
            raise ValueError("Batch leaves must include a batch axis for pmap")
        expected_global_batch = LOCAL_DEVICE_COUNT * per_device_batch
        if arr.shape[0] != expected_global_batch:
            raise ValueError(
                f"Expected global batch {expected_global_batch} for pmap input, got {arr.shape[0]}"
            )
        return arr.reshape((LOCAL_DEVICE_COUNT, per_device_batch, *arr.shape[1:]))

    return jax.tree_util.tree_map(reshape_leaf, batch)


def _batch_partition_for_leaf(ndim: int, *, chunked: bool):
    if not USE_SHARD_MAP:
        return None
    if ndim == 0:
        return P()
    if chunked:
        if ndim < 2:
            raise ValueError(f"Chunked batch leaf must have ndim >= 2, got ndim={ndim}")
        return P(None, "data", *([None] * (ndim - 2)))
    return P("data", *([None] * (ndim - 1)))


def _place_batch_on_training_devices(batch, *, chunked: bool):
    if not USE_SHARD_MAP:
        return batch

    def place_leaf(x):
        arr = jnp.asarray(x)
        spec = _batch_partition_for_leaf(arr.ndim, chunked=chunked)
        sharding = NamedSharding(TRAINING_MESH, spec)
        return jax.device_put(arr, sharding)

    return jax.tree_util.tree_map(place_leaf, batch)


def _prefetch_training_batches(iterator, size: int, *, per_device_batch: int, state_getter=None):
    def snapshot_state():
        return copy.deepcopy(state_getter()) if state_getter is not None else None

    def pack(batch, state):
        return (batch, state) if state_getter is not None else batch

    def source_iter():
        for batch in iterator:
            state = snapshot_state()
            if USE_PMAP:
                batch = _reshape_batch_for_pmap(batch, per_device_batch=per_device_batch)
            elif not USE_SHARD_MAP:
                batch = jax.device_put(batch, DEFAULT_DEVICE)
            yield pack(batch, state)

    source = source_iter()
    if size <= 0:
        yield from source
        return

    buf = []
    try:
        for _ in range(size):
            buf.append(next(source))
    except StopIteration:
        pass

    while buf:
        item = buf.pop(0)
        yield item
        try:
            buf.append(next(source))
        except StopIteration:
            pass


def _startup_marker(label: str):
    print(f"[startup] {label}", flush=True)


def _parse_int_candidates(raw: Optional[object], default: Iterable[int]) -> List[int]:
    values: List[int] = []
    if raw is None:
        values = [int(v) for v in default]
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        for part in parts:
            if not part:
                continue
            values.append(int(part))
    else:
        try:
            values = [int(v) for v in raw]  # type: ignore[arg-type]
        except TypeError:
            values = [int(raw)]

    cleaned: List[int] = []
    seen = set()
    for value in values:
        iv = int(value)
        if iv < 0:
            continue
        if iv in seen:
            continue
        seen.add(iv)
        cleaned.append(iv)
    return cleaned


def _persist_training_int(config_path: Optional[str], key: str, value: int) -> bool:
    if config_path is None:
        return False
    path = Path(config_path)
    if not path.exists():
        return False
    cfg_doc = OmegaConf.load(path)
    if cfg_doc.get("training") is None:
        cfg_doc.training = {}
    cfg_doc.training[key] = int(value)
    OmegaConf.save(cfg_doc, path)
    return True


def _persist_prefetch_size(config_path: Optional[str], value: int) -> bool:
    return _persist_training_int(config_path, "prefetch_size", value)


def _persist_scan_chunk(config_path: Optional[str], value: int) -> bool:
    return _persist_training_int(config_path, "scan_chunk", value)


def _format_wall_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_train_progress_log(
    *,
    step_val: int,
    total_steps: int,
    stage_name: str,
    stage_name_width: int,
    loss_val: float,
    elapsed: float,
) -> str:
    ppl = float(np.exp(loss_val)) if loss_val < 20 else float("inf")
    ppl_str = f"{ppl:6.0f}" if np.isfinite(ppl) else f"{'inf':>6}"
    return (
        f"step {step_val:>7}/{total_steps:<7} | stage {stage_name:<{stage_name_width}} "
        f"loss {loss_val:7.3f} ppl {ppl_str} ({elapsed:.1f}s)"
    )


def _artifact_cfg(cfg: OmegaConf) -> OmegaConf:
    return cfg.get("artifacts", OmegaConf.create({}))


def _artifact_upload_enabled(cfg: OmegaConf, kind: str) -> bool:
    upload = _artifact_cfg(cfg).get("upload", {})
    return bool(upload.get("enabled", False) and upload.get(kind, False))


def _s3_uri_for_local_path(local_path: Path, cfg: OmegaConf) -> str:
    artifacts = _artifact_cfg(cfg)
    local_root = Path(str(artifacts.get("local_root", str(DEFAULT_ARTIFACT_LOCAL_ROOT)))).expanduser().resolve()
    s3_root = str(artifacts.get("s3_root", DEFAULT_ARTIFACT_S3_ROOT)).rstrip("/")
    candidate = local_path.expanduser().resolve()
    try:
        relative = candidate.relative_to(local_root)
    except ValueError as exc:
        raise ValueError(
            f"Upload path {candidate} is not under {local_root}; cannot derive S3 destination."
        ) from exc
    return f"{s3_root}/{relative.as_posix()}"


def _build_upload_command(local_path: Path, s3_uri: str, cfg: OmegaConf) -> str:
    artifacts = _artifact_cfg(cfg)
    tool = str(artifacts.get("tool", "s5cmd"))
    size_only = bool(artifacts.get("size_only", True))
    env_file = artifacts.get("env_file", "~/.env-R2")
    sync_args = [tool, "sync"]
    if size_only:
        sync_args.append("--size-only")
    source = f"{local_path}/" if local_path.is_dir() else str(local_path)
    sync_args.extend([source, s3_uri])
    quoted = " ".join(shlex.quote(part) for part in sync_args)
    if env_file:
        return f"set -a; source {shlex.quote(str(Path(str(env_file)).expanduser()))}; set +a; {quoted}"
    return quoted


def _launch_async_upload(
    *,
    label: str,
    local_path: Path,
    log_dir: Path,
    upload_processes: List[tuple[str, subprocess.Popen[str], Path]],
    cfg: OmegaConf,
) -> None:
    local_path = local_path.absolute()
    if not local_path.exists():
        print(f"[upload] skipping {label}: path does not exist: {local_path}")
        return

    s3_uri = _s3_uri_for_local_path(local_path, cfg)
    if local_path.is_dir() and not s3_uri.endswith("/"):
        s3_uri = f"{s3_uri}/"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    log_path = log_dir / f"{label}_{stamp}.log"
    cmd = _build_upload_command(local_path, s3_uri, cfg)
    log_handle = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    upload_processes.append((label, proc, log_path))
    print(f"[upload] launched {label}: {local_path} -> {s3_uri} (pid={proc.pid})")


def _reap_async_uploads(upload_processes: List[tuple[str, subprocess.Popen[str], Path]]) -> None:
    active: List[tuple[str, subprocess.Popen[str], Path]] = []
    for label, proc, log_path in upload_processes:
        ret = proc.poll()
        if ret is None:
            active.append((label, proc, log_path))
            continue
        status = "finished" if ret == 0 else f"failed (exit {ret})"
        print(f"[upload] {label} {status}; log={log_path}")
        if proc.stdout is not None:
            proc.stdout.close()
    upload_processes[:] = active


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("SUPER-GIANT training")
    cli.add_argument("--config", default=None, help="Path to training config YAML file")
    cli.add_argument("--global_config", default=None, help="Path to global config YAML file")
    cli.add_argument("--checkpoint_dir", default="checkpoints")
    cli.add_argument("--checkpoint_every", type=int, default=None)
    cli.add_argument("--resume", nargs="?", const="latest", default=None)
    cli.add_argument("--init_checkpoint", type=str, default=None)
    cli.add_argument(
        "--scan_chunk",
        type=int,
        default=None,
        help="Number of steps to fuse with lax.scan inside a single compiled call (reduces host dispatch overhead). "
             "Defaults to training.scan_chunk in the config.",
    )
    cli.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Override training.prefetch_size (device batch prefetch depth).",
    )
    cli.add_argument(
        "--upload-on-checkpoint",
        action="store_true",
        help="Asynchronously sync the checkpoint root after each full checkpoint. Uses artifacts.s3_root from Global_Config.yml.",
    )
    return cli.parse_args()


def main() -> None:
    command_start = time.perf_counter()
    args = parse_args()
    cfg = load_configs(config_path=args.config, global_config_path=args.global_config)

    matmul_precision = str(cfg.model.compute_dtype)
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    global_seed = cfg.get("global_seed")
    if global_seed is not None:
        np.random.seed(int(global_seed))
    tokenizer = load_tokenizer(cfg)

    stage_cfgs = parse_stage_configs(cfg)

    base_root = Path(cfg.paths.data_root)

    dataset_root = Path(cfg.paths.processed_data_root)
    if not dataset_root.is_absolute():
        dataset_root = (base_root / dataset_root).resolve()
    configured_global_batch_size = cfg.training.get("global_batch_size", None)
    if configured_global_batch_size is not None:
        global_batch_size = int(configured_global_batch_size)
        if global_batch_size <= 0 or global_batch_size % LOCAL_DEVICE_COUNT != 0:
            raise ValueError(
                "training.global_batch_size must be positive and divisible by local device count "
                f"({LOCAL_DEVICE_COUNT})"
            )
        per_device_batch_size = global_batch_size // LOCAL_DEVICE_COUNT
        cfg.training.batch_size = int(per_device_batch_size)
    else:
        per_device_batch_size = int(cfg.training.batch_size)
        global_batch_size = per_device_batch_size * LOCAL_DEVICE_COUNT
    grad_accum = max(1, int(getattr(cfg.training, "gradient_accumulation", 1)))
    use_grad_accum = grad_accum > 1
    seed = getattr(cfg.training, "seed", None)
    if seed is None:
        seed = global_seed if global_seed is not None else 0
    seed = int(seed)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )
    stage_runtimes = build_stage_runtimes(
        stage_cfgs,
        dataset_root=dataset_root,
        batch_size=global_batch_size,
        seed=seed,
        pad_token_id=pad_token_id,
    )
    total_steps = sum(stage.total_steps for stage in stage_runtimes)
    stage_name_width = max((len(stage.config.name) for stage in stage_runtimes), default=1)
    total_update_steps = max(1, (total_steps + grad_accum - 1) // grad_accum)
    warmup_updates, warmup_tokens, resolved_warmup_tokens = resolve_warmup_updates(
        cfg,
        stage_runtimes,
        global_batch_size=global_batch_size,
        grad_accum=grad_accum,
    )
    print(
        f"[devices] backend={DEFAULT_DEVICE.platform} local_devices={LOCAL_DEVICE_COUNT} "
        f"multi_gpu_backend={MULTI_GPU_BACKEND} data_parallel={LOCAL_DEVICE_COUNT > 1} "
        f"per_device_batch={per_device_batch_size} "
        f"global_batch={global_batch_size}"
    )
    if LOCAL_DEVICE_COUNT > 1 and os.environ.get("NCCL_P2P_DISABLE") == "1":
        detail = NCCL_P2P_NOTE or "using NCCL_P2P_DISABLE=1"
        print(f"[devices] {detail}")
    optimizer_msg = (
        f"[optimizer] micro_steps={total_steps} update_steps={total_update_steps} "
        f"grad_accum={grad_accum} warmup_updates={warmup_updates}"
    )
    if warmup_tokens is not None:
        optimizer_msg += f" warmup_tokens={warmup_tokens}"
        if resolved_warmup_tokens is not None:
            optimizer_msg += f" resolved_warmup_tokens={resolved_warmup_tokens}"
    print(optimizer_msg)

    max_seq_len = max(stage.config.seq_len for stage in stage_runtimes)
    num_kv_heads = int(cfg.model.num_kv_heads)
    rotary_dim = int(cfg.model.rope_dim)
    use_remat = bool(cfg.model.use_remat)
    enable_xsa = bool(cfg.model.enable_xsa)
    model_mode = resolve_model_mode(cfg.model)
    causal = model_mode_is_causal(model_mode)
    mask_answer_token_for_encoder = bool(cfg.model.get("mask_answer_token_for_encoder", False))
    param_dtype = _to_dtype(cfg.model.param_dtype)
    compute_dtype = _to_dtype(cfg.model.compute_dtype)
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        num_kv_heads=num_kv_heads,
        rotary_dim=rotary_dim,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        use_remat=use_remat,
        enable_xsa=enable_xsa,
        mode=model_mode,
        causal=causal,
        mask_answer_token_for_encoder=mask_answer_token_for_encoder,
    )

    rng = jax.random.PRNGKey(seed)
    init_batch_size = per_device_batch_size if (USE_SHARD_MAP or USE_PMAP) else global_batch_size
    params = model.init(rng, jnp.zeros((init_batch_size, max_seq_len), dtype=jnp.int32))["params"]
    params = _freeze_if_dict(params)

    optimizer = build_optimizer(cfg, total_update_steps, warmup_updates, params)
    opt_state = None
    global_step = 0

    checkpoint_dir_arg = args.checkpoint_dir
    if checkpoint_dir_arg == "checkpoints":
        cfg_checkpoint_root = cfg.paths.get("checkpoints_root", None)
        if cfg_checkpoint_root:
            checkpoint_dir_arg = str(cfg_checkpoint_root)

    checkpoint_root = Path(checkpoint_dir_arg)
    if not checkpoint_root.is_absolute():
        checkpoint_root = (base_root / checkpoint_root).resolve()
    if checkpoint_root.name in {"params", "training_states"}:
        checkpoint_root = checkpoint_root.parent
    params_dir = checkpoint_root / "params"
    training_states_dir = checkpoint_root / "training_states"
    upload_log_dir = checkpoint_root / "upload_logs"
    cfg.paths.dataloader_state_root = str(training_states_dir / "dataloader_state")
    params_dir_str = str(params_dir)
    training_states_dir_str = str(training_states_dir)
    checkpoint_every = args.checkpoint_every or cfg.training.checkpoint_every
    upload_on_checkpoint = bool(args.upload_on_checkpoint or _artifact_upload_enabled(cfg, "checkpoints"))
    upload_processes: List[tuple[str, subprocess.Popen[str], Path]] = []

    params_dir.mkdir(parents=True, exist_ok=True)
    log_path = params_dir / "logs.txt"
    log_file = open(log_path, "a", encoding="utf-8")

    training_cfg = cfg.training
    mini_every = int(getattr(training_cfg, "mini_checkpoint_every", max(1, checkpoint_every // 10)))
    mini_max_to_keep = int(getattr(training_cfg, "mini_max_to_keep", 3))
    mini_ckpt_dir = training_states_dir / "mini"
    mini_ckpt_mgr: Optional[AsyncMiniCheckpointManager] = None
    if mini_every > 0:
        mini_ckpt_mgr = AsyncMiniCheckpointManager(
            ckpt_dir=mini_ckpt_dir,
            max_to_keep=mini_max_to_keep,
        )
    else:
        print()
        print("=" * 96)
        print(
            f"[mini-checkpoints] DISABLED by config: training.mini_checkpoint_every={mini_every}. "
            "Mini checkpoints will not be created or resumed; only full checkpoints are eligible."
        )
        print("=" * 96)
        print()

    stage_states: Dict[str, Dict[str, int]] = {
        runtime.config.name: runtime.loader.state_dict() for runtime in stage_runtimes
    }
    current_stage_idx = 0
    stage_step_total = 0

    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init_checkpoint, not both.")

    resume_request = args.resume
    init_checkpoint_request = args.init_checkpoint
    if resume_request == "latest":
        opt_state = optimizer.init(params)
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
        if mini_ckpt_mgr is None:
            print(
                f"[mini-checkpoints] Skipping any existing mini checkpoints under {mini_ckpt_dir} because "
                f"training.mini_checkpoint_every={mini_every} disables them in this config and their "
                "provenance cannot be trusted for resume."
            )
            print("[mini-checkpoints] Falling back to the latest full checkpoint.")
            resume_request = "latest_full"
        else:
            restored_state, restored_step = mini_ckpt_mgr.restore_latest(mini_state_template)
            if restored_step:
                params = _freeze_if_dict(restored_state["params"])
                opt_state = restored_state["opt_state"]
                global_step = int(restored_state.get("global_step", restored_step))
                current_stage_idx = int(restored_state.get("stage_index", 0))
                stage_step_total = int(restored_state.get("stage_step_total", 0))
                stage_states = restored_state.get("stage_states", {})
                resumed_from_mini = True
                print(f"↩ Resumed from mini checkpoint at step {restored_step}")
            else:
                resume_request = "latest_full"
    elif resume_request and resume_request not in {"latest", "latest_full"}:
        resume_path = Path(resume_request)
        if not resume_path.is_absolute():
            resume_request = str((base_root / resume_path).resolve())
    if init_checkpoint_request is not None:
        init_path = Path(init_checkpoint_request)
        if not init_path.is_absolute():
            init_checkpoint_request = str((base_root / init_path).resolve())

    if resume_request and not resumed_from_mini:
        if resume_request == "latest_full":
            ckpt_path = latest_ckpt(params_dir_str)
            if ckpt_path is None:
                raise FileNotFoundError("No checkpoints available to resume from.")
        else:
            ckpt_path = resume_request
        params, global_step = load_ckpt(ckpt_path)
        params = _freeze_if_dict(params)
        opt_state = optimizer.init(params)
        opt_bytes = load_opt_state(global_step, training_states_dir_str)
        if opt_bytes is not None:
            try:
                opt_state = serialization.from_bytes(opt_state, opt_bytes)
                print(f"▶ Resumed optimizer state from {training_states_dir_str}")
            except Exception as exc:
                print(f"⚠ Failed to restore optimizer state ({exc}); reinitializing.")
        else:
            print("⚠ No optimizer state found; proceeding with fresh AdamW buffers.")
        if use_grad_accum and global_step % grad_accum != 0:
            print("⚠ Resuming mid-gradient-accumulation window; partial accumulated gradients are dropped.")
        print(f"▶ Resumed parameters from {ckpt_path} at step {global_step}")
    elif init_checkpoint_request is not None:
        params, _ = load_ckpt(init_checkpoint_request)
        params = _freeze_if_dict(params)
        opt_state = optimizer.init(params)
        global_step = 0
        print(f"▶ Initialized parameters from {init_checkpoint_request} at step 0")

    loader_state = None
    if not resumed_from_mini:
        loader_state = load_dataloader_state(dataloader_state_path(cfg, global_step)) if global_step else None
        if loader_state:
            current_stage_idx = int(loader_state.get("stage_index", 0))
            stage_step_total = int(loader_state.get("stage_step_total", 0))
            stage_states = loader_state.get("stage_states", {})
            print(f"▶ Restored dataloader state at stage {current_stage_idx} step {stage_step_total}")

    for idx, runtime in enumerate(stage_runtimes):
        if idx != current_stage_idx:
            continue
        state_dict = stage_states.get(runtime.config.name, {"epoch": 0, "step_in_epoch": 0})
        try:
            runtime.loader.load_state(state_dict)
        except Exception as exc:
            is_current_stage = idx == current_stage_idx
            if is_current_stage:
                print(
                    f"⚠ Failed to restore dataloader state for stage {runtime.config.name} ({exc}); "
                    "restarting this stage from the beginning."
                )
                runtime.loader.load_state({"epoch": 0, "step_in_epoch": 0})
                stage_states[runtime.config.name] = {"epoch": 0, "step_in_epoch": 0}
                stage_step_total = 0
            else:
                raise

    if 0 <= current_stage_idx < len(stage_runtimes):
        current_runtime = stage_runtimes[current_stage_idx]
        stage_step_total = (
            current_runtime.loader.epoch * current_runtime.loader.steps_per_epoch
            + current_runtime.loader.step_in_epoch
        )

    _startup_marker("placing params on training devices")
    params = _place_on_training_devices(params)
    _startup_marker("params placed")
    if opt_state is None and USE_PMAP:
        _startup_marker("initializing optimizer state on training devices")

        @partial(jax.pmap, axis_name="data")
        def _init_opt_state_pmap(replicated_params):
            return optimizer.init(replicated_params)

        opt_state = _init_opt_state_pmap(params)
        _startup_marker("optimizer state ready")
    else:
        if opt_state is None:
            _startup_marker("initializing optimizer state on host")
            opt_state = optimizer.init(_checkpoint_tree(params))
            _startup_marker("optimizer state initialized on host")
        _startup_marker("placing optimizer state on training devices")
        opt_state = _place_on_training_devices(opt_state)
        _startup_marker("optimizer state placed")
    if USE_SHARD_MAP:
        _startup_marker("building shard_map specs")
        params_spec = jax.tree_util.tree_map(lambda _: P(), jax.device_get(params))
        opt_state_spec = jax.tree_util.tree_map(lambda _: P(), jax.device_get(opt_state))
        batch_chunk_spec = {
            "input": P(None, "data", None),
            "target": P(None, "data", None),
            "mask": P(None, "data", None),
            "length": P(None, "data"),
        }
        _startup_marker("shard_map specs ready")

    base_rng = jax.random.PRNGKey(seed)
    cfg_chunk = int(getattr(cfg.training, "scan_chunk", 1))
    chunk_size = max(1, int(args.scan_chunk)) if args.scan_chunk is not None else max(1, cfg_chunk)
    autotune_scan_chunk = bool(getattr(cfg.training, "autotune_scan_chunk", False))
    autotune_scan_chunk_chunks = max(1, int(getattr(cfg.training, "autotune_scan_chunk_chunks", 4)))
    autotune_scan_chunk_persist = bool(getattr(cfg.training, "autotune_scan_chunk_persist", False))
    autotune_scan_chunk_candidates = _parse_int_candidates(
        getattr(cfg.training, "autotune_scan_chunk_candidates", None),
        default=[1, 2, 4, 8, 16],
    )
    if chunk_size not in autotune_scan_chunk_candidates:
        autotune_scan_chunk_candidates = [chunk_size] + autotune_scan_chunk_candidates
    default_scan_steps = autotune_scan_chunk_chunks * max(autotune_scan_chunk_candidates or [chunk_size])
    autotune_scan_chunk_steps = max(
        1,
        int(getattr(cfg.training, "autotune_scan_chunk_steps", default_scan_steps)),
    )
    cfg.training.scan_chunk = int(chunk_size)

    prefetch_default = 2 if IS_GPU else 0
    cfg_prefetch = int(getattr(cfg.training, "prefetch_size", prefetch_default))
    prefetch_size = max(0, int(args.prefetch)) if args.prefetch is not None else max(0, cfg_prefetch)
    autotune_prefetch = bool(getattr(cfg.training, "autotune_prefetch", False))
    autotune_prefetch_chunks = max(1, int(getattr(cfg.training, "autotune_prefetch_chunks", 4)))
    autotune_prefetch_persist = bool(getattr(cfg.training, "autotune_prefetch_persist", False))
    autotune_prefetch_candidates = _parse_int_candidates(
        getattr(cfg.training, "autotune_prefetch_candidates", None),
        default=[2, 4, 8, 16],
    )
    if prefetch_size not in autotune_prefetch_candidates:
        autotune_prefetch_candidates = [prefetch_size] + autotune_prefetch_candidates
    cfg.training.prefetch_size = int(prefetch_size)

    nan_check = bool(getattr(cfg.training, "nan_check", True))

    def _init_accum_grads(pytree):
        if not use_grad_accum:
            value = jnp.asarray(0, dtype=jnp.int32)
            return _place_on_training_devices(value)
        return jax.tree_util.tree_map(jnp.zeros_like, pytree)

    def _init_accum_count():
        value = jnp.asarray(0, dtype=jnp.int32)
        return _place_on_training_devices(value)

    def _stack_batches(batches):
        return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *batches)

    def _next_chunk(batch_iter, chunk_len: int, *, with_state: bool = False):
        batch_list = []
        last_state = None
        for _ in range(chunk_len):
            try:
                item = next(batch_iter)
            except StopIteration:
                break
            if with_state:
                batch, last_state = item
            else:
                batch = item
            batch_list.append(batch)
        if not batch_list:
            return (None, None) if with_state else None
        chunk = _stack_batches(batch_list)
        if USE_SHARD_MAP:
            chunk = _place_batch_on_training_devices(chunk, chunked=True)
        elif USE_PMAP:
            chunk = jax.tree_util.tree_map(lambda x: np.swapaxes(np.asarray(x), 0, 1), chunk)
        else:
            chunk = jax.device_put(chunk, DEFAULT_DEVICE)
        return (chunk, last_state) if with_state else chunk

    if USE_SHARD_MAP:
        accum_grads_spec = jax.tree_util.tree_map(lambda _: P(), jax.device_get(_init_accum_grads(params)))
        accum_count_spec = P()
        loss_spec = P(None)

        def _run_chunk_local(params, opt_state, batch_chunk, start_step, accum_grads, accum_count):
            grad_scale = jnp.asarray(1.0 / grad_accum, dtype=jnp.float32)
            replica_index = jax.lax.axis_index("data")

            def body(carry, batch):
                params, opt_state, step, accum_grads, accum_count = carry
                dropout_rng = jax.random.fold_in(base_rng, step)
                dropout_rng = jax.random.fold_in(dropout_rng, replica_index)

                loss, grads = loss_and_grad(
                    params,
                    batch,
                    model=model,
                    dropout_rng=dropout_rng,
                    axis_name="data",
                )

                if use_grad_accum:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)
                        accum_grads = jax.lax.cond(
                            is_finite,
                            lambda ag, g: jax.tree_util.tree_map(lambda a, g_: a + g_, ag, g),
                            lambda ag, g: ag,
                            accum_grads,
                            grads,
                        )
                        accum_count = jnp.where(is_finite, accum_count + 1, accum_count)
                    else:
                        accum_grads = jax.tree_util.tree_map(lambda a, g_: a + g_, accum_grads, grads)
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
                else:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)

                        def apply_updates(args):
                            params, opt_state, grads = args
                            updates, opt_state = optimizer.update(grads, opt_state, params)
                            params = optax.apply_updates(params, updates)
                            return params, opt_state

                        params, opt_state = jax.lax.cond(
                            is_finite,
                            apply_updates,
                            lambda args: (args[0], args[1]),
                            (params, opt_state, grads),
                        )
                    else:
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                return (params, opt_state, step + 1, accum_grads, accum_count), loss

            (params, opt_state, _, accum_grads, accum_count), losses = jax.lax.scan(
                body, (params, opt_state, start_step, accum_grads, accum_count), batch_chunk
            )
            return params, opt_state, accum_grads, accum_count, losses

        _run_chunk = jax.jit(
            jax.shard_map(
                _run_chunk_local,
                mesh=TRAINING_MESH,
                in_specs=(params_spec, opt_state_spec, batch_chunk_spec, P(), accum_grads_spec, P()),
                out_specs=(params_spec, opt_state_spec, accum_grads_spec, accum_count_spec, loss_spec),
                axis_names={"data"},
                check_vma=False,
            )
        )

        def _apply_accum_local(params, opt_state, accum_grads, denom):
            grads = jax.tree_util.tree_map(lambda g: g / denom, accum_grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        _apply_accum = jax.jit(
            jax.shard_map(
                _apply_accum_local,
                mesh=TRAINING_MESH,
                in_specs=(params_spec, opt_state_spec, accum_grads_spec, P()),
                out_specs=(params_spec, opt_state_spec),
                axis_names={"data"},
            )
        )
    elif USE_PMAP:
        @partial(jax.pmap, axis_name="data")
        def _run_chunk(params, opt_state, batch_chunk, start_step, accum_grads, accum_count):
            grad_scale = jnp.asarray(1.0 / grad_accum, dtype=jnp.float32)
            replica_index = jax.lax.axis_index("data")

            def body(carry, batch):
                params, opt_state, step, accum_grads, accum_count = carry
                dropout_rng = jax.random.fold_in(base_rng, step)
                dropout_rng = jax.random.fold_in(dropout_rng, replica_index)

                loss, grads = loss_and_grad(
                    params,
                    batch,
                    model=model,
                    dropout_rng=dropout_rng,
                    axis_name="data",
                )

                if use_grad_accum:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)
                        accum_grads = jax.lax.cond(
                            is_finite,
                            lambda ag, g: jax.tree_util.tree_map(lambda a, g_: a + g_, ag, g),
                            lambda ag, g: ag,
                            accum_grads,
                            grads,
                        )
                        accum_count = jnp.where(is_finite, accum_count + 1, accum_count)
                    else:
                        accum_grads = jax.tree_util.tree_map(lambda a, g_: a + g_, accum_grads, grads)
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
                else:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)

                        def apply_updates(args):
                            params, opt_state, grads = args
                            updates, opt_state = optimizer.update(grads, opt_state, params)
                            params = optax.apply_updates(params, updates)
                            return params, opt_state

                        params, opt_state = jax.lax.cond(
                            is_finite,
                            apply_updates,
                            lambda args: (args[0], args[1]),
                            (params, opt_state, grads),
                        )
                    else:
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                return (params, opt_state, step + 1, accum_grads, accum_count), loss

            (params, opt_state, _, accum_grads, accum_count), losses = jax.lax.scan(
                body, (params, opt_state, start_step, accum_grads, accum_count), batch_chunk
            )
            return params, opt_state, accum_grads, accum_count, losses

        @partial(jax.pmap, axis_name="data")
        def _apply_accum(params, opt_state, accum_grads, denom):
            grads = jax.tree_util.tree_map(lambda g: g / denom, accum_grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state
    else:
        @jax.jit
        def _run_chunk(params, opt_state, batch_chunk, start_step, accum_grads, accum_count):
            grad_scale = jnp.asarray(1.0 / grad_accum, dtype=jnp.float32)

            def body(carry, batch):
                params, opt_state, step, accum_grads, accum_count = carry
                dropout_rng = jax.random.fold_in(base_rng, step)

                loss, grads = loss_and_grad(
                    params,
                    batch,
                    model=model,
                    dropout_rng=dropout_rng,
                )

                if use_grad_accum:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)
                        accum_grads = jax.lax.cond(
                            is_finite,
                            lambda ag, g: jax.tree_util.tree_map(lambda a, g_: a + g_, ag, g),
                            lambda ag, g: ag,
                            accum_grads,
                            grads,
                        )
                        accum_count = jnp.where(is_finite, accum_count + 1, accum_count)
                    else:
                        accum_grads = jax.tree_util.tree_map(lambda a, g_: a + g_, accum_grads, grads)
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
                else:
                    if nan_check:
                        is_finite = jnp.isfinite(loss)

                        def apply_updates(args):
                            params, opt_state, grads = args
                            updates, opt_state = optimizer.update(grads, opt_state, params)
                            params = optax.apply_updates(params, updates)
                            return params, opt_state

                        params, opt_state = jax.lax.cond(
                            is_finite,
                            apply_updates,
                            lambda args: (args[0], args[1]),
                            (params, opt_state, grads),
                        )
                    else:
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                return (params, opt_state, step + 1, accum_grads, accum_count), loss

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

    should_autotune_scan_chunk = (
        autotune_scan_chunk
        and IS_GPU
        and args.scan_chunk is None
        and args.resume is None
        and global_step == 0
        and len(stage_runtimes) > 0
    )
    if should_autotune_scan_chunk:
        tune_runtime = stage_runtimes[current_stage_idx]
        loader_snapshot = tune_runtime.loader.state_dict()
        candidates = [max(1, int(v)) for v in autotune_scan_chunk_candidates if int(v) > 0]
        if not candidates:
            candidates = [chunk_size]
        if chunk_size not in candidates:
            candidates = [chunk_size] + candidates

        print(
            f"[autotune] scan_chunk benchmark stage={tune_runtime.config.name} "
            f"prefetch={prefetch_size} candidates={candidates} steps={autotune_scan_chunk_steps}"
        )

        def _benchmark_scan_chunk(candidate: int) -> float:
            tune_runtime.loader.load_state(loader_snapshot)
            batch_iter = _prefetch_training_batches(
                tune_runtime.loader,
                size=prefetch_size,
                per_device_batch=per_device_batch_size,
            )

            probe_params = params
            probe_opt_state = opt_state
            probe_accum_grads = _init_accum_grads(params)
            probe_accum_count = _init_accum_count()
            probe_step = global_step

            warm_chunk = _next_chunk(batch_iter, candidate)
            if warm_chunk is None:
                return 0.0
            probe_params, probe_opt_state, probe_accum_grads, probe_accum_count, warm_losses = _run_chunk(
                probe_params,
                probe_opt_state,
                warm_chunk,
                _training_scalar(probe_step, jnp.int32),
                probe_accum_grads,
                probe_accum_count,
            )
            warm_host = _host_losses(warm_losses)
            if len(warm_host) > 0:
                probe_step += len(warm_host)

            measured_steps = 0
            t0 = time.perf_counter()
            loops = max(1, int(np.ceil(autotune_scan_chunk_steps / candidate)))
            for _ in range(loops):
                chunk = _next_chunk(batch_iter, candidate)
                if chunk is None:
                    break
                probe_params, probe_opt_state, probe_accum_grads, probe_accum_count, losses = _run_chunk(
                    probe_params,
                    probe_opt_state,
                    chunk,
                    _training_scalar(probe_step, jnp.int32),
                    probe_accum_grads,
                    probe_accum_count,
                )
                losses_host = _host_losses(losses)
                chunk_steps = len(losses_host)
                if chunk_steps == 0:
                    break
                measured_steps += chunk_steps
                probe_step += chunk_steps
            elapsed = time.perf_counter() - t0
            if measured_steps <= 0 or elapsed <= 0:
                return 0.0
            return measured_steps / elapsed

        best_chunk_size = chunk_size
        best_tps = -1.0
        for candidate in candidates:
            try:
                tps = _benchmark_scan_chunk(candidate)
                print(f"[autotune] scan_chunk={candidate} -> {tps:.3f} steps/s")
                if tps > best_tps:
                    best_tps = tps
                    best_chunk_size = int(candidate)
            except Exception as exc:
                print(f"[autotune] scan_chunk={candidate} failed: {exc}")

        tune_runtime.loader.load_state(loader_snapshot)
        chunk_size = int(best_chunk_size)
        cfg.training.scan_chunk = int(chunk_size)
        print(f"[autotune] selected scan_chunk={chunk_size}")

        if autotune_scan_chunk_persist and _persist_scan_chunk(args.config, chunk_size):
            print(f"[autotune] wrote training.scan_chunk={chunk_size} to {args.config}")

    should_autotune_prefetch = (
        autotune_prefetch
        and IS_GPU
        and args.prefetch is None
        and args.resume is None
        and global_step == 0
        and len(stage_runtimes) > 0
    )
    if should_autotune_prefetch:
        tune_runtime = stage_runtimes[current_stage_idx]
        loader_snapshot = tune_runtime.loader.state_dict()
        candidates = [int(v) for v in autotune_prefetch_candidates if int(v) >= 0]
        if not candidates:
            candidates = [prefetch_size]
        if prefetch_size not in candidates:
            candidates = [prefetch_size] + candidates

        print(
            f"[autotune] prefetch benchmark stage={tune_runtime.config.name} "
            f"chunk={chunk_size} candidates={candidates}"
        )

        def _benchmark_prefetch(candidate: int) -> float:
            tune_runtime.loader.load_state(loader_snapshot)
            batch_iter = _prefetch_training_batches(
                tune_runtime.loader,
                size=int(candidate),
                per_device_batch=per_device_batch_size,
            )

            probe_params = params
            probe_opt_state = opt_state
            probe_accum_grads = _init_accum_grads(params)
            probe_accum_count = _init_accum_count()
            probe_step = global_step

            warm_chunk = _next_chunk(batch_iter, chunk_size)
            if warm_chunk is None:
                return 0.0
            probe_params, probe_opt_state, probe_accum_grads, probe_accum_count, warm_losses = _run_chunk(
                probe_params,
                probe_opt_state,
                warm_chunk,
                _training_scalar(probe_step, jnp.int32),
                probe_accum_grads,
                probe_accum_count,
            )
            warm_host = _host_losses(warm_losses)
            if len(warm_host) > 0:
                probe_step += len(warm_host)

            measured_steps = 0
            t0 = time.perf_counter()
            for _ in range(autotune_prefetch_chunks):
                chunk = _next_chunk(batch_iter, chunk_size)
                if chunk is None:
                    break
                probe_params, probe_opt_state, probe_accum_grads, probe_accum_count, losses = _run_chunk(
                    probe_params,
                    probe_opt_state,
                    chunk,
                    _training_scalar(probe_step, jnp.int32),
                    probe_accum_grads,
                    probe_accum_count,
                )
                losses_host = _host_losses(losses)
                chunk_steps = len(losses_host)
                if chunk_steps == 0:
                    break
                measured_steps += chunk_steps
                probe_step += chunk_steps
            elapsed = time.perf_counter() - t0
            if measured_steps <= 0 or elapsed <= 0:
                return 0.0
            return measured_steps / elapsed

        best_prefetch = prefetch_size
        best_tps = -1.0
        for candidate in candidates:
            try:
                tps = _benchmark_prefetch(candidate)
                print(f"[autotune] prefetch={candidate} -> {tps:.3f} steps/s")
                if tps > best_tps:
                    best_tps = tps
                    best_prefetch = int(candidate)
            except Exception as exc:
                print(f"[autotune] prefetch={candidate} failed: {exc}")

        tune_runtime.loader.load_state(loader_snapshot)
        prefetch_size = int(best_prefetch)
        cfg.training.prefetch_size = int(prefetch_size)
        print(f"[autotune] selected prefetch_size={prefetch_size}")

        if autotune_prefetch_persist and _persist_prefetch_size(args.config, prefetch_size):
            print(f"[autotune] wrote training.prefetch_size={prefetch_size} to {args.config}")

    if not nan_check:
        print("[training] nan_check=False: NaN/Inf guard disabled")

    start = time.time()
    last_loss = None
    if use_grad_accum:
        _startup_marker("initializing accum_grads")
        accum_grads = _init_accum_grads(params)
        _startup_marker("accum_grads ready")
        _startup_marker("initializing accum_count")
        accum_count = _init_accum_count()
        _startup_marker("accum_count ready")
    else:
        _startup_marker("gradient_accumulation=1; skipping accum_grads allocation")
        accum_grads = _init_accum_grads(params)
        accum_count = _init_accum_count()
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
            dynamic_ncols=False,
            ncols=120,
        )
        last_loss = None

        batch_iter = _prefetch_training_batches(
            runtime.loader,
            size=prefetch_size,
            per_device_batch=per_device_batch_size,
            state_getter=runtime.loader.state_dict,
        )
        while completed_in_stage < stage_steps_target:
            _reap_async_uploads(upload_processes)
            remaining_steps = stage_steps_target - completed_in_stage
            request_chunk = min(chunk_size, max(1, remaining_steps))
            chunk, trained_loader_state = _next_chunk(batch_iter, request_chunk, with_state=True)
            if chunk is None:
                break
            params, opt_state, accum_grads, accum_count, losses = _run_chunk(
                params, opt_state, chunk, _training_scalar(global_step, jnp.int32), accum_grads, accum_count
            )
            losses = _host_losses(losses)
            chunk_len = len(losses)
            if chunk_len == 0:
                break
            chunk_start = global_step + 1
            global_step += chunk_len
            completed_in_stage += chunk_len
            pbar.update(chunk_len)
            stage_states[runtime.config.name] = (
                copy.deepcopy(trained_loader_state)
                if trained_loader_state is not None
                else runtime.loader.state_dict()
            )

            for offset, loss_val in enumerate(losses, start=0):
                loss_scalar = float(loss_val)
                last_loss = loss_scalar
                step_val = chunk_start + offset
                if step_val % cfg.training.log_every == 0:
                    elapsed = time.time() - start
                    log_msg = _format_train_progress_log(
                        step_val=step_val,
                        total_steps=total_steps,
                        stage_name=runtime.config.name,
                        stage_name_width=stage_name_width,
                        loss_val=loss_scalar,
                        elapsed=elapsed,
                    )
                    print()
                    print(log_msg)
                    log_file.write(log_msg + "\n")
                    log_file.flush()
                    start = time.time()

            if mini_ckpt_mgr is not None and (global_step % mini_every == 0):
                # Stage mini-checkpoint payload on host RAM so asynchronous orbax
                # writes do not retain additional large device buffers.
                mini_state = {
                    "params": _checkpoint_tree(params),
                    "opt_state": _checkpoint_tree(opt_state),
                    "global_step": global_step,
                    "stage_index": stage_idx,
                    "stage_step_total": completed_in_stage,
                    "stage_states": stage_states,
                }
                mini_ckpt_mgr.save(global_step, mini_state)

            if checkpoint_every and (global_step % checkpoint_every == 0):
                ckpt_file = save_training_state(
                    cfg=cfg,
                    params=params,
                    opt_state=opt_state,
                    params_dir=params_dir_str,
                    training_states_dir=training_states_dir_str,
                    global_step=global_step,
                    stage_idx=stage_idx,
                    completed_in_stage=completed_in_stage,
                    stage_states=stage_states,
                    runtime=runtime,
                    train_loss=last_loss,
                    loader_state=stage_states.get(runtime.config.name),
                )
                print(f"💾 checkpoint → {ckpt_file}")
                if upload_on_checkpoint:
                    _launch_async_upload(
                        label=f"checkpoint_{global_step}",
                        local_path=checkpoint_root,
                        log_dir=upload_log_dir,
                        upload_processes=upload_processes,
                        cfg=cfg,
                    )

            if _stop_requested:
                if use_grad_accum:
                    accum_count_host = _replicated_scalar_to_int(accum_count)
                    if accum_count_host > 0:
                        denom = _training_scalar(accum_count_host, jnp.float32)
                        params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
                        accum_grads = _init_accum_grads(params)
                        accum_count = _init_accum_count()
                if mini_ckpt_mgr is not None:
                    mini_ckpt_mgr.wait_until_finished(timeout=4.0)
                log_file.flush()
                log_file.close()
                print("[signal] Stop requested; exiting after current chunk.")
                pbar.close()
                return

        pbar.close()
        if use_grad_accum:
            accum_count_host = _replicated_scalar_to_int(accum_count)
            if accum_count_host > 0:
                denom = _training_scalar(accum_count_host, jnp.float32)
                params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
                accum_grads = _init_accum_grads(params)
                accum_count = _init_accum_count()
        if last_loss is not None:
            print(
                f"✓ Stage {runtime.config.name} completed (last loss {last_loss:.4f})"
            )
        else:
            print(f"✓ Stage {runtime.config.name} completed (no batches emitted)")
        # Stage finished → reset step tracker
        stage_step_total = 0

    if use_grad_accum:
        accum_count_host = _replicated_scalar_to_int(accum_count)
        if accum_count_host > 0:
            denom = _training_scalar(accum_count_host, jnp.float32)
            params, opt_state = _apply_accum(params, opt_state, accum_grads, denom)
            accum_grads = _init_accum_grads(params)
            accum_count = _init_accum_count()
    final_ckpt = save_ckpt(_checkpoint_tree(params), global_step, params_dir_str, train_loss=last_loss)
    if last_loss is not None:
        print(f"[metadata] Wrote train_loss={last_loss:.6f} to checkpoint {final_ckpt}")
    save_opt_state(_checkpoint_tree(opt_state), global_step, training_states_dir_str)
    if upload_on_checkpoint:
        _launch_async_upload(
            label=f"checkpoint_final_{global_step}",
            local_path=checkpoint_root,
            log_dir=upload_log_dir,
            upload_processes=upload_processes,
            cfg=cfg,
        )
    save_dataloader_state(
        dataloader_state_path(cfg, global_step),
        {
            "stage_index": len(stage_runtimes) - 1,
            "stage_step_total": stage_runtimes[-1].total_steps,
            "stage_states": stage_states,
        },
    )
    s3_outputs = []
    if upload_on_checkpoint:
        try:
            s3_outputs.append(_s3_uri_for_local_path(checkpoint_root, cfg))
        except ValueError:
            pass
    run_manifest_path = checkpoint_root / "run_manifest.json"
    write_manifest(
        run_manifest_path,
        build_manifest(
            kind="training",
            config_path=args.config,
            global_config_path=args.global_config,
            outputs=[checkpoint_root, final_ckpt],
            s3_outputs=s3_outputs,
            extra={"global_step": int(global_step), "last_loss": float(last_loss) if last_loss is not None else None},
        ),
    )
    print(f"[manifest] wrote {run_manifest_path}")
    print(f"✔ Training complete. Final checkpoint: {final_ckpt}")
    print("→ Use this checkpoint as --init_checkpoint for the QA finetune stage.")
    elapsed_seconds = time.perf_counter() - command_start
    runtime_msg = (
        f"Command executed in {_format_wall_time(elapsed_seconds)} "
        f"({elapsed_seconds:.1f}s)"
    )
    print(runtime_msg)
    log_file.write(runtime_msg + "\n")
    if mini_ckpt_mgr is not None:
        mini_ckpt_mgr.wait_until_finished(timeout=10.0)
    _reap_async_uploads(upload_processes)
    if upload_processes:
        print(f"[upload] {len(upload_processes)} upload subprocess(es) still running in background.")
    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    main()
