"""
Checkpoint utilities:
- NPZ (de)serialization with metadata
- Step-based checkpoint management
- Async mini-checkpoint manager
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict, unflatten_dict

NAME_KEY = "__checkpoint_name__"
META_PREFIX = "__meta__/"
META_KEYS = {
    "name": "Human-friendly checkpoint name.",
    "commit_hash": "Git commit hash where this checkpoint was created or validated.",
    "example_command": "Example command to run inference with this checkpoint.",
}


def _as_numpy(x):
    return np.asarray(x, dtype=x.dtype)


def _normalize_name(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _extract_scalar(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_name(value.item())
        if value.size > 0:
            return _normalize_name(value.flat[0])
        return None
    return _normalize_name(value)


def _load_metadata(npz: np.lib.npyio.NpzFile) -> Dict[str, Optional[str]]:
    meta: Dict[str, Optional[str]] = {}
    for key in npz.files:
        if key.startswith(META_PREFIX):
            meta_key = key[len(META_PREFIX) :]
            meta[meta_key] = _extract_scalar(npz[key])
    if "name" not in meta and NAME_KEY in npz:
        meta["name"] = _extract_scalar(npz[NAME_KEY])
    return meta


def save_npz(
    params: Dict,
    path,
    name: Optional[str] = None,
    commit_hash: Optional[str] = None,
    example_command: Optional[str] = None,
    metadata: Optional[Dict[str, Optional[str]]] = None,
):
    """
    Save *params* (a PyTree/FrozenDict) to **path** with names like
    'Embed_0/embedding', 'Block_3/kv_proj/kernel' …
    """
    flat = flatten_dict(jax.device_get(params))
    payload: Dict[str, Any] = {"/".join(k): _as_numpy(v) for k, v in flat.items()}
    merged_meta: Dict[str, Optional[str]] = {}
    if metadata:
        merged_meta.update(metadata)
    if name is not None:
        merged_meta["name"] = name
    if commit_hash is not None:
        merged_meta["commit_hash"] = commit_hash
    if example_command is not None:
        merged_meta["example_command"] = example_command
    for key, value in merged_meta.items():
        if value is None:
            continue
        payload[f"{META_PREFIX}{key}"] = np.array(str(value))
    if name is not None:
        payload[NAME_KEY] = np.array(str(name))
    np.savez_compressed(path, **payload)


def load_npz(path, *, print_name: bool = True) -> Dict:
    """
    Load params back as a *nested* dict of NumPy arrays.
    You can `jax.device_put` afterwards if you like.
    """
    with np.load(path) as npz:
        meta = _load_metadata(npz)
        name = meta.get("name")
        if print_name:
            display = name if name is not None else ""
            print(f'[i/o] Loading "{display}" params.')
        flat = {
            tuple(k.split("/")): v
            for k, v in npz.items()
            if k != NAME_KEY and not k.startswith(META_PREFIX)
        }
    return unflatten_dict(flat)


def get_npz_metadata(path) -> Dict[str, Optional[str]]:
    with np.load(path) as npz:
        return _load_metadata(npz)


def set_npz_metadata(path, key: str, value: str) -> None:
    if key not in META_KEYS:
        raise ValueError(f"Unknown metadata key: {key}")
    path = Path(path)
    with np.load(path) as npz:
        payload = {
            k: npz[k]
            for k in npz.files
            if k != NAME_KEY and not k.startswith(META_PREFIX)
        }
        meta = _load_metadata(npz)
    meta[key] = value
    for meta_key, meta_val in meta.items():
        if meta_val is None:
            continue
        payload[f"{META_PREFIX}{meta_key}"] = np.array(str(meta_val))
    if meta.get("name") is not None:
        payload[NAME_KEY] = np.array(str(meta["name"]))
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)


_CKPT_RE = re.compile(r"step_(\d{7})\.npz$")


def _step_from_name(fname: str) -> int:
    m = _CKPT_RE.search(os.path.basename(fname))
    return int(m.group(1)) if m else -1


def save(params, step: int, ckpt_dir: str = "checkpoints") -> str:
    """
    Save *params* to  <ckpt_dir>/step_XXXXXXX.npz  (7-digit zero-padded counter).
    Returns the file path so callers can log it.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:07d}.npz")
    tmp_path = path + ".tmp"
    # Write to temp file then atomically rename to avoid partial checkpoints.
    save_npz(params, tmp_path)
    try:
        os.replace(tmp_path, path)
    except FileNotFoundError:
        # Fallback: write directly if temp was removed mid-save (e.g., preemption)
        save_npz(params, path)
    return path


def latest(ckpt_dir: str = "checkpoints") -> Optional[str]:
    """Return path to the numerically latest checkpoint or *None*."""
    if not os.path.isdir(ckpt_dir):
        return None
    files = [
        f
        for f in glob.glob(os.path.join(ckpt_dir, "step_*.npz"))
        if _CKPT_RE.search(os.path.basename(f))
    ]
    if not files:
        return None
    files = sorted(files, key=_step_from_name)
    return files[-1]


def load(path: str):
    """Return (*params*, step_number)."""
    return load_npz(path), _step_from_name(path)


def _opt_state_name(step: int) -> str:
    return f"opt_state_{step:07d}.msgpack"


def save_opt_state(opt_state, step: int, ckpt_dir: str = "checkpoints") -> str:
    """
    Serialize the Optax optimizer state alongside model params.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, _opt_state_name(step))
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(serialization.to_bytes(opt_state))
    try:
        os.replace(tmp_path, path)
    except FileNotFoundError:
        with open(path, "wb") as handle:
            handle.write(serialization.to_bytes(opt_state))
    return path


def load_opt_state(step: int, ckpt_dir: str = "checkpoints"):
    """
    Return serialized optimizer bytes for *step* or None if missing.
    Callers should pass the bytes through flax.serialization.from_bytes
    using a freshly-initialized opt_state template.
    """
    path = os.path.join(ckpt_dir, _opt_state_name(step))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return handle.read()


@dataclass
class AsyncMiniCheckpointManager:
    ckpt_dir: Path
    max_to_keep: int = 3

    def __post_init__(self):
        self.ckpt_dir = Path(self.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        handler = ocp.PyTreeCheckpointHandler()
        self.checkpointer = ocp.AsyncCheckpointer(handler)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep,
            create=True,
        )
        self.manager = ocp.CheckpointManager(
            str(self.ckpt_dir),
            self.checkpointer,
            options,
        )

    def latest_step(self) -> int | None:
        return self.manager.latest_step()

    def restore_latest(self, target: Any) -> tuple[Any, int]:
        """
        Restore into `target` (PyTree template) from latest mini checkpoint.
        Returns (restored_target, step). If none exist, returns (target, 0).
        """
        step = self.manager.latest_step()
        if step is None:
            return target, 0
        restored = self.manager.restore(step, items=target)
        return restored, step

    def save(self, step: int, state: Mapping[str, Any]) -> None:
        """
        Asynchronously save `state` at `step`.
        Returns quickly; actual I/O is in background threads.
        """
        save_args = orbax_utils.save_args_from_target(state)
        self.manager.save(
            step,
            state,
            save_kwargs={"save_args": save_args},
        )

    def wait_until_finished(self, timeout: float | None = None) -> None:
        """
        Optionally wait for any pending async writes.
        """
        _ = timeout  # retained for API compatibility; AsyncCheckpointer does not take a timeout arg.
        self.checkpointer.wait_until_finished()


def _main() -> None:
    parser = argparse.ArgumentParser("Checkpoint name utility")
    parser.add_argument("checkpoint", help="Path to a .npz checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--get", metavar="KEY", help="Print a metadata value.")
    group.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set a metadata value.")
    args = parser.parse_args()

    if args.get:
        meta = get_npz_metadata(args.checkpoint)
        if args.get not in META_KEYS:
            raise ValueError(f"Unknown metadata key: {args.get}")
        value = meta.get(args.get)
        print("" if value is None else value)
        return

    key, value = args.set
    set_npz_metadata(args.checkpoint, key, value)


if __name__ == "__main__":
    _main()
