# checkpoint_manager.py
import os, re, glob
from typing import Optional

from checkpoint_io import save_npz, load_npz
from flax import serialization


def _step_from_name(fname: str) -> int:
    m = re.search(r"step_(\d+)\.npz$", fname)
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
    os.replace(tmp_path, path)
    return path


def latest(ckpt_dir: str = "checkpoints") -> Optional[str]:
    """Return path to the numerically latest checkpoint or *None*."""
    if not os.path.isdir(ckpt_dir):
        return None
    files = glob.glob(os.path.join(ckpt_dir, "step_*.npz"))
    return max(files, key=_step_from_name) if files else None


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
    os.replace(tmp_path, path)
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
