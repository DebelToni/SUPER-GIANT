# checkpoint_manager.py
import os, re, glob
from typing import Tuple, Optional

from checkpoint_io import save_npz, load_npz


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
    save_npz(params, path)
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

