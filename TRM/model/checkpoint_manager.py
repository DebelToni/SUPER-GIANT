import glob
import os
import re
from typing import Optional

from checkpoint_io import load_npz, save_npz
from flax import serialization


_CKPT_RE = re.compile(r"step_(\d{7})\.npz$")


def _step_from_name(fname: str) -> int:
    m = _CKPT_RE.search(os.path.basename(fname))
    return int(m.group(1)) if m else -1


def save(params, step: int, ckpt_dir: str = "checkpoints") -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:07d}.npz")
    tmp_path = (path[:-4] + ".tmp.npz") if path.endswith(".npz") else (path + ".tmp.npz")
    save_npz(params, tmp_path)
    try:
        os.replace(tmp_path, path)
    except FileNotFoundError:
        save_npz(params, path)
    return path


def latest(ckpt_dir: str = "checkpoints") -> Optional[str]:
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
    return load_npz(path), _step_from_name(path)


def _opt_state_name(step: int) -> str:
    return f"opt_state_{step:07d}.msgpack"


def save_opt_state(opt_state, step: int, ckpt_dir: str = "checkpoints") -> str:
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
    path = os.path.join(ckpt_dir, _opt_state_name(step))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return handle.read()
