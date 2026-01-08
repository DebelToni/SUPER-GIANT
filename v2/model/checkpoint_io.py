"""
Tiny utility to (de)serialize Flax/JAX parameter PyTrees
to a single *.npz* file. Works for both CPU and GPU tensors.

Why not pickle?
---------------
✓ portable between Python versions
✓ inspectable with 'np.load' if needed
✓ no security worries when sharing checkpoints
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


NAME_KEY = "__checkpoint_name__"


def _as_numpy(x):
    return np.asarray(x, dtype=x.dtype)


def _normalize_name(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _extract_name(npz: np.lib.npyio.NpzFile) -> Optional[str]:
    if NAME_KEY not in npz:
        return None
    raw = npz[NAME_KEY]
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return _normalize_name(raw.item())
        if raw.size > 0:
            return _normalize_name(raw.flat[0])
        return None
    return _normalize_name(raw)


def save_npz(params: Dict, path, name: Optional[str] = None):
    """
    Save *params* (a PyTree/FrozenDict) to **path** with names like
    'Embed_0/embedding', 'Block_3/kv_proj/kernel' …
    """
    flat: Dict[Tuple[str, ...], np.ndarray] = flatten_dict(jax.device_get(params))
    payload = {"/".join(k): _as_numpy(v) for k, v in flat.items()}
    if name is not None:
        payload[NAME_KEY] = np.array(str(name))
    np.savez_compressed(path, **payload)


def load_npz(path, *, print_name: bool = True) -> Dict:
    """
    Load params back as a *nested* dict of NumPy arrays.
    You can `jax.device_put` afterwards if you like.
    """
    with np.load(path) as npz:
        name = _extract_name(npz)
        if print_name:
            display = name if name is not None else ""
            print(f'[i/o] Loading "{display}" params.')
        flat = {tuple(k.split("/")): v for k, v in npz.items() if k != NAME_KEY}
    return unflatten_dict(flat)


def get_npz_name(path) -> Optional[str]:
    with np.load(path) as npz:
        return _extract_name(npz)


def rename_npz(path, name: str) -> None:
    path = Path(path)
    with np.load(path) as npz:
        payload = {k: npz[k] for k in npz.files if k != NAME_KEY}
    payload[NAME_KEY] = np.array(str(name))
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)


def _main() -> None:
    parser = argparse.ArgumentParser("Checkpoint name utility")
    parser.add_argument("checkpoint", help="Path to a .npz checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--get-name", action="store_true", help="Print the checkpoint name.")
    group.add_argument("--rename", metavar="NAME", help="Rename the checkpoint metadata.")
    args = parser.parse_args()

    if args.get_name:
        name = get_npz_name(args.checkpoint)
        print("" if name is None else name)
        return

    rename_npz(args.checkpoint, args.rename)


if __name__ == "__main__":
    _main()
