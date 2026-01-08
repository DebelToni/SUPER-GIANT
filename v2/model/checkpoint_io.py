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
    flat: Dict[Tuple[str, ...], np.ndarray] = flatten_dict(jax.device_get(params))
    payload = {"/".join(k): _as_numpy(v) for k, v in flat.items()}
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
