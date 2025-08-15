from __future__ import annotations

"""
checkpoint_torch_io.py — Translate JAX/Flax checkpoints to PyTorch and load them.

Usage (CLI):

  # Convert a JAX .npz checkpoint to a PyTorch .pt state_dict file
  python checkpoint_torch_io.py translate \
      --in ckpt_jax.npz --out ckpt_torch.pt

  # Dry-run to see key mappings
  python checkpoint_torch_io.py translate --in ckpt_jax.npz --dry-run

Library API:

  from checkpoint_torch_io import load_jax_npz_to_torch_state_dict
  state = load_jax_npz_to_torch_state_dict("ckpt_jax.npz", model)
  model.load_state_dict(state, strict=True)

This module is intentionally tolerant of small naming differences often seen in
Flax checkpoints (e.g., "Embed_0/embedding" vs "embed/embedding").
"""

import argparse
import re
from collections import OrderedDict
from typing import Dict, Tuple, Optional

import numpy as np
import torch

# ------------------------------ helper predicates ------------------------------

_LINEAR_SUFFIXES = ("/kernel", "kernel:0", ".kernel")
_BIAS_SUFFIXES = ("/bias", "bias:0", ".bias")
_SCALE_SUFFIXES = ("/scale", "scale:0", ".scale")
_EMBED_SUFFIXES = ("/embedding", "embedding:0", ".embedding")


def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix) :] if k.startswith(prefix) else k


def _normalize_key(k: str) -> str:
    # Remove common wrappers
    k = _strip_prefix(k, "params/")
    k = _strip_prefix(k, "target/")
    k = _strip_prefix(k, "opt_state/")
    # Unify separators
    k = k.replace("\\", "/")
    return k


def _is_linear_weight(k: str) -> bool:
    return k.endswith(_LINEAR_SUFFIXES)


def _is_bias(k: str) -> bool:
    return k.endswith(_BIAS_SUFFIXES)


def _is_scale(k: str) -> bool:
    return k.endswith(_SCALE_SUFFIXES)


def _is_embedding(k: str) -> bool:
    return k.endswith(_EMBED_SUFFIXES)


# --------------------------- JAX -> Torch key mapping ---------------------------

_BLOCK_PATTERNS = (
    re.compile(r"(?:^|.*/)(?:Block|TransformerBlock)_(?P<idx>\d+)/(?P<name>.+)$"),
)


def _map_block_subkey(idx: int, name: str) -> Optional[str]:
    """Map a subkey inside a Block_i to a PyTorch state_dict key.

    Returns None if we don't recognize the pattern.
    """
    # Norms
    if name in ("rms1/scale", "rms_1/scale", "ln1/scale"):
        return f"blocks.{idx}.rms1.weight"
    if name in ("rms2/scale", "rms_2/scale", "ln2/scale"):
        return f"blocks.{idx}.rms2.weight"

    # Attention projections
    if name.endswith("q_proj/kernel"):
        return f"blocks.{idx}.attn.q_proj.weight"
    if name.endswith("k_proj/kernel"):
        return f"blocks.{idx}.attn.k_proj.weight"
    if name.endswith("v_proj/kernel"):
        return f"blocks.{idx}.attn.v_proj.weight"
    if name.endswith("o_proj/kernel") or name.endswith("out_proj/kernel"):
        return f"blocks.{idx}.attn.o_proj.weight"

    # FFN (gated: fc1 splits into u and v in forward, so a single weight here)
    if name.endswith("fc1/kernel"):
        return f"blocks.{idx}.fc1.weight"
    if name.endswith("fc1/bias"):
        return f"blocks.{idx}.fc1.bias"
    if name.endswith("fc2/kernel"):
        return f"blocks.{idx}.fc2.weight"
    if name.endswith("fc2/bias"):
        return f"blocks.{idx}.fc2.bias"

    return None


def _map_global_key(k: str) -> Optional[str]:
    """Map non-block keys (e.g., embedding) to PyTorch names."""
    # Embedding variants
    if re.search(r"(?:^|.*/)(?:Embed(?:_0)?|embed|token_embed)(?:/)?embedding$", k):
        return "embed.weight"
    return None


def _transpose_if_needed(t: np.ndarray, torch_key: str) -> np.ndarray:
    # PyTorch Linear expects (out, in). Flax Dense kernel is (in, out).
    if any(torch_key.endswith(s) for s in (".weight",)) and t.ndim == 2:
        # Heuristic: only transpose for weights that correspond to Linear layers, not for embedding.
        if ".attn." in torch_key or ".fc" in torch_key:
            return t.T
    return t


def _np_to_torch_tensor(arr: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        pass
    else:
        # In some JAX saves values are object wrappers; try to coerce
        arr = np.array(arr)
    # JAX sometimes writes bfloat16 as uint16-encoded — assume float32 fallback
    if arr.dtype == np.dtype("float16"):
        arr = arr.view(np.uint16).astype(np.float32)
    if dtype is None:
        return torch.from_numpy(arr.copy())
    return torch.from_numpy(arr.copy()).to(dtype=dtype)


def load_jax_npz_to_torch_state_dict(
    npz_path: str,
    model: Optional[torch.nn.Module] = None,
    *,
    default_dtype: Optional[torch.dtype] = torch.float32,
    verbose: bool = True,
) -> "OrderedDict[str, torch.Tensor]":
    """Translate a JAX/Flax .npz checkpoint into a PyTorch state_dict.

    If `model` is provided, we align dtypes with each parameter and report missing/unexpected keys.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Build a dtype map from model if available
    param_dtype_map: Dict[str, torch.dtype] = {}
    if model is not None:
        for name, p in model.named_parameters():
            param_dtype_map[name] = p.dtype

    state = OrderedDict()
    used = set()

    # First pass: block keys
    for raw_key in data.files:
        k = _normalize_key(raw_key)
        v = data[raw_key]
        # Block mapping
        mapped = False
        for pat in _BLOCK_PATTERNS:
            m = pat.match(k)
            if m:
                idx = int(m.group("idx"))
                sub = m.group("name")
                torch_key = _map_block_subkey(idx, sub)
                if torch_key is not None:
                    arr = v
                    arr = _transpose_if_needed(arr, torch_key)
                    dtype = param_dtype_map.get(torch_key, default_dtype)
                    state[torch_key] = _np_to_torch_tensor(arr, dtype=dtype)
                    used.add(raw_key)
                    mapped = True
                break
        if mapped:
            continue
        # Global keys (embedding etc.)
        gkey = _map_global_key(k)
        if gkey is not None:
            dtype = param_dtype_map.get(gkey, default_dtype)
            state[gkey] = _np_to_torch_tensor(v, dtype=dtype)
            used.add(raw_key)

    if verbose:
        unmatched = [rk for rk in data.files if rk not in used]
        if unmatched:
            print("[checkpoint_torch_io] Warning: Unmatched JAX keys (ignored):")
            for u in unmatched:
                print("  -", u)

    # Optional: sanity check vs model
    if model is not None and verbose:
        msd = model.state_dict()
        missing = [k for k in msd.keys() if k not in state]
        unexpected = [k for k in state.keys() if k not in msd]
        if missing:
            print("[checkpoint_torch_io] Missing keys for model (after translation):")
            for k in missing:
                print("  -", k)
        if unexpected:
            print("[checkpoint_torch_io] Unexpected translated keys not in model:")
            for k in unexpected:
                print("  -", k)

    return state


def save_torch_state_dict_as_npz(state_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    """(Optional) Save a PyTorch state_dict back to a JAX-friendly .npz.

    We invert the key mapping and transpose Linear weights back to (in, out).
    Only a subset of keys are supported (embed, blocks.*.{attn,fc*,rms*}).
    """
    arrays: Dict[str, np.ndarray] = {}

    for k, t in state_dict.items():
        if k == "embed.weight":
            arrays["Embed_0/embedding"] = t.detach().cpu().numpy()
            continue
        m = re.match(r"blocks\.(\d+)\.(.+)$", k)
        if not m:
            continue
        idx = int(m.group(1))
        tail = m.group(2)

        def add(npz_key: str, arr: np.ndarray):
            arrays[npz_key] = arr

        if tail == "rms1.weight":
            add(f"Block_{idx}/rms1/scale", t.detach().cpu().numpy())
        elif tail == "rms2.weight":
            add(f"Block_{idx}/rms2/scale", t.detach().cpu().numpy())
        elif tail == "attn.q_proj.weight":
            add(f"Block_{idx}/q_proj/kernel", t.detach().cpu().numpy().T)
        elif tail == "attn.k_proj.weight":
            add(f"Block_{idx}/k_proj/kernel", t.detach().cpu().numpy().T)
        elif tail == "attn.v_proj.weight":
            add(f"Block_{idx}/v_proj/kernel", t.detach().cpu().numpy().T)
        elif tail == "attn.o_proj.weight":
            add(f"Block_{idx}/o_proj/kernel", t.detach().cpu().numpy().T)
        elif tail == "fc1.weight":
            add(f"Block_{idx}/fc1/kernel", t.detach().cpu().numpy().T)
        elif tail == "fc1.bias":
            add(f"Block_{idx}/fc1/bias", t.detach().cpu().numpy())
        elif tail == "fc2.weight":
            add(f"Block_{idx}/fc2/kernel", t.detach().cpu().numpy().T)
        elif tail == "fc2.bias":
            add(f"Block_{idx}/fc2/bias", t.detach().cpu().numpy())

    np.savez_compressed(out_path, **arrays)


# ------------------------------- high-level loader ------------------------------

def load_any_checkpoint(model: torch.nn.Module, path: str, *, device: Optional[torch.device] = None) -> None:
    """Load either a native PyTorch state_dict (.pt/.pth) or a JAX .npz into the model."""
    if device is None:
        device = next(model.parameters()).device

    if path.endswith((".pt", ".pth")):
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        return

    if path.endswith(".npz"):
        state = load_jax_npz_to_torch_state_dict(path, model=model, verbose=True)
        model.load_state_dict(state, strict=True)
        return

    raise ValueError(f"Unsupported checkpoint extension: {path}")


# ------------------------------------- CLI -------------------------------------

def _cmd_translate(args):
    state = load_jax_npz_to_torch_state_dict(args.input, model=None, verbose=not args.quiet)
    if args.dry_run:
        print("[checkpoint_torch_io] Dry-run complete. Keys translated:")
        for k in state.keys():
            print("  -", k)
        return
    torch.save(state, args.output)
    if not args.quiet:
        print(f"[checkpoint_torch_io] Wrote PyTorch state_dict to {args.output}")


def main():
    p = argparse.ArgumentParser(description="Translate JAX/Flax .npz checkpoints to PyTorch state_dicts")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="Convert .npz to .pt state_dict")
    t.add_argument("--in", dest="input", required=True, help="Input JAX .npz file")
    t.add_argument("--out", dest="output", required=False, default="checkpoint_torch.pt", help="Output .pt file")
    t.add_argument("--dry-run", action="store_true", help="Only print the translated keys")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=_cmd_translate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

