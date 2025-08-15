from  __future__ import annotations

"""
checkpoint_torch_io.py — Translate JAX/Flax checkpoints to PyTorch and load them.

Usage (CLI):
  python checkpoint_torch_io.py translate --in ckpt_jax.npz --out ckpt_torch.pt
  python checkpoint_torch_io.py translate --in ckpt_jax.npz --dry-run

Library API:
  from checkpoint_torch_io import load_jax_npz_to_torch_state_dict, load_any_checkpoint
  state = load_jax_npz_to_torch_state_dict("ckpt_jax.npz", model)
  model.load_state_dict(state, strict=True)

This version includes broader regexes to handle varied Flax naming styles
(e.g., TinyTransformerBlock_0/NativeJaxSelfAttention_0/q_proj/kernel).
"""

import argparse
import re
from collections import OrderedDict
from typing import Dict, Tuple, Optional

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

import numpy as np
import torch

# ------------------------------ helpers ------------------------------


def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix) :] if k.startswith(prefix) else k


def _normalize_key(k: str) -> str:
    k = _strip_prefix(k, "params/")
    k = _strip_prefix(k, "target/")
    k = _strip_prefix(k, "opt_state/")
    return k.replace("\\", "/")


def _np_to_torch_tensor(arr: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    #if arr.dtype == np.dtype(Config.compute_dtype):
    if arr.dtype == np.dtype("float16"):
        arr = arr.view(np.uint16).astype(np.float32)
    t = torch.from_numpy(arr.copy())
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


# --------------------------- key mapping ---------------------------

_BLOCK_INDEX_RE = re.compile(
    r"(?:(?:Block|Transformer_block|TransformerBlock|TinyTransformerBlock|layer|layers|h)_(?P<idx>\d+))"
)
_EMBED_RE = re.compile(r"(?:^|.*/)(?:Embed(?:_0)?|embed|token_embed)(?:/)?embedding$")

_DEF_MAP = {
    "attn.q_proj.weight": re.compile(r"/(?:q_proj)/(?:kernel)$"),
    "attn.k_proj.weight": re.compile(r"/(?:k_proj)/(?:kernel)$"),
    "attn.v_proj.weight": re.compile(r"/(?:v_proj)/(?:kernel)$"),
    "attn.o_proj.weight": re.compile(r"/(?:o_proj|out_proj)/(?:kernel)$"),
    "fc1.weight": re.compile(r"/(?:fc1)/(?:kernel)$"),
    "fc1.bias": re.compile(r"/(?:fc1)/(?:bias)$"),
    "fc2.weight": re.compile(r"/(?:fc2)/(?:kernel)$"),
    "fc2.bias": re.compile(r"/(?:fc2)/(?:bias)$"),
}

_NORM1_RE = re.compile(r"/(?:rms1|rms_1|ln1|norm1|input_layernorm)/(?:scale)$")
_NORM2_RE = re.compile(r"/(?:rms2|rms_2|ln2|norm2|post_attention_layernorm)/(?:scale)$")


def _find_block_index(k: str) -> Optional[int]:
    last = None
    for m in _BLOCK_INDEX_RE.finditer(k):
        last = m
    return int(last.group("idx")) if last else None


def _transpose_if_needed(arr: np.ndarray, torch_key: str) -> np.ndarray:
    if arr.ndim == 2 and torch_key.endswith(".weight") and (".attn." in torch_key or ".fc" in torch_key):
        return arr.T
    return arr


def _map_one_key(raw_key: str) -> Optional[Tuple[str, str, Optional[int]]]:
    k = _normalize_key(raw_key)
    if _EMBED_RE.search(k):
        return ("embed", "embed.weight", None)

    idx = _find_block_index(k)
    if idx is None:
        return None

    if _NORM1_RE.search(k):
        return ("block", f"blocks.{idx}.rms1.weight", idx)
    if _NORM2_RE.search(k):
        return ("block", f"blocks.{idx}.rms2.weight", idx)

    for name, pat in _DEF_MAP.items():
        if pat.search(k):
            return ("block", f"blocks.{idx}.{name}", idx)

    return None


def load_jax_npz_to_torch_state_dict(
    npz_path: str,
    model: Optional[torch.nn.Module] = None,
    *,
    default_dtype: Optional[torch.dtype] = torch.float32,
    verbose: bool = True,
) -> "OrderedDict[str, torch.Tensor]":
    data = np.load(npz_path, allow_pickle=True)

    param_dtype_map: Dict[str, torch.dtype] = {}
    if model is not None:
        for name, p in model.named_parameters():
            param_dtype_map[name] = p.dtype

    state = OrderedDict()
    used = set()

    for raw_key in data.files:
        mapping = _map_one_key(raw_key)
        if mapping is None:
            continue
        _, torch_key, _ = mapping
        arr = _transpose_if_needed(data[raw_key], torch_key)
        dtype = param_dtype_map.get(torch_key, default_dtype)
        state[torch_key] = _np_to_torch_tensor(arr, dtype=dtype)
        used.add(raw_key)

    if "embed.weight" not in state:
        for raw_key in data.files:
            k = _normalize_key(raw_key)
            if k.endswith("/embedding") or k.endswith(".embedding"):
                state["embed.weight"] = _np_to_torch_tensor(
                    data[raw_key], dtype=param_dtype_map.get("embed.weight", default_dtype)
                )
                used.add(raw_key)
                break

    if verbose:
        total = len(data.files)
        print(f"[checkpoint_torch_io] translated {len(used)}/{total} arrays into {len(state)} torch params")
        unmatched = [rk for rk in data.files if rk not in used]
        if unmatched:
            print("[checkpoint_torch_io] Warning: unmatched JAX keys (ignored):")
            for u in unmatched[:50]:
                print("  -", u)
            if len(unmatched) > 50:
                print(f"  ... and {len(unmatched)-50} more")

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


def load_any_checkpoint(model: torch.nn.Module, path: str, *, device: Optional[torch.device] = None) -> None:
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

