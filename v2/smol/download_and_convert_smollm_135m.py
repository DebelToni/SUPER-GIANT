#!/usr/bin/env python
"""
Download HuggingFaceTB/SmolLM-135M and convert its weights to a JAX .npz
matching GiantGPT + TinyTransformerBlock + NativeJaxSelfAttention layout.

Usage (from project root or model dir):

    python download_and_convert_smollm_135m.py \
      --hf-repo HuggingFaceTB/SmolLM-135M \
      --out checkpoints/smollm-135m.npz

Defaults to writing the NPZ to <data_root>/smol/smollm-135m.npz when
Global_Config.yml is present. Then run:

    python Generate_faster.py --checkpoint <path>/smollm-135m.npz --prompt "..." ...

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax import traverse_util
from omegaconf import OmegaConf

from v2.smol.GiantGPT import GiantGPT, MODEL_CFG  # uses your updated Config.yml


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_CFG_PATH = PROJECT_ROOT / "Global_Config.yml"


def build_flax_skeleton(
    vocab_size: int,
    context_length: int,
) -> Tuple[GiantGPT, Dict]:
    """
    Instantiate GiantGPT with SmolLM-135M config and initialize params
    so we get the exact param tree structure & names.
    """
    model = GiantGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=int(MODEL_CFG.embedding_size),
        n_heads=int(MODEL_CFG.num_heads),
        d_ff=int(MODEL_CFG.feed_forward_size),
        n_layers=int(MODEL_CFG.num_layers),
        dropout_rate=float(MODEL_CFG.dropout_rate),
    )

    key_params, key_dropout = jax.random.split(jax.random.PRNGKey(0), 2)
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)

    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy,
        deterministic=True,
        use_kv_cache=False,
        cur_index=0,
    )
    params = unfreeze(variables["params"])
    return model, params


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    # Store everything as float32; JAX will cast as needed.
    return t.detach().cpu().numpy().astype(np.float32)


def _find_attn_submodule(block_params: Dict) -> str:
    """
    In each TinyTransformerBlock, find the submodule that contains qkv_proj/o_proj.
    This makes us robust to the exact internal naming (NativeJaxSelfAttention_0, etc.).
    """
    for name, sub in block_params.items():
        if isinstance(sub, dict) and "qkv_proj" in sub:
            return name
    raise KeyError("Could not find attention submodule with 'qkv_proj' in block params")


def convert_smollm_to_flax(
    hf_repo: str,
    out_path: Path,
):
    # 1. Load HF model (CPU, full precision)
    print(f"[HF] Loading model: {hf_repo}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        torch_dtype=torch.float32,
        device_map=None,
    )
    hf_model.eval()
    hf_cfg = hf_model.config
    sd = hf_model.state_dict()

    # 2. Read SmolLM config & sanity-check against your Config.yml
    vocab_size = int(hf_cfg.vocab_size)
    ctx_len = int(hf_cfg.max_position_embeddings)
    hidden_size = int(hf_cfg.hidden_size)
    n_heads = int(hf_cfg.num_attention_heads)
    n_kv = int(hf_cfg.num_key_value_heads)
    n_layers = int(hf_cfg.num_hidden_layers)
    intermediate_size = int(hf_cfg.intermediate_size)

    print("[HF] Config:")
    print(f"  vocab_size          = {vocab_size}")
    print(f"  hidden_size         = {hidden_size}")
    print(f"  num_attention_heads = {n_heads}")
    print(f"  num_key_value_heads = {n_kv}")
    print(f"  num_hidden_layers   = {n_layers}")
    print(f"  intermediate_size   = {intermediate_size}")
    print(f"  max_position_embeds = {ctx_len}")

    # Cross-check with your config
    assert hidden_size == MODEL_CFG.embedding_size, \
        f"Config mismatch: hidden_size={hidden_size} vs embedding_size={MODEL_CFG.embedding_size}"
    assert n_heads == MODEL_CFG.num_heads, \
        f"Config mismatch: num_heads={n_heads} vs cfg.num_heads={MODEL_CFG.num_heads}"
    assert n_kv == MODEL_CFG.num_kv_heads, \
        f"Config mismatch: num_kv_heads={n_kv} vs cfg.num_kv_heads={MODEL_CFG.num_kv_heads}"
    assert n_layers == MODEL_CFG.num_layers, \
        f"Config mismatch: num_layers={n_layers} vs cfg.num_layers={MODEL_CFG.num_layers}"
    assert intermediate_size == MODEL_CFG.feed_forward_size, \
        f"Config mismatch: intermediate_size={intermediate_size} vs cfg.feed_forward_size={MODEL_CFG.feed_forward_size}"
    assert ctx_len == MODEL_CFG.context_length, \
        f"Config mismatch: max_position_embeddings={ctx_len} vs cfg.context_length={MODEL_CFG.context_length}"

    # 3. Build Flax skeleton
    print("[JAX] Building GiantGPT skeleton...")
    _, params = build_flax_skeleton(vocab_size, ctx_len)

    # 4. Map weights

    # 4a. Embeddings (tied)
    print("[MAP] Embeddings...")
    params["Embed_0"]["embedding"] = _to_numpy(sd["model.embed_tokens.weight"])

    # 4b. Final RMSNorm
    print("[MAP] Final RMSNorm (model.norm.weight -> final_norm/scale)...")
    if "final_norm" not in params:
        raise KeyError("Expected 'final_norm' module in GiantGPT; check GiantGPT.py edits.")
    params["final_norm"]["scale"] = _to_numpy(sd["model.norm.weight"])

    # 4c. Per-layer mapping
    print("[MAP] Transformer blocks...")
    # Find TinyTransformerBlock_* in order
    block_names = sorted(
        [k for k in params.keys() if k.startswith("TinyTransformerBlock_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if len(block_names) != n_layers:
        raise ValueError(
            f"Found {len(block_names)} TinyTransformerBlock_* in params, "
            f"but HF reports {n_layers} layers."
        )

    for layer_idx, block_name in enumerate(block_names):
        print(f"  Layer {layer_idx} -> {block_name}")
        block = params[block_name]
        prefix = f"model.layers.{layer_idx}."

        # Norms
        block["rms1"]["scale"] = _to_numpy(sd[prefix + "input_layernorm.weight"])
        block["rms2"]["scale"] = _to_numpy(sd[prefix + "post_attention_layernorm.weight"])

        # Attention submodule
        attn_name = _find_attn_submodule(block)
        attn = block[attn_name]

        # QKV: pack [q | k | v] along the output-dim (last axis after transpose)
        q_w = _to_numpy(sd[prefix + "self_attn.q_proj.weight"]).T  # (hidden, hidden)
        k_w = _to_numpy(sd[prefix + "self_attn.k_proj.weight"]).T  # (hidden, kv_dim)
        v_w = _to_numpy(sd[prefix + "self_attn.v_proj.weight"]).T  # (hidden, kv_dim)
        attn["qkv_proj"]["kernel"] = np.concatenate([q_w, k_w, v_w], axis=1)

        # O-proj
        attn["o_proj"]["kernel"] = _to_numpy(sd[prefix + "self_attn.o_proj.weight"]).T  # (hidden, hidden)

        # MLP: SwiGLU
        # gate_proj: (intermediate, hidden) -> (hidden, intermediate) after .T
        # up_proj:   (intermediate, hidden) -> (hidden, intermediate)
        # down_proj: (hidden, intermediate) -> (intermediate, hidden) after .T
        gate = _to_numpy(sd[prefix + "mlp.gate_proj.weight"]).T  # (hidden, d_ff)
        up   = _to_numpy(sd[prefix + "mlp.up_proj.weight"]).T    # (hidden, d_ff)
        down = _to_numpy(sd[prefix + "mlp.down_proj.weight"]).T  # (d_ff, hidden)

        # Your fc1 projects to 2 * d_ff and then splits (u, v) for SwiGLU
        block["fc1"]["kernel"] = np.concatenate([gate, up], axis=1)  # (hidden, 2*d_ff)
        block["fc2"]["kernel"] = down  # (d_ff, hidden)

    # 5. Save as NPZ with flattened keys
    print(f"[SAVE] Writing NPZ to {out_path} ...")
    flat = traverse_util.flatten_dict(params, sep="/")
    # Ensure everything is plain numpy
    flat_np = {k: np.asarray(v) for k, v in flat.items()}
    np.savez(out_path, **flat_np)
    print("[DONE] Conversion complete.")


def main():
    def _default_out() -> str:
        if GLOBAL_CFG_PATH.exists():
            cfg = OmegaConf.load(GLOBAL_CFG_PATH)
            paths = cfg.get("paths", {}) if hasattr(cfg, "get") else {}
            data_root = paths.get("data_root")
            if data_root:
                return str(Path(data_root).expanduser() / "smol" / "smollm-135m.npz")
        return "checkpoints/smollm-135m.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Hugging Face repo id for SmolLM-135M.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=_default_out(),
        help="Output .npz path (will be created). Defaults to <data_root>/smol/smollm-135m.npz when Global_Config.yml is present.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert_smollm_to_flax(args.hf_repo, out_path)


if __name__ == "__main__":
    main()
