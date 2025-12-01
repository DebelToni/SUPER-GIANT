#!/usr/bin/env python
"""
Download a Gemma-2-based BgGPT checkpoint from Hugging Face and convert it to
the JAX/Flax layout used by gemma-bg/GiantGPT.

Example (from repo root):

    python gemma-bg/download_and_convert_gemma_bg.py \
      --hf-repo INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 \
      --out checkpoints/bggpt-gemma-2-2.6b-it.npz

Default output is <data_root>/gemma-bg/bggpt-gemma-2-2.6b-it.npz when
Global_Config.yml is present. The script expects a *PyTorch/Transformers*
checkpoint (not GGUF); if you only have a GGUF repo, point --hf-repo at the
matching Transformers weights instead.
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

from GiantGPT import GiantGPT, MODEL_CFG  # gemma-bg GiantGPT/Config


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


def _rms_scale_from_gemma2(weight_tensor: "torch.Tensor") -> np.ndarray:
    """Map Gemma2RMSNorm.weight -> Flax RMSNorm.scale (scale = 1 + weight)."""
    w = _to_numpy(weight_tensor)
    return 1.0 + w


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_CFG_PATH = PROJECT_ROOT / "Global_Config.yml"


def build_flax_skeleton(
    vocab_size: int,
    context_length: int,
) -> Tuple[GiantGPT, Dict]:
    """
    Instantiate GiantGPT with Gemma/BgGPT config and initialize params
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
    # Store as float16 to cut parameter footprint; computation will upcast as needed.
    return t.detach().cpu().numpy().astype(np.float16)


def _find_attn_submodule(block_params: Dict) -> str:
    """
    In each TinyTransformerBlock, find the submodule that contains qkv_proj/o_proj.
    This makes us robust to the exact internal naming (NativeJaxSelfAttention_0, etc.).
    """
    for name, sub in block_params.items():
        if isinstance(sub, dict) and "qkv_proj" in sub:
            return name
    raise KeyError("Could not find attention submodule with 'qkv_proj' in block params")


def convert_gemma_to_flax(
    hf_repo: str,
    out_path: Path,
    revision: str | None = None,
):
    # 1. Load HF model (CPU, full precision)
    print(f"[HF] Loading model: {hf_repo}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        revision=revision,
        torch_dtype=torch.float32,
        device_map=None,
    )
    hf_model.eval()
    hf_cfg = hf_model.config
    sd = hf_model.state_dict()

    # 2. Read config & sanity-check against gemma-bg Config.yml
    vocab_size = int(hf_cfg.vocab_size)
    ctx_len = int(getattr(hf_cfg, "max_position_embeddings", hf_cfg.max_position_ids if hasattr(hf_cfg, "max_position_ids") else MODEL_CFG.context_length))
    target_ctx = int(MODEL_CFG.context_length)
    hidden_size = int(hf_cfg.hidden_size)
    n_heads = int(hf_cfg.num_attention_heads)
    n_kv = int(hf_cfg.num_key_value_heads)
    n_layers = int(hf_cfg.num_hidden_layers)
    intermediate_size = int(hf_cfg.intermediate_size)
    head_dim = int(getattr(hf_cfg, "head_dim", hidden_size // n_heads))
    attn_qkv_dim = n_heads * head_dim

    print("[HF] Config:")
    print(f"  vocab_size          = {vocab_size}")
    print(f"  hidden_size         = {hidden_size}")
    print(f"  num_attention_heads = {n_heads}")
    print(f"  num_key_value_heads = {n_kv}")
    print(f"  head_dim            = {head_dim}")
    print(f"  attn_qkv_dim        = {attn_qkv_dim}")
    print(f"  num_hidden_layers   = {n_layers}")
    print(f"  intermediate_size   = {intermediate_size}")
    print(f"  max_position_embeds = {ctx_len}")
    print(f"  target_context_len  = {target_ctx}")

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
    if ctx_len < target_ctx:
        raise ValueError(f"HF context_length ({ctx_len}) is smaller than target cfg ({target_ctx}).")
    assert attn_qkv_dim == MODEL_CFG.attn_qkv_dim, \
        f"Config mismatch: attn_qkv_dim={attn_qkv_dim} vs cfg.attn_qkv_dim={MODEL_CFG.attn_qkv_dim}"

    # 3. Build Flax skeleton
    print("[JAX] Building GiantGPT skeleton...")
    _, params = build_flax_skeleton(vocab_size, target_ctx)

    # 4. Map weights

    # 4a. Embeddings (tied)
    print("[MAP] Embeddings...")
    params["Embed_0"]["embedding"] = _to_numpy(sd["model.embed_tokens.weight"])

    # 4b. Per-layer mapping
    print("[MAP] Transformer blocks...")
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

        # Norms (Gemma2RMSNorm: scale = 1 + weight)
        block["rms1"]["scale"] = _rms_scale_from_gemma2(sd[prefix + "input_layernorm.weight"])
        block["rms_post_attn"]["scale"] = _rms_scale_from_gemma2(sd[prefix + "post_attention_layernorm.weight"])
        block["rms_pre_ff"]["scale"] = _rms_scale_from_gemma2(sd[prefix + "pre_feedforward_layernorm.weight"])
        block["rms_post_ff"]["scale"] = _rms_scale_from_gemma2(sd[prefix + "post_feedforward_layernorm.weight"])

        # Attention submodule
        attn_name = _find_attn_submodule(block)
        attn = block[attn_name]

        # QKV: pack [q | k | v] along the output-dim (last axis after transpose)
        q_w = _to_numpy(sd[prefix + "self_attn.q_proj.weight"]).T  # (hidden, q_dim)
        k_w = _to_numpy(sd[prefix + "self_attn.k_proj.weight"]).T  # (hidden, kv_dim)
        v_w = _to_numpy(sd[prefix + "self_attn.v_proj.weight"]).T  # (hidden, kv_dim)
        attn["qkv_proj"]["kernel"] = np.concatenate([q_w, k_w, v_w], axis=1)

        # O-proj (2048 -> 2304)
        attn["o_proj"]["kernel"] = _to_numpy(sd[prefix + "self_attn.o_proj.weight"]).T

        # MLP: GeGLU (gate_proj, up_proj, down_proj)
        gate = _to_numpy(sd[prefix + "mlp.gate_proj.weight"]).T  # (hidden, d_ff)
        up   = _to_numpy(sd[prefix + "mlp.up_proj.weight"]).T    # (hidden, d_ff)
        down = _to_numpy(sd[prefix + "mlp.down_proj.weight"]).T  # (d_ff, hidden)

        block["fc1"]["kernel"] = np.concatenate([gate, up], axis=1)  # (hidden, 2*d_ff)
        block["fc2"]["kernel"] = down  # (d_ff, hidden)

    print("[MAP] Final RMSNorm...")
    params["final_norm"]["scale"] = _rms_scale_from_gemma2(sd["model.norm.weight"])

    # 6. Save as NPZ with flattened keys
    print(f"[SAVE] Writing NPZ to {out_path} ...")
    flat = traverse_util.flatten_dict(params, sep="/")
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
                return str(Path(data_root).expanduser() / "gemma-fp16" / "bggpt-gemma-2-2.6b-it-fp16.npz")
        return "checkpoints/bggpt-gemma-2-2.6b-it-fp16.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
        help="Hugging Face repo id for the Gemma-2/BgGPT checkpoint (Transformers weights, not GGUF).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision/tag.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=_default_out(),
        help="Output .npz path (will be created). Defaults to <data_root>/gemma-bg/bggpt-gemma-2-2.6b-it.npz when Global_Config.yml is present.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert_gemma_to_flax(args.hf_repo, out_path, revision=args.revision)


if __name__ == "__main__":
    main()
