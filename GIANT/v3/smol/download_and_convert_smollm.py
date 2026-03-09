#!/usr/bin/env python
"""
Download and convert SmolLM/SmolLM2 models to JAX .npz format.

Supports:
- SmolLM-135M (Config_135m.yml)
- SmolLM2-360M (Config_360m.yml)

Usage:
    # Default: SmolLM-135M
    python download_and_convert_smollm.py

    # Explicit 135M:
    python download_and_convert_smollm.py --config Config_135m.yml

    # 360M:
    python download_and_convert_smollm.py --config Config_360m.yml

    # Custom output path:
    python download_and_convert_smollm.py --config Config_360m.yml --out /custom/path.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from flax import traverse_util
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    embedding_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    feed_forward_size: int
    rope_dim: int
    context_length: int
    dropout_rate: float
    param_dtype: str = "float32"
    compute_dtype: str = "float32"


@dataclass
class DownloadConfig:
    hf_repo: str
    output_filename: str


def load_config(config_path: Path) -> tuple[ModelConfig, DownloadConfig, OmegaConf]:
    """Load and parse a model config file."""
    cfg = OmegaConf.load(config_path)
    model_cfg = ModelConfig(
        embedding_size=int(cfg.model.embedding_size),
        num_heads=int(cfg.model.num_heads),
        num_kv_heads=int(cfg.model.num_kv_heads),
        num_layers=int(cfg.model.num_layers),
        feed_forward_size=int(cfg.model.feed_forward_size),
        rope_dim=int(cfg.model.rope_dim),
        context_length=int(cfg.model.context_length),
        dropout_rate=float(cfg.model.dropout_rate),
        param_dtype=str(cfg.model.get("param_dtype", "float32")),
        compute_dtype=str(cfg.model.get("compute_dtype", "float32")),
    )
    download_cfg = DownloadConfig(
        hf_repo=str(cfg.download.hf_repo),
        output_filename=str(cfg.download.output_filename),
    )
    return model_cfg, download_cfg, cfg


# ---------------------------------------------------------------------------
# Direct param dict construction (no JAX model init needed)
# ---------------------------------------------------------------------------

def build_empty_params(model_cfg: ModelConfig, vocab_size: int) -> Dict:
    """
    Build an empty params dict matching GiantGPT structure.
    
    Structure:
    - Embed_0/embedding: [vocab_size, d_model]
    - final_norm/scale: [d_model]
    - TinyTransformerBlock_{i}/rms1/scale: [d_model]
    - TinyTransformerBlock_{i}/rms2/scale: [d_model]
    - TinyTransformerBlock_{i}/NativeJaxSelfAttention_0/qkv_proj/kernel: [d_model, q_dim + 2*kv_dim]
    - TinyTransformerBlock_{i}/NativeJaxSelfAttention_0/o_proj/kernel: [d_model, d_model]
    - TinyTransformerBlock_{i}/fc1/kernel: [d_model, 2*d_ff]
    - TinyTransformerBlock_{i}/fc2/kernel: [d_ff, d_model]
    """
    d = model_cfg.embedding_size
    n_heads = model_cfg.num_heads
    n_kv = model_cfg.num_kv_heads
    n_layers = model_cfg.num_layers
    d_ff = model_cfg.feed_forward_size
    head_dim = d // n_heads
    
    # QKV sizes
    q_dim = d  # n_heads * head_dim
    kv_dim = n_kv * head_dim
    qkv_out = q_dim + 2 * kv_dim
    
    params = {
        "Embed_0": {"embedding": None},
        "final_norm": {"scale": None},
    }
    
    for i in range(n_layers):
        params[f"TinyTransformerBlock_{i}"] = {
            "rms1": {"scale": None},
            "rms2": {"scale": None},
            "NativeJaxSelfAttention_0": {
                "qkv_proj": {"kernel": None},
                "o_proj": {"kernel": None},
            },
            "fc1": {"kernel": None},
            "fc2": {"kernel": None},
        }
    
    return params


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def convert_smollm_to_flax(
    model_cfg: ModelConfig,
    hf_repo: str,
    out_path: Path,
    hf_cache_dir: str | None = None,
):
    """Download and convert a SmolLM model to JAX npz format."""
    # 1. Load HF model
    print(f"[HF] Loading model: {hf_repo}")
    kwargs = {"torch_dtype": torch.float32, "device_map": None}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    hf_model = AutoModelForCausalLM.from_pretrained(hf_repo, **kwargs)
    hf_model.eval()
    hf_cfg = hf_model.config
    sd = hf_model.state_dict()

    # 2. Read HF config
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

    # Cross-check with our config
    assert hidden_size == model_cfg.embedding_size, \
        f"Config mismatch: hidden_size={hidden_size} vs embedding_size={model_cfg.embedding_size}"
    assert n_heads == model_cfg.num_heads, \
        f"Config mismatch: num_heads={n_heads} vs cfg.num_heads={model_cfg.num_heads}"
    assert n_kv == model_cfg.num_kv_heads, \
        f"Config mismatch: num_kv_heads={n_kv} vs cfg.num_kv_heads={model_cfg.num_kv_heads}"
    assert n_layers == model_cfg.num_layers, \
        f"Config mismatch: num_layers={n_layers} vs cfg.num_layers={model_cfg.num_layers}"
    assert intermediate_size == model_cfg.feed_forward_size, \
        f"Config mismatch: intermediate_size={intermediate_size} vs cfg.feed_forward_size={model_cfg.feed_forward_size}"

    # 3. Build params dict directly (no JAX model needed)
    print("[BUILD] Building param structure...")
    params = build_empty_params(model_cfg, vocab_size)

    # 4. Map weights

    # 4a. Embeddings (tied)
    print("[MAP] Embeddings...")
    params["Embed_0"]["embedding"] = _to_numpy(sd["model.embed_tokens.weight"])

    # 4b. Final RMSNorm
    print("[MAP] Final RMSNorm...")
    params["final_norm"]["scale"] = _to_numpy(sd["model.norm.weight"])

    # 4c. Per-layer mapping
    print("[MAP] Transformer blocks...")
    for layer_idx in range(n_layers):
        block_name = f"TinyTransformerBlock_{layer_idx}"
        print(f"  Layer {layer_idx} -> {block_name}")
        block = params[block_name]
        prefix = f"model.layers.{layer_idx}."

        # Norms
        block["rms1"]["scale"] = _to_numpy(sd[prefix + "input_layernorm.weight"])
        block["rms2"]["scale"] = _to_numpy(sd[prefix + "post_attention_layernorm.weight"])

        # Attention: QKV packed [q | k | v]
        attn = block["NativeJaxSelfAttention_0"]
        q_w = _to_numpy(sd[prefix + "self_attn.q_proj.weight"]).T
        k_w = _to_numpy(sd[prefix + "self_attn.k_proj.weight"]).T
        v_w = _to_numpy(sd[prefix + "self_attn.v_proj.weight"]).T
        attn["qkv_proj"]["kernel"] = np.concatenate([q_w, k_w, v_w], axis=1)

        # O-proj
        attn["o_proj"]["kernel"] = _to_numpy(sd[prefix + "self_attn.o_proj.weight"]).T

        # MLP: SwiGLU - fc1 = [gate | up], fc2 = down
        gate = _to_numpy(sd[prefix + "mlp.gate_proj.weight"]).T
        up = _to_numpy(sd[prefix + "mlp.up_proj.weight"]).T
        down = _to_numpy(sd[prefix + "mlp.down_proj.weight"]).T

        block["fc1"]["kernel"] = np.concatenate([gate, up], axis=1)
        block["fc2"]["kernel"] = down

    # 5. Save as NPZ
    print(f"[SAVE] Writing NPZ to {out_path} ...")
    flat = traverse_util.flatten_dict(params, sep="/")
    flat_np = {k: np.asarray(v) for k, v in flat.items()}
    np.savez(out_path, **flat_np)
    print("[DONE] Conversion complete.")
    
    # Print summary
    total_params = sum(v.size for v in flat_np.values())
    print(f"[INFO] Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    GLOBAL_CFG_PATH = PROJECT_ROOT / "Global_Config.yml"

    def get_default_paths(config_name: str | None) -> tuple[str | None, str]:
        """Get default HF cache dir and output path."""
        hf_cache = None
        out_path = "checkpoints/smollm.npz"
        
        if GLOBAL_CFG_PATH.exists():
            global_cfg = OmegaConf.load(GLOBAL_CFG_PATH)
            paths = global_cfg.get("paths", {})
            data_root = paths.get("data_root")
            hf_cache_root = paths.get("hf_cache_root")
            if hf_cache_root:
                hf_cache = str(Path(hf_cache_root).expanduser())
            if data_root and config_name:
                # Determine output filename from config
                config_path = SCRIPT_DIR / config_name
                if config_path.exists():
                    cfg = OmegaConf.load(config_path)
                    filename = cfg.download.get("output_filename", "smollm.npz")
                    out_path = str(Path(data_root).expanduser() / "TiDAR" / "smol" / filename)
        
        return hf_cache, out_path

    parser = argparse.ArgumentParser(description="Download and convert SmolLM models to JAX format")
    parser.add_argument(
        "--config",
        type=str,
        default="Config_135m.yml",
        help="Config file name in the smol folder (e.g., Config_135m.yml, Config_360m.yml)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Defaults to <data_root>/TiDAR/smol/<output_filename>",
    )
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="HuggingFace cache directory. Defaults to paths.hf_cache_root from Global_Config.yml",
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    model_cfg, download_cfg, raw_cfg = load_config(config_path)
    
    # Get default paths
    default_hf_cache, default_out = get_default_paths(args.config)
    
    hf_cache = args.hf_cache or default_hf_cache
    out_path = Path(args.out) if args.out else Path(default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[CONFIG] Using config: {config_path}")
    print(f"[CONFIG] HF repo: {download_cfg.hf_repo}")
    print(f"[CONFIG] HF cache: {hf_cache or 'default'}")
    print(f"[CONFIG] Output: {out_path}")
    
    convert_smollm_to_flax(
        model_cfg=model_cfg,
        hf_repo=download_cfg.hf_repo,
        out_path=out_path,
        hf_cache_dir=hf_cache,
    )


if __name__ == "__main__":
    main()
