"""
Convert Qwen2.5-0.5B HF weights to our JAX/Flax param tree (.npz).
Requires HF token (export HF_TOKEN) to download the gated model.
"""
from __future__ import annotations

import os
from pathlib import Path
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def to_numpy(t: torch.Tensor, transpose: bool = False) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if transpose:
        arr = arr.T
    return arr


def bias_or_zeros(sd, key, out_dim):
    if key in sd:
        return to_numpy(sd[key])
    return np.zeros((out_dim,), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen HF weights to JAX npz.")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id")
    parser.add_argument("--out_dir", default="/Volumes/SSD/r2/qwen25_0.5b_instruct", help="Output directory for npz")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        use_auth_token=os.environ.get("HF_TOKEN"),
    )
    sd = model.state_dict()
    cfg = model.config

    kv = {}

    # Embeddings / head
    kv[("Embed_0", "embedding")] = to_numpy(sd["model.embed_tokens.weight"])
    kv[("norm", "scale")] = to_numpy(sd["model.norm.weight"])
    kv[("lm_head", "weight")] = to_numpy(sd["lm_head.weight"])

    n_layers = cfg.num_hidden_layers
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        base = (f"QwenBlock_{i}",)

        kv[base + ("rms1", "scale")] = to_numpy(sd[prefix + "input_layernorm.weight"])
        kv[base + ("rms2", "scale")] = to_numpy(sd[prefix + "post_attention_layernorm.weight"])

        kv[base + ("attn", "q_proj", "kernel")] = to_numpy(sd[prefix + "self_attn.q_proj.weight"], transpose=True)
        kv[base + ("attn", "k_proj", "kernel")] = to_numpy(sd[prefix + "self_attn.k_proj.weight"], transpose=True)
        kv[base + ("attn", "v_proj", "kernel")] = to_numpy(sd[prefix + "self_attn.v_proj.weight"], transpose=True)
        kv[base + ("attn", "o_proj", "kernel")] = to_numpy(sd[prefix + "self_attn.o_proj.weight"], transpose=True)

        q_dim = sd[prefix + "self_attn.q_proj.weight"].shape[0]
        k_dim = sd[prefix + "self_attn.k_proj.weight"].shape[0]
        v_dim = sd[prefix + "self_attn.v_proj.weight"].shape[0]
        o_dim = sd[prefix + "self_attn.o_proj.weight"].shape[0]

        kv[base + ("attn", "q_proj", "bias")] = bias_or_zeros(sd, prefix + "self_attn.q_proj.bias", q_dim)
        kv[base + ("attn", "k_proj", "bias")] = bias_or_zeros(sd, prefix + "self_attn.k_proj.bias", k_dim)
        kv[base + ("attn", "v_proj", "bias")] = bias_or_zeros(sd, prefix + "self_attn.v_proj.bias", v_dim)
        kv[base + ("attn", "o_proj", "bias")] = bias_or_zeros(sd, prefix + "self_attn.o_proj.bias", o_dim)

        kv[base + ("mlp", "gate_proj", "kernel")] = to_numpy(sd[prefix + "mlp.gate_proj.weight"], transpose=True)
        kv[base + ("mlp", "up_proj", "kernel")] = to_numpy(sd[prefix + "mlp.up_proj.weight"], transpose=True)
        kv[base + ("mlp", "down_proj", "kernel")] = to_numpy(sd[prefix + "mlp.down_proj.weight"], transpose=True)

    out_dir = Path(os.environ.get("QWEN_OUT_DIR", args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.model_id.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{name}.npz"
    np.savez(out_path, **{"/".join(k): v for k, v in kv.items()})
    print(f"Saved {out_path} with {len(kv)} tensors")


if __name__ == "__main__":
    main()
