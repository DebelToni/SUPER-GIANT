"""
Convert SmolLM HF weights to our fused-QKV JAX layout (.npz).
Defaults to HuggingFaceTB/SmolLM-135M.
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


def main():
    parser = argparse.ArgumentParser(description="Convert SmolLM HF weights to fused-QKV npz.")
    parser.add_argument("--model_id", default="HuggingFaceTB/SmolLM-135M", help="HF model id")
    parser.add_argument("--out_dir", default="/Volumes/SSD/r2/smol/SmolLM-135M", help="Output directory")
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

    kv[("Embed_0", "embedding")] = to_numpy(sd["model.embed_tokens.weight"])
    kv[("norm", "scale")] = to_numpy(sd["model.norm.weight"])

    n_layers = cfg.num_hidden_layers
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        base = (f"TinyTransformerBlock_{i}",)

        kv[base + ("rms1", "scale")] = to_numpy(sd[prefix + "input_layernorm.weight"])
        kv[base + ("rms2", "scale")] = to_numpy(sd[prefix + "post_attention_layernorm.weight"])

        # Fuse qkv: concat transposed weights along output dim
        q = to_numpy(sd[prefix + "self_attn.q_proj.weight"], transpose=True)
        k = to_numpy(sd[prefix + "self_attn.k_proj.weight"], transpose=True)
        v = to_numpy(sd[prefix + "self_attn.v_proj.weight"], transpose=True)
        qkv = np.concatenate([q, k, v], axis=1)
        kv[base + ("NativeJaxSelfAttention_0", "qkv_proj", "kernel")] = qkv
        kv[base + ("NativeJaxSelfAttention_0", "o_proj", "kernel")] = to_numpy(
            sd[prefix + "self_attn.o_proj.weight"], transpose=True
        )

        kv[base + ("fc1", "kernel")] = to_numpy(sd[prefix + "mlp.gate_proj.weight"], transpose=True)
        kv[base + ("fc1", "bias")] = np.zeros((sd[prefix + "mlp.gate_proj.weight"].shape[0],), dtype=np.float32)
        kv[base + ("fc2", "kernel")] = to_numpy(sd[prefix + "mlp.down_proj.weight"], transpose=True)
        kv[base + ("fc2", "bias")] = np.zeros((sd[prefix + "mlp.down_proj.weight"].shape[0],), dtype=np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.model_id.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{name}.npz"
    np.savez(out_path, **{"/".join(k): v for k, v in kv.items()})
    print(f"Saved {out_path} with {len(kv)} tensors")


if __name__ == "__main__":
    main()
