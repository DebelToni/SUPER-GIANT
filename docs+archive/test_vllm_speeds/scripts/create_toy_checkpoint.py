from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file


class ToyCheckpointLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int, eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden, eps=eps)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.ln_2 = nn.LayerNorm(hidden, eps=eps)
        self.fc1 = nn.Linear(hidden, intermediate, bias=False)
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)


class ToyCheckpointModel(nn.Module):
    """
    Parameter-only module matching names used by ToyDecoderForCausalLM,
    but without vLLM Attention (it has no trainable weights anyway).
    """

    def __init__(
        self,
        vocab: int,
        max_pos: int,
        hidden: int,
        layers: int,
        heads: int,
        intermediate: int,
        eps: float,
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.tok_emb = nn.Embedding(vocab, hidden)
        self.model.pos_emb = nn.Embedding(max_pos, hidden)
        self.model.layers = nn.ModuleList(
            [ToyCheckpointLayer(hidden, heads, intermediate, eps) for _ in range(layers)]
        )
        self.model.ln_f = nn.LayerNorm(hidden, eps=eps)

        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.model.tok_emb.weight


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="toy-100m")
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ~100M params target (tweak these freely)
    vocab_size = 50257
    max_pos = 2048
    hidden = 640
    heads = 10
    n_layers = 14
    intermediate = 2560
    eps = 1e-5

    model = ToyCheckpointModel(
        vocab_size, max_pos, hidden, n_layers, heads, intermediate, eps
    )

    n_params = count_params(model)
    print(f"Parameter count: {n_params/1e6:.2f}M")

    dtype = getattr(torch, args.dtype)
    model.to(dtype=dtype)

    # Write config.json in GPT-2 shape so transformers/vLLM can parse it.
    config = {
        "model_type": "gpt2",
        "architectures": ["ToyDecoderForCausalLM"],
        "vocab_size": vocab_size,
        "n_positions": max_pos,
        "n_embd": hidden,
        "n_layer": n_layers,
        "n_head": heads,
        "n_inner": intermediate,
        "layer_norm_epsilon": eps,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "tie_word_embeddings": True,
        "use_cache": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Save weights
    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))
    print(f"Wrote: {out_dir}/config.json and {out_dir}/model.safetensors")


if __name__ == "__main__":
    main()
