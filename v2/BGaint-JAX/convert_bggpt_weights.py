import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, Gemma2Config

from bggpt_compressed_kv_model import ModelConfig, load_bggpt_compressed


def parse_dtype(name: str):
    mapping = {
        "float32": (torch.float32, jnp.float32),
        "fp32": (torch.float32, jnp.float32),
        "bfloat16": (torch.bfloat16, jnp.bfloat16),
        "bf16": (torch.bfloat16, jnp.bfloat16),
        "float16": (torch.float16, jnp.float16),
        "fp16": (torch.float16, jnp.float16),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[name]


def build_torch_model(args) -> AutoModelForCausalLM:
    if args.tiny:
        head_dim = args.head_dim or (args.hidden_size // args.heads)
        q_scalar = args.query_pre_attn_scalar or float(head_dim)
        cfg = Gemma2Config(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads,
            head_dim=head_dim,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.layers,
            rope_theta=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            sliding_window=args.sliding_window,
            attention_dropout=0.0,
            attn_logit_softcapping=None,
            final_logit_softcapping=None,
            query_pre_attn_scalar=q_scalar,
            layer_types=["full_attention"] * args.layers,
        )
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        torch_dtype, _ = parse_dtype(args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            device_map=None,
        )
    return model


def layer_sliding_list(base_cfg) -> Optional[list]:
    layer_types = getattr(base_cfg, "layer_types", None)
    if layer_types is None:
        return None
    sw = getattr(base_cfg, "sliding_window", None)
    return [sw if t == "sliding_attention" else None for t in layer_types]


def make_flax_config(base_cfg, args) -> ModelConfig:
    return ModelConfig(
        hidden_size=base_cfg.hidden_size,
        num_attention_heads=base_cfg.num_attention_heads,
        num_key_value_heads=getattr(base_cfg, "num_key_value_heads", base_cfg.num_attention_heads),
        head_dim=getattr(base_cfg, "head_dim", None),
        vocab_size=base_cfg.vocab_size,
        intermediate_size=base_cfg.intermediate_size,
        num_hidden_layers=base_cfg.num_hidden_layers,
        attention_dropout=base_cfg.attention_dropout,
        attn_logit_softcapping=getattr(base_cfg, "attn_logit_softcapping", None),
        final_logit_softcapping=getattr(base_cfg, "final_logit_softcapping", None),
        query_pre_attn_scalar=getattr(base_cfg, "query_pre_attn_scalar", None),
        rope_theta=getattr(base_cfg, "rope_theta", 10000.0),
        rope_factor=args.rope_factor,
        sliding_window=getattr(base_cfg, "sliding_window", None),
        rms_norm_eps=getattr(base_cfg, "rms_norm_eps", 1e-6),
        dropout=getattr(base_cfg, "dropout", 0.0),
        kv_compression_ratio=args.kv_compression_ratio,
        layer_sliding_windows=layer_sliding_list(base_cfg),
    )


def to_jnp(weight: torch.Tensor, target_dtype):
    # numpy does not support bfloat16 tensors directly; cast to float32 first.
    np_arr = weight.detach().to(torch.float32).cpu().numpy()
    return jnp.asarray(np_arr, dtype=target_dtype)


def convert_params(base_model, flax_model, dtype, cfg: ModelConfig):
    rng = jax.random.PRNGKey(0)
    dummy_ids = jnp.zeros((1, 1), dtype=jnp.int32)
    variables = flax_model.init(rng, dummy_ids, past_key_values=None, use_cache=True, deterministic=True)
    params = flax.core.unfreeze(variables["params"])

    params["embed_tokens"]["embedding"] = to_jnp(base_model.model.embed_tokens.weight, dtype)

    for idx, layer in enumerate(base_model.model.layers):
        lp = params[f"layers_{idx}"]
        attn = layer.self_attn
        lp["self_attn"]["q_proj"]["kernel"] = to_jnp(attn.q_proj.weight.T, dtype)
        lp["self_attn"]["k_proj"]["kernel"] = to_jnp(attn.k_proj.weight.T, dtype)
        lp["self_attn"]["v_proj"]["kernel"] = to_jnp(attn.v_proj.weight.T, dtype)
        lp["self_attn"]["o_proj"]["kernel"] = to_jnp(attn.o_proj.weight.T, dtype)

        lp["input_layernorm"]["scale"] = to_jnp(layer.input_layernorm.weight, dtype)
        lp["post_attention_layernorm"]["scale"] = to_jnp(layer.post_attention_layernorm.weight, dtype)
        lp["pre_feedforward_layernorm"]["scale"] = to_jnp(layer.pre_feedforward_layernorm.weight, dtype)
        lp["post_feedforward_layernorm"]["scale"] = to_jnp(layer.post_feedforward_layernorm.weight, dtype)

        mlp = layer.mlp
        lp["mlp"]["Dense_0"]["kernel"] = to_jnp(mlp.up_proj.weight.T, dtype)
        lp["mlp"]["Dense_1"]["kernel"] = to_jnp(mlp.gate_proj.weight.T, dtype)
        lp["mlp"]["Dense_2"]["kernel"] = to_jnp(mlp.down_proj.weight.T, dtype)

    params["final_norm"]["scale"] = to_jnp(base_model.model.norm.weight, dtype)
    params["lm_head"]["kernel"] = to_jnp(base_model.lm_head.weight.T, dtype)
    return flax.core.freeze(params)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch Gemma2/BgGPT weights to Flax compressed-KV format")
    parser.add_argument("--model-name", default="INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0")
    parser.add_argument("--output", required=True, help="Path to save msgpack params")
    parser.add_argument("--config-out", default=None, help="Optional path to save ModelConfig JSON")
    parser.add_argument("--dtype", default="float32", help="float32|bfloat16|float16")
    parser.add_argument("--rope-factor", type=float, default=1.0)
    parser.add_argument("--kv-compression-ratio", type=float, default=1.0)
    parser.add_argument("--tiny", action="store_true", help="Use a tiny random Gemma2 config instead of downloading weights")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--max-position-embeddings", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--query-pre-attn-scalar", type=float, default=None)
    args = parser.parse_args()

    torch_dtype, jax_dtype = parse_dtype(args.dtype)

    base_model = build_torch_model(args)
    base_cfg = base_model.config
    flax_cfg = make_flax_config(base_cfg, args)
    flax_model = load_bggpt_compressed(flax_cfg)

    print(f"Converting model with hidden={flax_cfg.hidden_size}, layers={flax_cfg.num_hidden_layers}, heads={flax_cfg.num_attention_heads}")
    params = convert_params(base_model, flax_model, jax_dtype, flax_cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = flax.serialization.to_bytes({"params": params})
    out_path.write_bytes(payload)
    print(f"Saved params to {out_path}")

    cfg_path = Path(args.config_out) if args.config_out else out_path.with_suffix(".config.json")
    cfg_path.write_text(json.dumps(asdict(flax_cfg), indent=2))
    print(f"Saved config to {cfg_path}")


if __name__ == "__main__":
    main()
