# Generate_text_fast.py
# Fast JIT-compiled text generation using GiantGPT + jit_infer
# Works with your /mnt/data/GiantGPT.py and /mnt/data/Transformer_block.py
import argparse
import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from GiantGPT import GiantGPT                     # your model
from jit_infer import init_inference_state, make_generate_fn  # new wrappers

# Optional: if you use OmegaConf + Config.yml, try to load it; else fall back to flags
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # fallback


def maybe_load_config(cli_args) -> Dict[str, Any]:
    if OmegaConf is not None and os.path.exists("Config.yml"):
        cfg = OmegaConf.load("Config.yml")
        # Expecting fields like these in your setup (adjust if yours differ)
        return dict(
            vocab_size=int(cfg.vocab_size),
            context_length=int(cfg.context_length),
            d_model=int(cfg.d_model),
            n_heads=int(cfg.n_heads),
            n_layers=int(cfg.n_layers),
            d_ff=int(cfg.d_ff),
            dropout_rate=float(cfg.dropout_rate),
        )
    # fallback to CLI
    return dict(
        vocab_size=cli_args.vocab_size,
        context_length=cli_args.context_length,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        n_layers=cli_args.n_layers,
        d_ff=cli_args.d_ff,
        dropout_rate=cli_args.dropout_rate,
    )


def build_model(cfg: Dict[str, Any]) -> GiantGPT:
    return GiantGPT(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        dropout_rate=cfg["dropout_rate"],
    )


def parse_ids(s: str) -> jnp.ndarray:
    # parse "1,2,3" -> [1,2,3]
    ids = [int(x) for x in s.strip().split(",") if x.strip()]
    return jnp.array(ids, dtype=jnp.int32)[None, :]  # [1, L]


def main():
    ap = argparse.ArgumentParser(description="Fast JIT decoding for GiantGPT")
    # Model/config (used if Config.yml not present)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--context_length", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout_rate", type=float, default=0.0)

    # Inference
    ap.add_argument("--prompt_ids", type=str, default="", help="Comma-separated token ids, e.g. '50256,123,456'")
    ap.add_argument("--bos_id", type=int, default=0, help="Used if prompt_ids empty")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)

    # Checkpoint loading (optional; stubbed here)
    ap.add_argument("--params_path", type=str, default="", help="Optional path to params (flax msgpack, orbax, etc.)")

    args = ap.parse_args()

    cfg = maybe_load_config(args)
    model = build_model(cfg)

    # RNGs
    key = jax.random.PRNGKey(0)
    k_params, k_drop, k_sample = jax.random.split(key, 3)

    # Init params and a cache-ready nonparam state
    params, nonparam = init_inference_state(
        model, k_params, k_drop, batch_size=1, pad_token_id=args.bos_id, use_kv_cache=True
    )

    # TODO (optional): load real params from checkpoint here if you have a loader.
    # Keep random params for now so the script always runs.

    # Prompt tokens
    if args.prompt_ids.strip():
        prompt = parse_ids(args.prompt_ids)        # [1, Lp]
    else:
        prompt = jnp.array([[args.bos_id]], dtype=jnp.int32)  # start with BOS if no prompt

    # Build JIT-ed generate function (captures `model` statically)
    generate = make_generate_fn(model)

    # Run generation
    rng_for_sampling = k_sample if args.do_sample else None
    tokens_new, nonparam = generate(
        params,
        nonparam,
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        temperature=args.temperature,
        rng_key=rng_for_sampling,
    )

    # Output token ids as a simple proof-of-life; integrate your tokenizer for text
    tokens_new_host = jax.device_get(tokens_new)[0].tolist()
    print("Generated token IDs:", tokens_new_host)


if __name__ == "__main__":
    main()

