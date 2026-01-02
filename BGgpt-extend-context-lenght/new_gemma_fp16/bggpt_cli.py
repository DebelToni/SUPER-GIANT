# bggpt_cli.py

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from bggpt_config import BgGPTConfig
from bggpt_gguf_loader import download_gguf_model, load_bggpt_params
from bggpt_generate import GenerationConfigJax, generate


HF_MODEL_REPO = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JAX inference for BgGPT Gemma-2 2.6B from GGUF FP16 weights"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt (Bulgarian or English)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'gpu'",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to cache GGUF file",
    )
    args = parser.parse_args()

    # Choose device
    if args.device == "cpu":
        device = jax.devices("cpu")[0]
    elif args.device == "gpu":
        device = jax.devices("gpu")[0]
    else:
        # auto: prefer GPU if available
        devs = jax.devices("gpu") or jax.devices("cpu")
        device = devs[0]

    print(f"Using device: {device}")

    # Download GGUF
    gguf_path = download_gguf_model(local_dir=args.models_dir)
    print(f"Using GGUF file: {gguf_path}")

    # Load config + params
    base_cfg = BgGPTConfig()
    cfg, params = load_bggpt_params(
        gguf_path, config=base_cfg, dtype=jnp.float16, device=device
    )
    print(f"Loaded model with vocab_size={cfg.vocab_size}")

    # Load tokenizer from HF original model
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_REPO, use_default_system_prompt=False
    )

    gen_cfg = GenerationConfigJax(
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        top_k=25,
        top_p=1.0,
        repetition_penalty=1.1,
        eos_token_ids=(cfg.eos_token_id, 107),
    )

    rng_key = jax.random.PRNGKey(args.seed)

    # Run generation
    output_text = generate(
        params,
        cfg,
        tokenizer,
        args.prompt,
        gen_cfg,
        rng_key,
    )

    print("=== MODEL OUTPUT ===")
    print(output_text)


if __name__ == "__main__":
    main()

