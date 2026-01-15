"""
Simple demo script to test JAX BgGPT generation.
This script demonstrates the full pipeline from loading to generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import time

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from bggpt_compressed_kv_model_jax import create_bggpt_compressed_model
from jit_inference_bggpt import make_generate_fn, benchmark_generate


def load_global_config():
    """Load Global_Config.yml to get data_root."""
    config_path = Path(__file__).resolve().parent.parent / "Global_Config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(config_dir: Path):
    """Load model configuration."""
    config_file = config_dir / "model_config.yaml"
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_jax_params(params_file: Path):
    """Load JAX parameters from npz file."""
    print(f"Loading parameters from {params_file}")
    
    flat_params = np.load(params_file)
    
    # Unflatten the dict
    params = {}
    for key, value in flat_params.items():
        parts = key.split('.')
        current = params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = jnp.asarray(value)
    
    return params


def main():
    parser = argparse.ArgumentParser(description="Demo JAX BgGPT generation")
    parser.add_argument(
        "--params_dir",
        type=str,
        default=None,
        help="Directory with JAX params (default: {data_root}/bggpt_jax_params)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--kv_compression",
        type=float,
        default=1.0,
        help="KV compression ratio"
    )
    parser.add_argument(
        "--rope_factor",
        type=float,
        default=1.0,
        help="RoPE scaling factor"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark"
    )
    
    args = parser.parse_args()
    
    # Configure JAX
    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())
    
    # Load configs
    global_config = load_global_config()
    data_root = Path(global_config['paths']['data_root'])
    
    if args.params_dir:
        params_dir = Path(args.params_dir)
    else:
        params_dir = data_root / "bggpt_jax_params"
    
    if not params_dir.exists():
        print(f"\nERROR: Parameter directory not found: {params_dir}")
        print("\nTo download and convert the model, run:")
        print("  python BGiant-JAX/convert_pytorch_to_jax.py")
        print("\nThis will:")
        print("  1. Download BgGPT-Gemma from HuggingFace")
        print("  2. Convert PyTorch parameters to JAX format")
        print(f"  3. Save to {params_dir}")
        return 1
    
    # Load model config
    model_config = load_model_config(params_dir)
    print(f"\nModel configuration:")
    print(f"  Vocab size: {model_config['vocab_size']}")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Heads: {model_config['num_heads']}")
    print(f"  KV heads: {model_config['num_kv_heads']}")
    
    # Create model
    print("\nCreating model...")
    model = create_bggpt_compressed_model(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_kv_heads=model_config['num_kv_heads'],
        d_ff=model_config['intermediate_size'],
        head_dim=model_config.get('head_dim', 256),  # Pass head_dim from config
        sliding_window=model_config.get('sliding_window', 4096),
        kv_compression_ratio=args.kv_compression,
        rope_factor=args.rope_factor,
        attn_logit_softcapping=model_config.get('attn_logit_softcapping', 50.0),
        final_logit_softcapping=model_config.get('final_logit_softcapping', 30.0),
        query_pre_attn_scalar=model_config.get('query_pre_attn_scalar', 256),
    )
    
    # Initialize cache with correct dtype (float16 to match params)
    # The model uses float16 for computation if params are float16
    # But we need to make sure the cache matches what the model outputs
    
    # Load parameters
    params_file = params_dir / "bggpt_params.npz"
    params = load_jax_params(params_file)
    print(f"Loaded {len(params)} parameter groups")
    
    # Load tokenizer
    tokenizer_dir = params_dir / "tokenizer"
    print(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    
    # Tokenize prompt
    print(f"\n{'='*80}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*80}")
    
    input_ids = tokenizer(args.prompt, return_tensors="np").input_ids
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # Create generate function
    print("\nCompiling generation function...")
    generate_fn = make_generate_fn(model, params)
    
    # Warmup
    print("Warming up (first run may be slow due to JIT compilation)...")
    _ = generate_fn(input_ids, 2, args.temperature, args.top_k, args.seed)
    _.block_until_ready()
    
    # Generate
    print(f"\nGenerating {args.max_tokens} tokens...")
    start_time = time.time()
    
    output_ids = generate_fn(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    
    output_ids.block_until_ready()
    elapsed = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print("Generated text:")
    print(f"{'='*80}")
    print(output_text)
    print(f"{'='*80}")
    
    toks_per_sec = args.max_tokens / elapsed
    print(f"\nPerformance: {elapsed:.3f}s ({toks_per_sec:.2f} tok/s)")
    
    # Optional benchmark
    if args.benchmark:
        print(f"\n{'='*80}")
        print("Running benchmark...")
        print(f"{'='*80}")
        benchmark_generate(
            model=model,
            params=params,
            prompt_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            num_warmup=2,
            num_runs=5,
        )
    
    return 0


if __name__ == "__main__":
    exit(main())
