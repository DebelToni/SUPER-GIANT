"""
JAX/FLAX inference script for compressed BgGPT model.
Converted from PyTorch version in BGiant/run_bggpt_compressed.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional
import yaml

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from bggpt_compressed_kv_model_jax import create_bggpt_compressed_model


def load_global_config():
    """Load Global_Config.yml to get data_root."""
    config_path = Path(__file__).resolve().parent.parent / "Global_Config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(config_dir: Path):
    """Load model configuration from YAML."""
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


def load_tokenizer(tokenizer_dir: Path):
    """Load tokenizer from directory."""
    print(f"Loading tokenizer from {tokenizer_dir}")
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def init_model_state(model, rng_key, batch_size: int = 1):
    """Initialize model parameters and state."""
    dummy_input = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    variables = model.init(
        rng_key,
        dummy_input,
        past_key_values=None,
        use_cache=False,
        deterministic=True,
    )
    return variables['params']


@jax.jit
def forward_pass(
    model,
    params,
    input_ids: jnp.ndarray,
    past_key_values: Optional[list] = None,
    use_cache: bool = True,
):
    """Single forward pass through the model."""
    logits, new_past = model.apply(
        {'params': params},
        input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        deterministic=True,
    )
    return logits, new_past


def top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Filter logits to keep only top-k values."""
    if k <= 0:
        return logits
    
    top_values, top_indices = jax.lax.top_k(logits, k)
    kth_value = top_values[..., -1, None]
    return jnp.where(logits < kth_value, float('-inf'), logits)


def generate_autoregressive(
    model,
    params,
    input_ids: jnp.ndarray,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    rng_key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Generate tokens autoregressively."""
    
    generated = input_ids
    past_kv = None
    
    for step in range(max_new_tokens):
        # Forward pass
        if past_kv is None:
            # First pass: full prompt
            logits, past_kv = forward_pass(model, params, generated, None, use_cache=True)
        else:
            # Subsequent passes: only last token
            logits, past_kv = forward_pass(
                model, params, generated[:, -1:], past_kv, use_cache=True
            )
        
        # Get next token logits
        next_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            next_logits = top_k_logits(next_logits, top_k)
        
        # Sample or greedy decode
        if temperature > 0.0 and rng_key is not None:
            rng_key, subkey = jax.random.split(rng_key)
            probs = jax.nn.softmax(next_logits, axis=-1)
            next_token = jax.random.categorical(subkey, jnp.log(probs), axis=-1)
            next_token = next_token[:, None]
        else:
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Append to generated sequence
        generated = jnp.concatenate([generated, next_token], axis=1)
        
        # Optional: Print progress
        if (step + 1) % 10 == 0:
            print(f"Generated {step + 1}/{max_new_tokens} tokens...", end='\r')
    
    print()  # New line after progress
    return generated


def main():
    parser = argparse.ArgumentParser(description="Run JAX BgGPT inference")
    parser.add_argument(
        "--params_dir",
        type=str,
        default=None,
        help="Directory containing JAX parameters (defaults to {data_root}/bggpt_jax_params)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How are you?",
        help="Prompt text for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--kv_compression_ratio",
        type=float,
        default=1.0,
        help="KV cache compression ratio (1.0 = no compression)"
    )
    parser.add_argument(
        "--rope_factor",
        type=float,
        default=1.0,
        help="RoPE scaling factor for context extension"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    global_config = load_global_config()
    data_root = Path(global_config['paths']['data_root'])
    
    if args.params_dir:
        params_dir = Path(args.params_dir)
    else:
        params_dir = data_root / "bggpt_jax_params"
    
    if not params_dir.exists():
        print(f"Error: Parameter directory not found: {params_dir}")
        print("Please run convert_pytorch_to_jax.py first to download and convert the model")
        return
    
    # Load model config
    model_config = load_model_config(params_dir)
    print(f"Model config: {model_config}")
    
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
        kv_compression_ratio=args.kv_compression_ratio,
        rope_factor=args.rope_factor,
        attn_logit_softcapping=model_config.get('attn_logit_softcapping', 50.0),
        final_logit_softcapping=model_config.get('final_logit_softcapping', 30.0),
        query_pre_attn_scalar=model_config.get('query_pre_attn_scalar', 256),
    )
    
    # Load parameters
    params_file = params_dir / "bggpt_params.npz"
    params = load_jax_params(params_file)
    
    # Load tokenizer
    tokenizer_dir = params_dir / "tokenizer"
    tokenizer = load_tokenizer(tokenizer_dir)
    
    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    input_ids = tokenizer(args.prompt, return_tensors="np").input_ids
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    print(f"Input shape: {input_ids.shape}")
    
    # Set random seed
    rng_key = jax.random.PRNGKey(args.seed) if args.temperature > 0 else None
    
    # Generate
    print("\nGenerating...")
    start_time = time.time()
    
    generated_ids = generate_autoregressive(
        model=model,
        params=params,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng_key=rng_key,
    )
    
    generated_ids.block_until_ready()  # Ensure computation completes
    elapsed = time.time() - start_time
    
    # Decode output
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("Generated text:")
    print("="*80)
    print(output_text)
    print("="*80)
    
    tokens_per_second = args.max_new_tokens / elapsed
    print(f"\nGenerated {args.max_new_tokens} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tok/s)")


if __name__ == "__main__":
    main()
