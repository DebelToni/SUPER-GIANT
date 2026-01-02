"""
Demo script using the existing trained GiantGPT model (not BgGPT).
This demonstrates that our JAX implementation can generate coherent text.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from Generate_faster import load_configs, load_tokenizer, load_params, resolve_checkpoint_path, build_model
from jit_inference import make_prefill_and_decode_fns


def simple_generate_demo():
    """Simple generation demo with existing GiantGPT model."""
    print("="*80)
    print("JAX GiantGPT Generation Demo")
    print("="*80)
    print()
    
    # Load config
    cfg = load_configs()
    print(f"Loaded configuration")
    
    # Try to find a checkpoint
    try:
        checkpoint_path = resolve_checkpoint_path(cfg, "latest", "checkpoints")
        print(f"Using checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print("No trained checkpoint found. Trying mini checkpoints...")
        try:
            checkpoint_path = resolve_checkpoint_path(cfg, "latest", "checkpoints/mini")
            print(f"Using mini checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            print("\nNo checkpoints found. Please train a model first with:")
            print("  cd model && python Run_training.py")
            return
    
    # Load tokenizer
    tokenizer = load_tokenizer(cfg)
    print(f"Loaded tokenizer (vocab size: {len(tokenizer)})")
    
    # Build model
    context_length = int(cfg.model.context_length)
    model = build_model(cfg, len(tokenizer), context_length)
    print(f"Built model")
    print(f"  d_model: {cfg.model.embedding_size}")
    print(f"  n_heads: {cfg.model.num_heads}")
    print(f"  n_layers: {cfg.model.num_layers}")
    print(f"  context_length: {context_length}")
    
    # Load parameters
    params = load_params(checkpoint_path)
    print(f"Loaded parameters from checkpoint")
    
    # Create prompts to test
    prompts = [
        "Once upon a time",
        "The capital of France is",
        "In the beginning",
        "Science is"
    ]
    
    print("\n" + "="*80)
    print("Generating text...")
    print("="*80)
    
    # Setup inference
    rng = jax.random.PRNGKey(42)
    key_params, key_dropout, key_sample = jax.random.split(rng, 3)
    
    # Get inference functions
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)
    
    max_new_tokens = 32
    temperature = 0.8
    top_k = 40
    
    for i, prompt_text in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: '{prompt_text}'")
        print("-" * 80)
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) >= context_length:
            prompt_ids = prompt_ids[-context_length:]
        prompt = jnp.asarray(np.array(prompt_ids)[None, :], dtype=jnp.int32)
        
        # Initialize non-param state
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        from jit_inference import init_inference_state
        _, nonparam = init_inference_state(
            model,
            key_params,
            key_dropout,
            batch_size=1,
            pad_token_id=pad_token_id,
            use_kv_cache=True,
        )
        
        # Prefill
        nonparam_filled, t_cur, last_tok = prefill_fn(params, nonparam, prompt)
        
        # Decode
        tokens_new, _ = decode_fn(
            params,
            nonparam_filled,
            last_tok,
            t_cur,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            rng_key=key_sample,
        )
        
        # Combine and decode
        generated = jnp.concatenate([prompt, tokens_new], axis=1)
        full_tokens = np.asarray(generated[0])
        text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        print(text)
    
    print("\n" + "="*80)
    print("✓ Demo complete!")
    print("="*80)
    print()
    print("This demonstrates that the JAX implementation can generate coherent text.")
    print("The BGiant-JAX folder contains a similar implementation for the BgGPT model.")


if __name__ == "__main__":
    simple_generate_demo()
