"""
Standalone demo showing JAX generation capability.
Uses random weights to demonstrate the generation mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from jit_inference import make_prefill_and_decode_fns, init_inference_state
from omegaconf import OmegaConf


def demo_generation_mechanism():
    """Demo showing the generation mechanism works with random weights."""
    print("="*80)
    print("JAX Text Generation Mechanism Demo")
    print("="*80)
    print()
    print("This demo shows the JAX generation pipeline with a small model.")
    print("(Using random weights - won't produce coherent text, but shows it works)")
    print()
    
    # Load tokenizer
    config_path = Path(__file__).parent.parent / "Global_Config.yml"
    cfg = OmegaConf.load(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.name,
        use_fast=True,
        cache_dir=cfg.tokenizer.cache_dir,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    print(f"Loaded tokenizer: {cfg.tokenizer.name}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create a small model for demo (matching config proportions)
    # Config has: 640 dim, 10 heads, 5 kv_heads, 64 rope_dim
    # head_dim = 640/10 = 64, rope_dim = 64 works
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=256,
        d_model=640,  # 10 heads * 64 dim
        n_heads=10,
        d_ff=2560,
        n_layers=4,
        dropout_rate=0.0,
    )
    
    print(f"\nCreated small model:")
    print(f"  d_model: 640")
    print(f"  n_heads: 10")
    print(f"  n_kv_heads: 5 (from config)")
    print(f"  head_dim: 64")
    print(f"  rope_dim: 64 (from config)")
    print(f"  n_layers: 4")
    print(f"  context: 256")
    
    # Initialize with random weights
    rng = jax.random.PRNGKey(42)
    key_params, key_dropout, key_sample = jax.random.split(rng, 3)
    
    pad_token_id = tokenizer.pad_token_id or 0
    params, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=pad_token_id,
        use_kv_cache=True,
    )
    
    print(f"\nInitialized random parameters")
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "Hello world"
    ]
    
    print("\n" + "="*80)
    print("Generating tokens (random weights - demonstrating mechanism only)")
    print("="*80)
    
    # Create inference functions
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)
    
    max_new_tokens = 20
    
    for i, prompt_text in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: '{prompt_text}'")
        print("-" * 60)
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = jnp.asarray(np.array(prompt_ids)[None, :], dtype=jnp.int32)
        
        print(f"Input tokens: {prompt_ids}")
        
        # Prefill
        nonparam_filled, t_cur, last_tok = prefill_fn(params, nonparam, prompt)
        
        # Decode with greedy (deterministic for demo)
        tokens_new, _ = decode_fn(
            params,
            nonparam_filled,
            last_tok,
            t_cur,
            steps=max_new_tokens,  # Required positional arg
            temperature=0.0,  # Greedy
            do_sample=False,
            top_k=0,
            rng_key=None,
        )
        
        # Decode tokens
        generated = jnp.concatenate([prompt, tokens_new], axis=1)
        full_tokens = np.asarray(generated[0])
        
        print(f"Generated token IDs: {full_tokens.tolist()}")
        
        text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        print(f"Decoded text: {text}")
    
    print("\n" + "="*80)
    print("✓ Generation mechanism working!")
    print("="*80)
    print()
    print("Key points demonstrated:")
    print("  ✓ JAX model initialization")
    print("  ✓ Parameter creation")
    print("  ✓ Tokenization")
    print("  ✓ Prefill phase (processing prompt)")
    print("  ✓ Decode phase (autoregressive generation)")
    print("  ✓ KV cache usage")
    print("  ✓ Token decoding")
    print()
    print("The BGiant-JAX implementation follows the same pattern but with:")
    print("  - Compressed KV cache")
    print("  - RoPE position embeddings")
    print("  - Gemma2 architecture features")
    print("  - Trained weights from HuggingFace")
    print()
    print("To use with actual trained weights:")
    print("  1. Download BgGPT: python BGiant-JAX/convert_pytorch_to_jax.py")
    print("  2. Generate text: python BGiant-JAX/demo_generate.py")
    print()


if __name__ == "__main__":
    demo_generation_mechanism()
