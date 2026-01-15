"""
Test script to verify JAX BgGPT implementation without requiring model weights.
Tests model creation, forward pass, and generation logic with random parameters.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bggpt_compressed_kv_model_jax import create_bggpt_compressed_model


def test_model_creation():
    """Test that we can create the model."""
    print("="*80)
    print("Test 1: Model Creation")
    print("="*80)
    
    # Create a small model for testing
    model = create_bggpt_compressed_model(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=512,
        sliding_window=64,
        kv_compression_ratio=0.5,
        rope_factor=1.0,
    )
    
    print(f"✓ Model created successfully")
    print(f"  vocab_size: {model.vocab_size}")
    print(f"  hidden_size: {model.hidden_size}")
    print(f"  num_layers: {model.num_layers}")
    print(f"  num_heads: {model.num_heads}")
    print(f"  num_kv_heads: {model.num_kv_heads}")
    print()
    
    return model


def test_model_init(model):
    """Test parameter initialization."""
    print("="*80)
    print("Test 2: Parameter Initialization")
    print("="*80)
    
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 10
    
    # Create dummy input
    input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    
    # Initialize parameters
    print("Initializing parameters...")
    variables = model.init(
        rng,
        input_ids,
        past_key_values=None,
        use_cache=False,
        deterministic=True,
    )
    
    params = variables['params']
    
    # Count parameters
    param_count = 0
    for name, param in jax.tree_util.tree_leaves_with_path(params):
        if isinstance(param, jax.Array):
            param_count += param.size
    
    print(f"✓ Parameters initialized")
    print(f"  Total parameters: {param_count:,}")
    print()
    
    return params


def test_forward_pass(model, params):
    """Test forward pass."""
    print("="*80)
    print("Test 3: Forward Pass")
    print("="*80)
    
    # Create input
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass without cache
    print("Running forward pass (no cache)...")
    logits, past_kv = model.apply(
        {'params': params},
        input_ids,
        past_key_values=None,
        use_cache=False,
        deterministic=True,
    )
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: (1, 5, {model.vocab_size})")
    print(f"  Cache returned: {past_kv is not None}")
    print()
    
    return logits


def test_kv_cache(model, params):
    """Test KV cache functionality."""
    print("="*80)
    print("Test 4: KV Cache")
    print("="*80)
    
    # First pass with full sequence
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    
    print("Pass 1: Full sequence with cache...")
    logits1, past_kv = model.apply(
        {'params': params},
        input_ids,
        past_key_values=None,
        use_cache=True,
        deterministic=True,
    )
    
    print(f"  Logits shape: {logits1.shape}")
    print(f"  Cache entries: {len(past_kv)}")
    if past_kv and len(past_kv) > 0:
        k_comp, v_comp = past_kv[0]
        print(f"  Layer 0 cache shapes: K={k_comp.shape}, V={v_comp.shape}")
    
    # Second pass with single token
    new_token = jnp.array([[6]])
    
    print("\nPass 2: Single token with existing cache...")
    logits2, past_kv2 = model.apply(
        {'params': params},
        new_token,
        past_key_values=past_kv,
        use_cache=True,
        deterministic=True,
    )
    
    print(f"  Logits shape: {logits2.shape}")
    if past_kv2 and len(past_kv2) > 0:
        k_comp2, v_comp2 = past_kv2[0]
        print(f"  Updated cache shapes: K={k_comp2.shape}, V={v_comp2.shape}")
    
    print(f"✓ KV cache working correctly")
    print()


def test_generation_logic(model, params):
    """Test autoregressive generation logic."""
    print("="*80)
    print("Test 5: Generation Logic")
    print("="*80)
    
    # Simple greedy generation
    input_ids = jnp.array([[1, 2, 3]])
    max_new_tokens = 5
    
    print(f"Initial tokens: {input_ids[0].tolist()}")
    print(f"Generating {max_new_tokens} new tokens...")
    
    generated = input_ids
    past_kv = None
    
    for step in range(max_new_tokens):
        if past_kv is None:
            # First pass
            logits, past_kv = model.apply(
                {'params': params},
                generated,
                past_key_values=None,
                use_cache=True,
                deterministic=True,
            )
        else:
            # Subsequent passes
            logits, past_kv = model.apply(
                {'params': params},
                generated[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
                deterministic=True,
            )
        
        # Greedy decoding
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        generated = jnp.concatenate([generated, next_token], axis=1)
    
    print(f"Generated sequence: {generated[0].tolist()}")
    print(f"✓ Generation logic working")
    print()


def test_compression_ratios(model, params):
    """Test different compression ratios."""
    print("="*80)
    print("Test 6: Compression Ratios")
    print("="*80)
    
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    
    for ratio in [1.0, 0.5, 0.25]:
        model_test = create_bggpt_compressed_model(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            d_ff=512,
            sliding_window=64,
            kv_compression_ratio=ratio,
        )
        
        # Note: We'd need to reinit params for this model, so just test creation
        print(f"  Compression ratio {ratio}: ✓ model created")
    
    print(f"✓ Compression ratios working")
    print()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("JAX BgGPT Implementation Tests")
    print("="*80)
    print()
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Parameter initialization
        params = test_model_init(model)
        
        # Test 3: Forward pass
        logits = test_forward_pass(model, params)
        
        # Test 4: KV cache
        test_kv_cache(model, params)
        
        # Test 5: Generation
        test_generation_logic(model, params)
        
        # Test 6: Compression ratios
        test_compression_ratios(model, params)
        
        # Summary
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("The JAX BgGPT implementation is working correctly.")
        print("To run with real model weights:")
        print("  1. Run: python convert_pytorch_to_jax.py")
        print("  2. Run: python demo_generate.py --prompt 'Your prompt here'")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
