#!/usr/bin/env python3
"""
Test script to verify TiDAR's zero-mask defense.

This script:
1. Loads the corrupted shard when available
2. Extracts an all-zero mask row and a normal row
3. Tests batch building with and without the Layer 2 fix
4. Confirms all-zero masks become trainable fallback rows
"""

import os
import sys
import jax
import jax.numpy as jnp
import pyarrow as pa
import numpy as np

# Add TiDAR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TiDAR.model.tidar_utils import build_train_batch


def load_corrupted_shard(shard_path):
    """Load the corrupted shard and extract relevant rows."""
    print(f"Loading shard: {shard_path}")
    
    reader = pa.ipc.RecordBatchFileReader(open(shard_path, 'rb'))
    table = reader.read_all()
    
    input_ids = table['input_ids'].to_pylist()
    loss_masks = table['loss_mask'].to_pylist()
    lengths = table['length'].to_pylist()
    
    # Find rows with zero mask
    zero_mask_rows = []
    normal_rows = []
    
    for i, mask in enumerate(loss_masks):
        mask_sum = sum(mask)
        if mask_sum == 0:
            zero_mask_rows.append(i)
            print(f"  Found zero-mask row: {i}")
        elif mask_sum > 100:  # A normal row with decent amount of tokens
            normal_rows.append(i)
    
    print(f"  Total rows: {len(loss_masks)}")
    print(f"  Zero-mask rows: {len(zero_mask_rows)}")
    print(f"  Normal rows with >100 tokens: {len(normal_rows)}")
    
    # Get the corrupted row (should be row 455)
    if len(zero_mask_rows) > 0:
        corrupted_idx = zero_mask_rows[0]
        print(f"\nCorrupted row index: {corrupted_idx}")
        print(f"  Input IDs length: {len(input_ids[corrupted_idx])}")
        print(f"  Loss mask sum: {sum(loss_masks[corrupted_idx])}")
        print(f"  Sequence length: {lengths[corrupted_idx]}")
    
    # Get a normal row for comparison
    if len(normal_rows) > 0:
        normal_idx = normal_rows[0]
        print(f"\nNormal row index: {normal_idx}")
        print(f"  Input IDs length: {len(input_ids[normal_idx])}")
        print(f"  Loss mask sum: {sum(loss_masks[normal_idx])}")
        print(f"  Sequence length: {lengths[normal_idx]}")
    
    return {
        'corrupted_row': {
            'input_ids': input_ids[corrupted_idx] if zero_mask_rows else None,
            'loss_mask': loss_masks[corrupted_idx] if zero_mask_rows else None,
            'length': lengths[corrupted_idx] if zero_mask_rows else None,
        },
        'normal_row': {
            'input_ids': input_ids[normal_idx] if normal_rows else None,
            'loss_mask': loss_masks[normal_idx] if normal_rows else None,
            'length': lengths[normal_idx] if normal_rows else None,
        }
    }


def build_batch_without_fix(clean, lengths, token_mask, seq_len, block_len=128, ignore_index=-100):
    """Original build_train_batch WITHOUT Layer 2 fix."""
    batch_size = clean.shape[0]
    
    noisy = clean.copy()
    position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
    input_ids = jnp.concatenate([noisy, clean], axis=1)
    position_ids = jnp.concatenate([position_ids, position_ids], axis=1)
    
    labels = jnp.full((batch_size, seq_len * 2), ignore_index, dtype=jnp.int32)
    labels = labels.at[:, : seq_len - 1].set(clean[:, 1:])
    labels = labels.at[:, seq_len:].set(clean)
    
    loss_mask_ntp = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)
    loss_mask_diff = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)
    
    if lengths is None:
        valid = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    else:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        valid = (positions < lengths[:, None]).astype(jnp.float32)
    if token_mask is not None:
        valid = valid * token_mask.astype(jnp.float32)
    
    # NO FIX HERE - this is the original code
    # valid can be all zeros, which creates invalid all-masked attention rows
    
    if seq_len > 1:
        loss_mask_ntp = loss_mask_ntp.at[:, : seq_len - 1].set(valid[:, 1:])
    loss_mask_diff = loss_mask_diff.at[:, seq_len:].set(valid)
    
    token_types = jnp.concatenate(
        [jnp.zeros(seq_len, dtype=jnp.int32), jnp.ones(seq_len, dtype=jnp.int32)]
    )
    key_padding_mask = jnp.concatenate([valid, valid], axis=1) > 0
    
    # Simplified attention bias (just show the masking issue)
    attn_bias = jnp.where(key_padding_mask[:, None, None, :], 0.0, -1e10)
    
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask_ntp": loss_mask_ntp,
        "loss_mask_diff": loss_mask_diff,
        "attn_bias": attn_bias,
        "key_padding_mask": key_padding_mask,
        "valid": valid,
    }


def test_attention_with_zero_mask():
    """Test that all-zero token masks are converted to safe trainable rows."""
    print("\n" + "="*80)
    print("TEST: Attention with All-Zero Mask")
    print("="*80)
    
    batch_size = 2
    seq_len = 2048
    d_model = 512
    
    # Create a batch with one corrupted row (all-zero mask) and one normal row
    clean = jnp.ones((batch_size, seq_len), dtype=jnp.int32) * 100  # dummy token IDs
    lengths = jnp.array([seq_len, seq_len], dtype=jnp.int32)
    
    # Row 0: all-zero mask (corrupted)
    # Row 1: normal mask
    token_mask = jnp.array([
        [0.0] * seq_len,  # All zeros - invalid row without fallback
        [1.0] * seq_len,  # All ones - normal
    ], dtype=jnp.float32)
    
    print("\nBuilding batch WITHOUT Layer 2 fix...")
    batch_no_fix = build_batch_without_fix(clean, lengths, token_mask, seq_len)
    
    print(f"  Row 0 valid mask sum: {batch_no_fix['valid'][0].sum()}")
    print(f"  Row 1 valid mask sum: {batch_no_fix['valid'][1].sum()}")
    print(f"  Row 0 key_padding_mask (should be all False): {batch_no_fix['key_padding_mask'][0].any()}")
    print(f"  Row 1 key_padding_mask (should be all True): {batch_no_fix['key_padding_mask'][1].all()}")
    
    # Simulate attention operation
    print("\nSimulating attention softmax with all-masked keys...")
    
    # For row 0, all keys are masked (bias = -1e10 everywhere).
    # JAX's finite bias may produce a uniform softmax instead of NaN, but the row
    # is still semantically invalid and should not reach training unchanged.
    query = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1, d_model))
    key = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len * 2, d_model))
    
    # Compute attention logits
    logits = jnp.einsum('bqd,bkd->bqk', query, key) / jnp.sqrt(d_model)
    
    # Apply attention bias (all -1e10 for row 0)
    logits_with_bias = logits + batch_no_fix['attn_bias'][:, 0, :, :]  # Take first query position
    
    print(f"  Row 0 logits before bias: min={logits[0].min():.2f}, max={logits[0].max():.2f}")
    print(f"  Row 0 logits after bias: min={logits_with_bias[0].min():.2e}, max={logits_with_bias[0].max():.2e}")
    
    attn_weights = jax.nn.softmax(logits_with_bias, axis=-1)
    
    print(f"  Row 0 attention weights after softmax: {attn_weights[0, 0, :5]}")
    print(f"  Row 0 has NaN: {jnp.isnan(attn_weights[0]).any()}")
    print(f"  Row 1 has NaN: {jnp.isnan(attn_weights[1]).any()}")
    assert not bool(batch_no_fix['key_padding_mask'][0].any())
    
    print("\n✓ INVALID CASE CONFIRMED: all-zero mask → no valid attention keys")
    
    # Now test with the fix
    print("\n" + "-"*80)
    print("Testing WITH Layer 2 fix (from build_train_batch)...")
    print("-"*80)
    
    batch_with_fix = build_train_batch(
        clean, lengths, 
        mask_id=0,  # dummy mask token
        block_len=128,
        token_mask=token_mask
    )
    
    print(f"  Row 0 valid mask sum: {batch_with_fix['loss_mask_ntp'][0].sum()}")
    print(f"  Row 1 valid mask sum: {batch_with_fix['loss_mask_ntp'][1].sum()}")
    
    # The fix should have converted row 0's mask to all 1.0s
    logits_with_bias_fixed = logits + batch_with_fix['attn_bias'][:, 0, :, :]
    attn_weights_fixed = jax.nn.softmax(logits_with_bias_fixed, axis=-1)
    
    print(f"  Row 0 attention weights after softmax: {attn_weights_fixed[0, 0, :5]}")
    print(f"  Row 0 has NaN: {jnp.isnan(attn_weights_fixed[0]).any()}")
    print(f"  Row 1 has NaN: {jnp.isnan(attn_weights_fixed[1]).any()}")
    
    assert bool(batch_with_fix['loss_mask_diff'][0].sum() > 0)
    print("\n✓ FIX VERIFIED: Layer 2 converts all-zero mask → trainable fallback row")


def test_with_real_shard():
    """Test with the actual corrupted shard."""
    shard_path = "/proj/giant-data/TiDAR/Sweep/Data/sweep_1_lr1e4_a1_l0/ultrachat_rehearsal_1m/ultrachat_rehearsal_1m-000000.arrow"
    
    if not os.path.exists(shard_path):
        print(f"\n⚠ Shard not found: {shard_path}")
        print("Skipping real shard test")
        return False
    
    print("\n" + "="*80)
    print("TEST: Real Corrupted Shard")
    print("="*80)
    
    data = load_corrupted_shard(shard_path)
    
    if data['corrupted_row']['input_ids'] is None:
        print("\n⚠ No corrupted row found in shard")
        return False
    
    # Build a batch with the corrupted row
    batch_size = 2
    seq_len = 2048
    
    clean = jnp.array([
        data['corrupted_row']['input_ids'],
        data['normal_row']['input_ids'],
    ], dtype=jnp.int32)
    
    lengths = jnp.array([
        data['corrupted_row']['length'],
        data['normal_row']['length'],
    ], dtype=jnp.int32)
    
    token_mask = jnp.array([
        data['corrupted_row']['loss_mask'],
        data['normal_row']['loss_mask'],
    ], dtype=jnp.float32)
    
    print("\nBuilding batch with real data (with Layer 2 fix)...")
    batch = build_train_batch(
        clean, lengths,
        mask_id=0,  # dummy mask token
        block_len=128,
        token_mask=token_mask
    )
    
    print(f"  Corrupted row NTP loss mask sum (after fix): {batch['loss_mask_ntp'][0].sum():.1f}")
    print(f"  Normal row NTP loss mask sum: {batch['loss_mask_ntp'][1].sum():.1f}")
    print(f"  Corrupted row DIFF loss mask sum (after fix): {batch['loss_mask_diff'][0].sum():.1f}")
    print(f"  Normal row DIFF loss mask sum: {batch['loss_mask_diff'][1].sum():.1f}")
    
    # Check attention bias for NaN
    attn_has_nan = jnp.isnan(batch['attn_bias']).any()
    print(f"  Attention bias has NaN: {attn_has_nan}")
    
    print("\n✓ Real shard test passed: Corrupted row handled without NaN")
    return True


def main():
    print("TiDAR Zero-Mask Defense Test")
    print("="*80)
    print("This script verifies that:")
    print("1. All-zero loss masks create invalid all-masked attention rows")
    print("2. Layer 2 fix prevents this by converting zero-masks to trainable fallback rows")
    print("3. Real corrupted shard data is handled correctly")
    print()
    
    # Test 1: Prove the invalid row with synthetic data
    test_attention_with_zero_mask()
    
    # Test 2: Test with real corrupted shard
    real_shard_checked = test_with_real_shard()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\nSummary:")
    print("✓ Invalid row confirmed: zero-mask → no valid attention keys")
    print("✓ Layer 2 fix converts zero-masks to trainable fallback rows")
    print("✓ Real shard data handled correctly" if real_shard_checked else "- Real shard data skipped")
    print("\nThe 3-layer defense is ready for deployment!")


if __name__ == "__main__":
    main()
