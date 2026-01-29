#!/usr/bin/env python3
"""
Edge case tests for the 5-term TiDAR loss function.

Tests cover:
1. Numerical stability with extreme logits
2. Empty/degenerate masks
3. Gradient correctness
4. Loss term independence
5. Integration with build_train_batch
6. Config loading and parsing
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Import the actual loss functions from TiDAR
from TiDAR.model.Training_step import loss_and_metrics, loss_and_grad
from TiDAR.model.tidar_utils import build_train_batch


def create_mock_model():
    """Create a minimal mock model for testing."""
    class MockModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            # Return random logits with correct shape
            B, seq_len_2 = input_ids.shape
            V = 100
            key = jax.random.PRNGKey(42)
            return jax.random.normal(key, (B, seq_len_2, V))
    return MockModel()


def create_test_batch(batch_size=2, seq_len=16, vocab_size=100, seed=42):
    """Create a test batch with standard TiDAR structure."""
    key = jax.random.PRNGKey(seed)
    
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    lengths = jnp.array([seq_len] * batch_size, dtype=jnp.int32)
    
    batch = build_train_batch(
        tokens, lengths,
        mask_id=0,
        block_len=4,
    )
    return batch


# =============================================================================
# Test 1: Numerical Stability
# =============================================================================

def test_extreme_logits():
    """Test that loss doesn't produce NaN/Inf with extreme logit values."""
    print("Test 1: Extreme logits stability")
    
    B, S, V = 2, 8, 100
    
    # Test with very large logits (softmax should still be stable)
    key = jax.random.PRNGKey(42)
    large_logits = jax.random.normal(key, (B, 2 * S, V)) * 100
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)
    
    # Mock model that returns the large logits
    class LargeLogitModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return large_logits
    
    batch = {
        "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "labels": labels,
        "loss_mask_ntp": mask_ntp,
        "loss_mask_diff": mask_diff,
        "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
    }
    
    total, (ar, diff, kl_fwd, kl_rev, hard, acc) = loss_and_metrics(
        {}, batch, model=LargeLogitModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=1.0, beta=1.0, rho=0.1, chi=0.1, delta=0.1,
        compute_accept=False, accept_top_k=64, accept_max_positions=256
    )
    
    assert jnp.isfinite(total), f"Total loss is not finite: {total}"
    assert jnp.isfinite(ar), f"AR loss is not finite: {ar}"
    assert jnp.isfinite(diff), f"Diff loss is not finite: {diff}"
    assert jnp.isfinite(kl_fwd), f"KL_fwd is not finite: {kl_fwd}"
    assert jnp.isfinite(kl_rev), f"KL_rev is not finite: {kl_rev}"
    assert jnp.isfinite(hard), f"Hard agree is not finite: {hard}"
    
    print(f"  Large logits: total={float(total):.4f} (all finite)")
    
    # Test with very small logits
    small_logits = jax.random.normal(key, (B, 2 * S, V)) * 0.001
    
    class SmallLogitModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return small_logits
    
    total2, _ = loss_and_metrics(
        {}, batch, model=SmallLogitModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=1.0, beta=1.0, rho=0.1, chi=0.1, delta=0.1,
        compute_accept=False, accept_top_k=64, accept_max_positions=256
    )
    
    assert jnp.isfinite(total2), f"Total loss with small logits is not finite: {total2}"
    print(f"  Small logits: total={float(total2):.4f} (all finite)")
    print("  PASSED")


# =============================================================================
# Test 2: Degenerate Masks
# =============================================================================

def test_single_valid_position():
    """Test loss with only a single valid position."""
    print("\nTest 2: Single valid position")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (B, 2 * S, V))
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    
    # Only one valid position for NTP
    mask_ntp = jnp.zeros((B, 2 * S))
    mask_ntp = mask_ntp.at[:, 0].set(1.0)  # Only first position
    
    # Only one valid position for Diff
    mask_diff = jnp.zeros((B, 2 * S))
    mask_diff = mask_diff.at[:, S].set(1.0)  # Only first diff position
    
    class SinglePosModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return logits
    
    batch = {
        "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "labels": labels,
        "loss_mask_ntp": mask_ntp,
        "loss_mask_diff": mask_diff,
        "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
    }
    
    total, (ar, diff, kl_fwd, kl_rev, hard, acc) = loss_and_metrics(
        {}, batch, model=SinglePosModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=1.0, beta=1.0, rho=0.0, chi=0.0, delta=0.0,
        compute_accept=False, accept_top_k=64, accept_max_positions=256
    )
    
    assert jnp.isfinite(total), f"Loss not finite with single valid position: {total}"
    print(f"  Single position: total={float(total):.4f}, ar={float(ar):.4f}, diff={float(diff):.4f}")
    print("  PASSED")


def test_all_zero_coefficient():
    """Test that setting all coefficients to 0 returns 0 loss."""
    print("\nTest 3: All zero coefficients")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (B, 2 * S, V))
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)
    
    class ZeroCoeffModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return logits
    
    batch = {
        "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "labels": labels,
        "loss_mask_ntp": mask_ntp,
        "loss_mask_diff": mask_diff,
        "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
    }
    
    # All coefficients = 0 should give 0 loss
    total, (ar, diff, kl_fwd, kl_rev, hard, acc) = loss_and_metrics(
        {}, batch, model=ZeroCoeffModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=0.0, beta=0.0, rho=0.0, chi=0.0, delta=0.0,
        compute_accept=False, accept_top_k=64, accept_max_positions=256
    )
    
    assert float(total) == 0.0, f"Expected 0 loss with all zero coefficients, got {total}"
    print(f"  All zero coeffs: total={float(total):.4f}")
    print("  PASSED")


# =============================================================================
# Test 4: Coefficient Independence
# =============================================================================

def test_coefficient_additivity():
    """Test that loss terms combine correctly: L = sum(coef_i * L_i)."""
    print("\nTest 4: Coefficient additivity")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    ar_logits = jax.random.normal(key, (B, S, V))
    key, subkey = jax.random.split(key)
    diff_logits = jax.random.normal(subkey, (B, S, V)) * 1.5
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)
    
    class AdditivityModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return logits
    
    batch = {
        "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "labels": labels,
        "loss_mask_ntp": mask_ntp,
        "loss_mask_diff": mask_diff,
        "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
    }
    
    def get_loss(**kwargs):
        total, _ = loss_and_metrics(
            {}, batch, model=AdditivityModel(), dropout_rng=jax.random.PRNGKey(0),
            compute_accept=False, accept_top_k=64, accept_max_positions=256,
            **kwargs
        )
        return float(total)
    
    # Get individual loss terms
    l_ar = get_loss(alpha=1.0, beta=0.0, rho=0.0, chi=0.0, delta=0.0)
    l_diff = get_loss(alpha=0.0, beta=1.0, rho=0.0, chi=0.0, delta=0.0)
    l_rho = get_loss(alpha=0.0, beta=0.0, rho=1.0, chi=0.0, delta=0.0)
    l_chi = get_loss(alpha=0.0, beta=0.0, rho=0.0, chi=1.0, delta=0.0)
    l_delta = get_loss(alpha=0.0, beta=0.0, rho=0.0, chi=0.0, delta=1.0)
    
    # Get combined loss
    alpha, beta, rho, chi, delta = 0.5, 0.7, 0.1, 0.05, 0.2
    l_combined = get_loss(alpha=alpha, beta=beta, rho=rho, chi=chi, delta=delta)
    
    expected = alpha * l_ar + beta * l_diff + rho * l_rho + chi * l_chi + delta * l_delta
    
    assert abs(l_combined - expected) < 1e-4, f"Additivity failed: {l_combined} vs {expected}"
    
    print(f"  L_AR={l_ar:.4f}, L_Diff={l_diff:.4f}, L_rho={l_rho:.4f}, L_chi={l_chi:.4f}, L_delta={l_delta:.4f}")
    print(f"  Combined: {l_combined:.4f}, Expected: {expected:.4f}")
    print("  PASSED")


# =============================================================================
# Test 5: Gradient Flow
# =============================================================================

def test_gradient_finite():
    """Test that gradients are finite for all parameter configurations."""
    print("\nTest 5: Gradient finiteness")
    
    B, S, V = 2, 8, 100  # V must be >= accept_top_k
    D = V  # D = V for this test since we treat embedding output as logits
    
    key = jax.random.PRNGKey(42)
    
    # Create simple "params" - just an embedding matrix
    params = {"embedding": jax.random.normal(key, (V, D))}
    
    key, subkey = jax.random.split(key)
    tokens = jax.random.randint(subkey, (B, S), 0, V)
    lengths = jnp.array([S, S], dtype=jnp.int32)
    
    batch = build_train_batch(tokens, lengths, mask_id=0, block_len=4)
    
    # Model that uses params
    class GradModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            emb = variables["params"]["embedding"]
            B, seq_len_2 = input_ids.shape
            # Simple "logits" from embedding lookup
            return emb[input_ids % V]  # [B, 2S, D] - treating D as vocab
    
    # Test gradient computation (accept_top_k must be <= V)
    (total, aux), grads = loss_and_grad(
        params, batch, model=GradModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=1.0, beta=1.0, rho=0.1, chi=0.1, delta=0.1,
        compute_accept=False, accept_top_k=min(64, V), accept_max_positions=256
    )
    
    grad_finite = jax.tree_util.tree_reduce(
        lambda a, b: jnp.logical_and(a, b),
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        jnp.asarray(True)
    )
    
    assert grad_finite, "Gradients contain NaN or Inf"
    assert jnp.isfinite(total), f"Loss is not finite: {total}"
    
    print(f"  Loss={float(total):.4f}, gradients finite: {bool(grad_finite)}")
    print("  PASSED")


# =============================================================================
# Test 6: Integration with build_train_batch
# =============================================================================

def test_build_train_batch_integration():
    """Test that loss function works with real build_train_batch output."""
    print("\nTest 6: build_train_batch integration")
    
    B, S, V = 4, 32, 100
    key = jax.random.PRNGKey(42)
    
    tokens = jax.random.randint(key, (B, S), 1, V)  # Avoid 0 (mask token)
    lengths = jnp.array([S, S - 5, S - 10, S // 2], dtype=jnp.int32)
    
    # Token mask (some positions masked out)
    key, subkey = jax.random.split(key)
    token_mask = (jax.random.uniform(subkey, (B, S)) > 0.1).astype(jnp.float32)
    
    batch = build_train_batch(
        tokens, lengths,
        mask_id=0,
        block_len=8,
        token_mask=token_mask
    )
    
    # Verify batch structure
    assert batch["input_ids"].shape == (B, 2 * S), f"Wrong input_ids shape: {batch['input_ids'].shape}"
    assert batch["labels"].shape == (B, 2 * S), f"Wrong labels shape: {batch['labels'].shape}"
    assert batch["loss_mask_ntp"].shape == (B, 2 * S), f"Wrong mask_ntp shape"
    assert batch["loss_mask_diff"].shape == (B, 2 * S), f"Wrong mask_diff shape"
    
    # Check masks are valid
    assert batch["loss_mask_ntp"].sum() > 0, "NTP mask is all zeros"
    assert batch["loss_mask_diff"].sum() > 0, "Diff mask is all zeros"
    
    # Test with mock model
    key, subkey = jax.random.split(key)
    logits = jax.random.normal(subkey, (B, 2 * S, V))
    
    class IntegrationModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return logits
    
    total, (ar, diff, kl_fwd, kl_rev, hard, acc) = loss_and_metrics(
        {}, batch, model=IntegrationModel(), dropout_rng=jax.random.PRNGKey(0),
        alpha=1.0, beta=1.0, rho=0.1, chi=0.1, delta=0.1,
        compute_accept=True, accept_top_k=64, accept_max_positions=256
    )
    
    assert jnp.isfinite(total), f"Loss not finite: {total}"
    assert 0.0 <= float(acc) <= 1.0, f"Accept rate out of range: {acc}"
    
    print(f"  Batch shapes OK, loss={float(total):.4f}, accept_rate={float(acc):.4f}")
    print("  PASSED")


# =============================================================================
# Test 7: Config Loading
# =============================================================================

def test_config_parsing():
    """Test that loss config is parsed correctly from YAML."""
    print("\nTest 7: Config parsing")
    
    from omegaconf import OmegaConf
    
    config_content = """
training:
  loss:
    alpha: 0.8
    beta: 1.2
    rho: 0.15
    chi: 0.05
    delta: 0.25

stages:
  - name: test_stage
    dataset: test_data
    seq_len: 512
    epochs: 1
    end_ratio: 1.0
    loss:
      alpha: 0.5
      beta: 1.0
"""
    
    cfg = OmegaConf.create(config_content)
    
    # Check global defaults
    assert cfg.training.loss.alpha == 0.8
    assert cfg.training.loss.beta == 1.2
    assert cfg.training.loss.rho == 0.15
    assert cfg.training.loss.chi == 0.05
    assert cfg.training.loss.delta == 0.25
    
    # Check stage override
    assert cfg.stages[0].loss.alpha == 0.5
    assert cfg.stages[0].loss.beta == 1.0
    
    print(f"  Global: alpha={cfg.training.loss.alpha}, beta={cfg.training.loss.beta}")
    print(f"  Stage: alpha={cfg.stages[0].loss.alpha}, beta={cfg.stages[0].loss.beta}")
    print("  PASSED")


# =============================================================================
# Test 8: Accept Rate Computation
# =============================================================================

def test_accept_rate_bounds():
    """Test that accept rate is always in [0, 1]."""
    print("\nTest 8: Accept rate bounds")
    
    B, S, V = 4, 16, 100
    
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        logits = jax.random.normal(key, (B, 2 * S, V)) * (seed + 1)  # Vary scale
        labels = jax.random.randint(key, (B, 2 * S), 0, V)
        mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
        mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)
        
        class AcceptModel:
            def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
                return logits
        
        batch = {
            "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
            "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
            "labels": labels,
            "loss_mask_ntp": mask_ntp,
            "loss_mask_diff": mask_diff,
            "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
        }
        
        _, (_, _, _, _, _, acc) = loss_and_metrics(
            {}, batch, model=AcceptModel(), dropout_rng=jax.random.PRNGKey(0),
            alpha=1.0, beta=1.0, rho=0.0, chi=0.0, delta=0.0,
            compute_accept=True, accept_top_k=64, accept_max_positions=256
        )
        
        assert 0.0 <= float(acc) <= 1.0, f"Accept rate {acc} out of bounds for seed {seed}"
    
    print("  All accept rates in [0, 1]")
    print("  PASSED")


# =============================================================================
# Test 9: Consistent Behavior Under JIT
# =============================================================================

def test_jit_consistency():
    """Test that loss is consistent between JIT and non-JIT execution."""
    print("\nTest 9: JIT consistency")
    
    B, S, V = 2, 8, 100  # V must be >= accept_top_k
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (B, 2 * S, V))
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)
    
    class JitModel:
        def apply(self, variables, input_ids, rngs, deterministic, attn_bias, position_ids):
            return logits
    
    batch = {
        "input_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "position_ids": jnp.zeros((B, 2 * S), dtype=jnp.int32),
        "labels": labels,
        "loss_mask_ntp": mask_ntp,
        "loss_mask_diff": mask_diff,
        "attn_bias": jnp.zeros((B, 1, 2 * S, 2 * S)),
    }
    
    def compute_loss():
        return loss_and_metrics(
            {}, batch, model=JitModel(), dropout_rng=jax.random.PRNGKey(0),
            alpha=1.0, beta=1.0, rho=0.1, chi=0.1, delta=0.1,
            compute_accept=False, accept_top_k=min(64, V), accept_max_positions=256
        )
    
    # Non-JIT
    result_nojit = compute_loss()
    
    # JIT
    compute_loss_jit = jax.jit(compute_loss)
    result_jit = compute_loss_jit()
    
    # Compare
    total_nojit = float(result_nojit[0])
    total_jit = float(result_jit[0])
    
    assert abs(total_nojit - total_jit) < 1e-5, f"JIT mismatch: {total_nojit} vs {total_jit}"
    
    print(f"  Non-JIT: {total_nojit:.6f}, JIT: {total_jit:.6f}")
    print("  PASSED")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("TiDAR 5-Term Loss Edge Case Tests")
    print("=" * 60)
    
    test_extreme_logits()
    test_single_valid_position()
    test_all_zero_coefficient()
    test_coefficient_additivity()
    test_gradient_finite()
    test_build_train_batch_integration()
    test_config_parsing()
    test_accept_rate_bounds()
    test_jit_consistency()
    
    print("\n" + "=" * 60)
    print("All edge case tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
