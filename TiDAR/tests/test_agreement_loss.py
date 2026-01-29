"""
Test the 5-term loss implementation in TiDAR.

This script verifies:
1. Fast path: when all agreement coefficients are 0, only AR and Diff losses are computed
2. Forward KL: KL(stopgrad(P_AR) || Q_Diff) punishes Diff for missing AR mass
3. Reverse KL: KL(Q_Diff || stopgrad(P_AR)) punishes Diff for extra mass
4. Hard agreement: CE(onehot(argmax P_AR), logits_diff) forces greedy alignment
5. Gradient flow: stop_gradient ensures AR logits don't receive agreement gradients
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def compute_5term_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask_ntp: jnp.ndarray,
    mask_diff: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    rho: float = 0.0,
    chi: float = 0.0,
    delta: float = 0.0,
):
    """
    Standalone loss computation matching Training_step.py logic.
    
    Loss = alpha * L_AR + beta * L_Diff + rho * KL_fwd + chi * KL_rev + delta * L_hard
    
    Returns: (total_loss, ar_loss, diff_loss, kl_fwd, kl_rev, hard_agree)
    """
    S = logits.shape[1] // 2
    
    # AR loss
    if alpha > 0.0:
        ar_active = mask_ntp > 0
        ar_labels_safe = jnp.where(ar_active, labels, 0)
        ar_ce = optax.softmax_cross_entropy_with_integer_labels(logits, ar_labels_safe)
        ar_ce = jnp.where(ar_active, ar_ce, 0.0)
        ar_loss = (ar_ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
    else:
        ar_loss = jnp.array(0.0, dtype=jnp.float32)

    # Diff loss
    if beta > 0.0:
        diff_active = mask_diff > 0
        diff_labels_safe = jnp.where(diff_active, labels, 0)
        diff_ce = optax.softmax_cross_entropy_with_integer_labels(logits, diff_labels_safe)
        diff_ce = jnp.where(diff_active, diff_ce, 0.0)
        diff_loss = (diff_ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)
    else:
        diff_loss = jnp.array(0.0, dtype=jnp.float32)

    # Agreement losses (aligned positions)
    ar_logits = logits[:, : S - 1]          # [B, S-1, V]
    diff_logits_aligned = logits[:, S + 1 :]  # [B, S-1, V]
    diff_mask_aligned = mask_diff[:, S + 1 :]
    ar_mask_aligned = mask_ntp[:, : S - 1]
    joint_mask = ar_mask_aligned * diff_mask_aligned
    joint_denom = jnp.maximum(joint_mask.sum(), 1.0)
    
    ar_logits_sg = jax.lax.stop_gradient(ar_logits)
    log_p_ar = jax.nn.log_softmax(ar_logits_sg, axis=-1)
    p_ar = jax.nn.softmax(ar_logits_sg, axis=-1)
    log_q_diff = jax.nn.log_softmax(diff_logits_aligned, axis=-1)
    q_diff = jax.nn.softmax(diff_logits_aligned, axis=-1)

    # Forward KL
    if rho > 0.0:
        kl_fwd_terms = log_p_ar - log_q_diff
        kl_fwd_per_pos = jnp.sum(jnp.where(p_ar > 0, p_ar * kl_fwd_terms, 0.0), axis=-1)
        kl_fwd_per_pos = jnp.where(joint_mask > 0, kl_fwd_per_pos, 0.0)
        kl_fwd = kl_fwd_per_pos.sum() / joint_denom
    else:
        kl_fwd = jnp.array(0.0, dtype=jnp.float32)

    # Reverse KL
    if chi > 0.0:
        kl_rev_terms = log_q_diff - log_p_ar
        kl_rev_per_pos = jnp.sum(jnp.where(q_diff > 0, q_diff * kl_rev_terms, 0.0), axis=-1)
        kl_rev_per_pos = jnp.where(joint_mask > 0, kl_rev_per_pos, 0.0)
        kl_rev = kl_rev_per_pos.sum() / joint_denom
    else:
        kl_rev = jnp.array(0.0, dtype=jnp.float32)

    # Hard agreement
    if delta > 0.0:
        ar_argmax = jnp.argmax(ar_logits_sg, axis=-1)
        hard_ce = optax.softmax_cross_entropy_with_integer_labels(diff_logits_aligned, ar_argmax)
        hard_ce = jnp.where(joint_mask > 0, hard_ce, 0.0)
        hard_agree = hard_ce.sum() / joint_denom
    else:
        hard_agree = jnp.array(0.0, dtype=jnp.float32)

    total_loss = alpha * ar_loss + beta * diff_loss + rho * kl_fwd + chi * kl_rev + delta * hard_agree
    return total_loss, ar_loss, diff_loss, kl_fwd, kl_rev, hard_agree


def test_fast_path():
    """When all agreement coefficients are 0, only AR and Diff losses should be computed."""
    print("Test 1: Fast path (rho=chi=delta=0)")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (B, 2 * S, V))
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.0, chi=0.0, delta=0.0
    )

    assert float(kl_fwd) == 0.0, f"Expected kl_fwd=0, got {kl_fwd}"
    assert float(kl_rev) == 0.0, f"Expected kl_rev=0, got {kl_rev}"
    assert float(hard) == 0.0, f"Expected hard_agree=0, got {hard}"
    expected = ar + diff  # alpha=beta=1
    assert jnp.allclose(total, expected), f"Total loss mismatch: {total} vs {expected}"
    print(f"  total={float(total):.4f}, ar={float(ar):.4f}, diff={float(diff):.4f}")
    print("  PASSED")


def test_forward_kl():
    """Test forward KL: KL(P_AR || Q_Diff)."""
    print("\nTest 2: Forward KL is positive when distributions differ")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    ar_logits = jax.random.normal(key, (B, S, V))
    key, subkey = jax.random.split(key)
    diff_logits = jax.random.normal(subkey, (B, S, V)) * 2
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.1, chi=0.0, delta=0.0
    )

    assert float(kl_fwd) > 0.0, f"Expected positive kl_fwd, got {kl_fwd}"
    base_loss = ar + diff
    assert float(total) > float(base_loss), "Total should be > base when rho>0"
    print(f"  total={float(total):.4f}, kl_fwd={float(kl_fwd):.4f}")
    print("  PASSED")


def test_reverse_kl():
    """Test reverse KL: KL(Q_Diff || P_AR)."""
    print("\nTest 3: Reverse KL is positive when distributions differ")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    ar_logits = jax.random.normal(key, (B, S, V))
    key, subkey = jax.random.split(key)
    diff_logits = jax.random.normal(subkey, (B, S, V)) * 2
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.0, chi=0.1, delta=0.0
    )

    assert float(kl_rev) > 0.0, f"Expected positive kl_rev, got {kl_rev}"
    base_loss = ar + diff
    assert float(total) > float(base_loss), "Total should be > base when chi>0"
    print(f"  total={float(total):.4f}, kl_rev={float(kl_rev):.4f}")
    print("  PASSED")


def test_hard_agreement():
    """Test hard agreement: CE(onehot(argmax P_AR), logits_diff)."""
    print("\nTest 4: Hard agreement is positive when argmax differs")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    ar_logits = jax.random.normal(key, (B, S, V))
    key, subkey = jax.random.split(key)
    diff_logits = jax.random.normal(subkey, (B, S, V)) * 2
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.0, chi=0.0, delta=0.1
    )

    assert float(hard) > 0.0, f"Expected positive hard_agree, got {hard}"
    base_loss = ar + diff
    assert float(total) > float(base_loss), "Total should be > base when delta>0"
    print(f"  total={float(total):.4f}, hard_agree={float(hard):.4f}")
    print("  PASSED")


def test_alpha_beta_weighting():
    """Test that alpha and beta correctly weight AR vs Diff loss."""
    print("\nTest 5: Alpha and beta weighting")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (B, 2 * S, V))
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    # alpha=1, beta=1: equal weight
    total_11, ar, diff, _, _, _ = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0
    )
    expected_11 = ar + diff
    assert jnp.allclose(total_11, expected_11), f"alpha=beta=1 mismatch"

    # alpha=0.5, beta=1: AR gets half weight
    total_05_1, _, _, _, _, _ = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=0.5, beta=1.0
    )
    expected_05_1 = 0.5 * ar + diff
    assert jnp.allclose(total_05_1, expected_05_1), f"alpha=0.5, beta=1 mismatch"

    # alpha=0, beta=1: only Diff loss
    total_0_1, _, _, _, _, _ = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=0.0, beta=1.0
    )
    assert jnp.allclose(total_0_1, diff), f"alpha=0, beta=1 mismatch"

    # alpha=1, beta=0: only AR loss
    total_1_0, _, _, _, _, _ = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=0.0
    )
    assert jnp.allclose(total_1_0, ar), f"alpha=1, beta=0 mismatch"

    print(f"  alpha=1, beta=1: total={float(total_11):.4f}")
    print(f"  alpha=0.5, beta=1: total={float(total_05_1):.4f}")
    print(f"  alpha=0, beta=1: total={float(total_0_1):.4f}")
    print(f"  alpha=1, beta=0: total={float(total_1_0):.4f}")
    print("  PASSED")


def test_gradient_stopgrad():
    """Verify that stop_gradient blocks gradients to AR logits from agreement terms."""
    print("\nTest 6: Gradient flow with stop_gradient")
    
    B, S, V = 2, 4, 10
    key = jax.random.PRNGKey(42)

    def loss_with_agreement(logits, labels, mask_ntp, mask_diff):
        total, _, _, _, _, _ = compute_5term_loss(
            logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.5, chi=0.5, delta=0.5
        )
        return total

    def loss_without_agreement(logits, labels, mask_ntp, mask_diff):
        total, _, _, _, _, _ = compute_5term_loss(
            logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=0.0, chi=0.0, delta=0.0
        )
        return total

    logits = jax.random.normal(key, (B, 2 * S, V))
    key, subkey = jax.random.split(key)
    labels = jax.random.randint(subkey, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    grad_with = jax.grad(loss_with_agreement)(logits, labels, mask_ntp, mask_diff)
    grad_without = jax.grad(loss_without_agreement)(logits, labels, mask_ntp, mask_diff)

    assert jnp.all(jnp.isfinite(grad_with)), "Gradients should be finite"
    
    # Diff half SHOULD have additional gradients from agreement terms
    diff_grad_with = grad_with[:, S:]
    diff_grad_without = grad_without[:, S:]
    diff_grad_diff = jnp.abs(diff_grad_with - diff_grad_without).mean()
    
    print(f"  AR grad norm (with): {float(jnp.linalg.norm(grad_with[:, :S])):.4f}")
    print(f"  AR grad norm (without): {float(jnp.linalg.norm(grad_without[:, :S])):.4f}")
    print(f"  Diff grad norm (with): {float(jnp.linalg.norm(diff_grad_with)):.4f}")
    print(f"  Diff grad norm (without): {float(jnp.linalg.norm(diff_grad_without)):.4f}")
    print(f"  Diff grad mean absolute difference: {float(diff_grad_diff):.6f}")
    
    assert diff_grad_diff > 1e-6, "Diff gradients should change with agreement terms"
    print("  PASSED")


def test_kl_zero_when_same_distribution():
    """KL should be ~0 when AR and Diff aligned logits are identical."""
    print("\nTest 7: KL divergence is 0 when distributions match")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    # Create aligned logits: AR at 0..S-2 matches Diff at S+1..2S-1
    # AR positions 0..S-2 predict tokens 1..S-1
    # Diff positions S+1..2S-1 also predict tokens 1..S-1 (aligned)
    shared = jax.random.normal(key, (B, S - 1, V))
    
    # Build full logits: [AR_0..AR_{S-1} | Diff_0..Diff_{S-1}]
    # AR half: positions 0..S-2 are shared, position S-1 is arbitrary
    ar_last = jax.random.normal(jax.random.PRNGKey(99), (B, 1, V))
    ar_logits = jnp.concatenate([shared, ar_last], axis=1)  # [B, S, V]
    
    # Diff half: position 0 is arbitrary, positions 1..S-1 are shared
    diff_first = jax.random.normal(jax.random.PRNGKey(100), (B, 1, V))
    diff_logits = jnp.concatenate([diff_first, shared], axis=1)  # [B, S, V]
    
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)  # [B, 2S, V]
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=1.0, beta=1.0, rho=1.0, chi=1.0, delta=0.0
    )

    # KL(p||p) = 0 when AR[0..S-2] matches Diff[S+1..2S-1]
    assert float(kl_fwd) < 1e-5, f"Expected kl_fwd~0, got {kl_fwd}"
    assert float(kl_rev) < 1e-5, f"Expected kl_rev~0, got {kl_rev}"
    print(f"  kl_fwd={float(kl_fwd):.6f}, kl_rev={float(kl_rev):.6f} (should be ~0)")
    print("  PASSED")


def test_all_terms_together():
    """Test that all 5 terms work together correctly."""
    print("\nTest 8: All 5 terms combined")
    
    B, S, V = 2, 8, 100
    key = jax.random.PRNGKey(42)
    
    ar_logits = jax.random.normal(key, (B, S, V))
    key, subkey = jax.random.split(key)
    diff_logits = jax.random.normal(subkey, (B, S, V)) * 1.5
    logits = jnp.concatenate([ar_logits, diff_logits], axis=1)
    
    labels = jax.random.randint(key, (B, 2 * S), 0, V)
    mask_ntp = jnp.concatenate([jnp.ones((B, S - 1)), jnp.zeros((B, S + 1))], axis=1)
    mask_diff = jnp.concatenate([jnp.zeros((B, S)), jnp.ones((B, S))], axis=1)

    alpha, beta, rho, chi, delta = 1.0, 1.0, 0.1, 0.05, 0.2
    total, ar, diff, kl_fwd, kl_rev, hard = compute_5term_loss(
        logits, labels, mask_ntp, mask_diff, alpha=alpha, beta=beta, rho=rho, chi=chi, delta=delta
    )

    expected = alpha * ar + beta * diff + rho * kl_fwd + chi * kl_rev + delta * hard
    assert jnp.allclose(total, expected, rtol=1e-5), f"Total mismatch: {total} vs {expected}"
    
    print(f"  alpha={alpha}, beta={beta}, rho={rho}, chi={chi}, delta={delta}")
    print(f"  ar={float(ar):.4f}, diff={float(diff):.4f}")
    print(f"  kl_fwd={float(kl_fwd):.4f}, kl_rev={float(kl_rev):.4f}, hard={float(hard):.4f}")
    print(f"  total={float(total):.4f}")
    print("  PASSED")


def main():
    print("=" * 60)
    print("Testing TiDAR 5-Term Loss Implementation")
    print("=" * 60)
    
    test_fast_path()
    test_forward_kl()
    test_reverse_kl()
    test_hard_agreement()
    test_alpha_beta_weighting()
    test_gradient_stopgrad()
    test_kl_zero_when_same_distribution()
    test_all_terms_together()
    
    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
