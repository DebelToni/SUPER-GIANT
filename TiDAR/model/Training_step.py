from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


def _compute_accept_rate(
    logits,
    mask_ntp,
    mask_diff,
    *,
    accept_top_k: int,
    accept_max_positions: int,
):
    """Theoretical acceptance rate on the last batch row (greedy-style metric)."""
    # Align AR logits (predict token t at position t-1) with Diff logits (predict token t).
    S = logits.shape[1] // 2
    if S <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    max_positions = min(accept_max_positions, S - 1)
    ar_logits_last = logits[-1, : S - 1]
    diff_logits_last = logits[-1, S + 1 :]
    ar_mask = mask_ntp[-1, : S - 1]
    diff_mask = mask_diff[-1, S + 1 :]
    joint_mask = (ar_mask * diff_mask) > 0

    pos_idx = jnp.nonzero(joint_mask, size=max_positions, fill_value=0)[0]
    valid_count = jnp.minimum(joint_mask.sum().astype(jnp.int32), max_positions)
    pos_mask = (jnp.arange(max_positions) < valid_count).astype(jnp.float32)

    ar_logits_pos = ar_logits_last[pos_idx]
    diff_logits_pos = diff_logits_last[pos_idx]

    ar_top_vals, ar_top_idx = jax.lax.top_k(ar_logits_pos, accept_top_k)
    diff_top_vals, _ = jax.lax.top_k(diff_logits_pos, accept_top_k)

    ar_log_norm = jax.nn.logsumexp(ar_top_vals, axis=-1, keepdims=True)
    diff_log_norm = jax.nn.logsumexp(diff_top_vals, axis=-1, keepdims=True)

    p_ar = jnp.exp(ar_top_vals - ar_log_norm)
    diff_logits_for_ar = jnp.take_along_axis(diff_logits_pos, ar_top_idx, axis=-1)
    p_diff = jnp.exp(diff_logits_for_ar - diff_log_norm)

    accept_per_pos = jnp.minimum(p_ar, p_diff).sum(axis=-1)
    denom = jnp.maximum(valid_count.astype(jnp.float32), 1.0)
    return (accept_per_pos * pos_mask).sum() / denom


def _compute_greedy_accept_rate(
    logits,
    mask_ntp,
    mask_diff,
    *,
    accept_max_positions: int,
):
    """
    Greedy acceptance rate: fraction of positions where argmax(AR) == argmax(Diff).
    
    This is the metric that directly measures speculative decoding performance in greedy mode.
    Computed over the full batch (not just last row) for stability.
    """
    S = logits.shape[1] // 2
    if S <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    
    # AR logits at positions 0..S-2 predict tokens 1..S-1
    ar_logits = logits[:, : S - 1]          # [B, S-1, V]
    # Diff logits at positions S+1..2S-1 predict tokens 1..S-1 (aligned)
    diff_logits = logits[:, S + 1 :]        # [B, S-1, V]
    
    # Masks
    ar_mask = mask_ntp[:, : S - 1]          # [B, S-1]
    diff_mask = mask_diff[:, S + 1 :]       # [B, S-1]
    joint_mask = (ar_mask * diff_mask) > 0  # [B, S-1]
    
    # Argmax for both
    ar_argmax = jnp.argmax(ar_logits, axis=-1)    # [B, S-1]
    diff_argmax = jnp.argmax(diff_logits, axis=-1)  # [B, S-1]
    
    # Match where both are valid
    matches = (ar_argmax == diff_argmax) & joint_mask
    
    # Compute rate
    num_matches = matches.sum()
    num_valid = jnp.maximum(joint_mask.sum(), 1.0)
    
    return num_matches / num_valid


def _compute_alignment_losses(
    logits: jnp.ndarray,
    mask_ntp: jnp.ndarray,
    mask_diff: jnp.ndarray,
    *,
    rho: float,
    chi: float,
    delta: float,
    delta_masked: float,
    eta: float,
    eta_T: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute KL, hard-agreement, and distillation losses between AR and Diff distributions.
    
    Returns: (kl_fwd_loss, kl_rev_loss, hard_agree_loss, hard_agree_masked_loss, distill_loss)
    """
    S = logits.shape[1] // 2
    
    # AR logits at positions 0..S-2 predict tokens 1..S-1
    ar_logits = logits[:, : S - 1]          # [B, S-1, V]
    # Diff logits at positions S+1..2S-1 predict tokens 1..S-1 (aligned)
    diff_logits_aligned = logits[:, S + 1 :]  # [B, S-1, V]
    
    # Mask for valid positions (both AR and Diff active)
    diff_mask_aligned = mask_diff[:, S + 1 :]  # [B, S-1]
    ar_mask_aligned = mask_ntp[:, : S - 1]      # [B, S-1]
    joint_mask = ar_mask_aligned * diff_mask_aligned  # [B, S-1]
    joint_denom = jnp.maximum(joint_mask.sum(), 1.0)
    
    # Stop-gradient AR (all agreement losses only update Diff)
    ar_logits_sg = jax.lax.stop_gradient(ar_logits)
    
    # Probabilities (log and regular)
    log_p_ar = jax.nn.log_softmax(ar_logits_sg, axis=-1)
    p_ar = jax.nn.softmax(ar_logits_sg, axis=-1)
    log_q_diff = jax.nn.log_softmax(diff_logits_aligned, axis=-1)
    q_diff = jax.nn.softmax(diff_logits_aligned, axis=-1)
    
    # Forward KL: KL(P_AR || Q_Diff)
    if rho > 0.0:
        kl_fwd_terms = log_p_ar - log_q_diff
        kl_fwd_per_pos = jnp.sum(jnp.where(p_ar > 0, p_ar * kl_fwd_terms, 0.0), axis=-1)
        kl_fwd_per_pos = jnp.where(joint_mask > 0, kl_fwd_per_pos, 0.0)
        kl_fwd_loss = kl_fwd_per_pos.sum() / joint_denom
    else:
        kl_fwd_loss = jnp.array(0.0, dtype=jnp.float32)
    
    # Reverse KL: KL(Q_Diff || P_AR)
    if chi > 0.0:
        kl_rev_terms = log_q_diff - log_p_ar
        kl_rev_per_pos = jnp.sum(jnp.where(q_diff > 0, q_diff * kl_rev_terms, 0.0), axis=-1)
        kl_rev_per_pos = jnp.where(joint_mask > 0, kl_rev_per_pos, 0.0)
        kl_rev_loss = kl_rev_per_pos.sum() / joint_denom
    else:
        kl_rev_loss = jnp.array(0.0, dtype=jnp.float32)
    
    # Hard agreement losses: full and prefix-capped-to-first-mismatch variants.
    if (delta > 0.0) or (delta_masked > 0.0):
        ar_argmax = jnp.argmax(ar_logits_sg, axis=-1)  # [B, S-1]
        diff_argmax = jnp.argmax(diff_logits_aligned, axis=-1)  # [B, S-1]
        hard_ce = optax.softmax_cross_entropy_with_integer_labels(
            diff_logits_aligned, ar_argmax
        )  # [B, S-1]

        if delta > 0.0:
            hard_ce_full = jnp.where(joint_mask > 0, hard_ce, 0.0)
            hard_agree_loss = hard_ce_full.sum() / joint_denom
        else:
            hard_agree_loss = jnp.array(0.0, dtype=jnp.float32)

        if delta_masked > 0.0:
            # Keep only positions up to the first mismatch (inclusive) per row.
            mismatch = (ar_argmax != diff_argmax) & (joint_mask > 0)
            mismatch_count_before = jnp.cumsum(mismatch.astype(jnp.int32), axis=-1) - mismatch.astype(jnp.int32)
            prefix_mask = (mismatch_count_before == 0) & (joint_mask > 0)
            prefix_denom = jnp.maximum(prefix_mask.sum(), 1.0)
            hard_ce_prefix = jnp.where(prefix_mask, hard_ce, 0.0)
            hard_agree_masked_loss = hard_ce_prefix.sum() / prefix_denom
        else:
            hard_agree_masked_loss = jnp.array(0.0, dtype=jnp.float32)
    else:
        hard_agree_loss = jnp.array(0.0, dtype=jnp.float32)
        hard_agree_masked_loss = jnp.array(0.0, dtype=jnp.float32)
    
    # Soft distillation: KL(softmax(AR/T) || softmax(Diff/T)) with AR stopgrad
    # Uses the same aligned positions as greedy acceptance (AR 0..S-2 vs Diff S+1..2S-1).
    if eta > 0.0:
        temp = jnp.asarray(eta_T, dtype=jnp.float32)
        temp = jnp.maximum(temp, 1.0e-4)
        log_p_ar_t = jax.nn.log_softmax(ar_logits_sg / temp, axis=-1)
        p_ar_t = jnp.exp(log_p_ar_t)
        log_q_diff_t = jax.nn.log_softmax(diff_logits_aligned / temp, axis=-1)
        kl_t = log_p_ar_t - log_q_diff_t
        distill_per_pos = jnp.sum(p_ar_t * kl_t, axis=-1)
        distill_per_pos = jnp.where(joint_mask > 0, distill_per_pos, 0.0)
        distill_loss = (distill_per_pos.sum() / joint_denom) * (temp * temp)
    else:
        distill_loss = jnp.array(0.0, dtype=jnp.float32)
    
    return kl_fwd_loss, kl_rev_loss, hard_agree_loss, hard_agree_masked_loss, distill_loss


def _compute_topk_set_distill_loss(
    logits: jnp.ndarray,
    mask_ntp: jnp.ndarray,
    mask_diff: jnp.ndarray,
    *,
    gamma_topk: int,
    eps: float = 1.0e-8,
) -> jnp.ndarray:
    """
    Top-K set distillation loss:
    Encourage Diff to put probability mass on AR's top-K set at drafted positions.
    """
    S = logits.shape[1] // 2
    if S <= 1 or gamma_topk <= 0:
        return jnp.array(0.0, dtype=jnp.float32)

    ar_logits = logits[:, : S - 1]
    diff_logits_aligned = logits[:, S + 1 :]

    diff_mask_aligned = mask_diff[:, S + 1 :]
    ar_mask_aligned = mask_ntp[:, : S - 1]
    joint_mask = ar_mask_aligned * diff_mask_aligned
    joint_denom = jnp.maximum(joint_mask.sum(), 1.0)

    ar_logits_sg = jax.lax.stop_gradient(ar_logits)
    _, ar_top_idx = jax.lax.top_k(ar_logits_sg, gamma_topk)

    diff_top_logits = jnp.take_along_axis(diff_logits_aligned, ar_top_idx, axis=-1)
    log_mass = jax.nn.logsumexp(diff_top_logits, axis=-1) - jax.nn.logsumexp(diff_logits_aligned, axis=-1)
    mass = jnp.exp(log_mass)
    loss_per_pos = -jnp.log(mass + eps)
    loss_per_pos = jnp.where(joint_mask > 0, loss_per_pos, 0.0)
    return loss_per_pos.sum() / joint_denom


def loss_and_metrics(
    params,
    batch,
    *,
    model,
    dropout_rng,
    # Loss coefficients (each can be 0 to skip computation)
    alpha: float,      # AR NTP loss coefficient
    beta: float,       # Diffusion loss coefficient
    rho: float,        # Forward KL: KL(P_AR || Q_Diff)
    chi: float,        # Reverse KL: KL(Q_Diff || P_AR)
    delta: float,      # Hard agreement: CE(onehot(argmax P_AR), logits_diff)
    eta: float,        # Soft distillation: KL(softmax(AR/T) || softmax(Diff/T))
    eta_T: float,      # Distillation temperature
    gamma: float,      # Top-K set distillation loss coefficient
    gamma_topk: int,   # Top-K size for set distillation
    # Acceptance metric
    compute_accept,
    accept_top_k: int,
    accept_max_positions: int,
    delta_masked: float = 0.0,  # Hard agreement capped to first mismatch (inclusive)
) -> Tuple:
    """
    Compute TiDAR training loss with 8 configurable terms:
    
    Loss = alpha * L_AR + beta * L_Diff + rho * KL_fwd + chi * KL_rev
           + delta * L_hard + delta_masked * L_hard_prefix + eta * L_distill + gamma * L_topk
    
    Where:
      - L_AR: AR next-token prediction CE loss (clean half, shifted)
      - L_Diff: Diffusion denoising CE loss (diff half, aligned with clean tokens)
      - KL_fwd: KL(stopgrad(P_AR) || Q_Diff) - punishes Diff for missing AR mass
      - KL_rev: KL(Q_Diff || stopgrad(P_AR)) - punishes Diff for extra mass
      - L_hard: CE(onehot(argmax stopgrad(P_AR)), logits_diff) - greedy agreement
      - L_hard_prefix: L_hard on positions up to first AR/Diff argmax mismatch (inclusive)
      - L_distill: KL(softmax(AR/T) || softmax(Diff/T)) on drafted positions (AR stopgrad)
      - L_topk: -log(sum(q_diff[ar_topk])) on drafted positions (AR stopgrad)
    
    All AR terms are stopgrad'd to prevent gradients flowing into AR from Diff losses.
    Terms with coefficient == 0 are skipped entirely (no compute).
    """
    logits = model.apply(
        {"params": params},
        batch["input_ids"],
        rngs={"dropout": dropout_rng},
        deterministic=False,
        attn_bias=batch["attn_bias"],
        position_ids=batch["position_ids"],
    )
    labels = jnp.asarray(batch["labels"])
    mask_ntp = jnp.asarray(batch["loss_mask_ntp"])
    mask_diff = jnp.asarray(batch["loss_mask_diff"])

    # ---------------------------------------------------------------------------
    # Term 1: AR NTP loss (alpha)
    # Positions 0..S-2 predict tokens 1..S-1
    # ---------------------------------------------------------------------------
    if alpha > 0.0:
        ar_active = mask_ntp > 0
        ar_labels_safe = jnp.where(ar_active, labels, 0)
        ar_ce = optax.softmax_cross_entropy_with_integer_labels(logits, ar_labels_safe)
        ar_ce = jnp.where(ar_active, ar_ce, 0.0)
        ar_loss = (ar_ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
    else:
        ar_loss = jnp.array(0.0, dtype=jnp.float32)

    # ---------------------------------------------------------------------------
    # Term 2: Diffusion CE loss (beta)
    # Positions S..2S-1 predict tokens 0..S-1 (aligned denoising)
    # ---------------------------------------------------------------------------
    if beta > 0.0:
        diff_active = mask_diff > 0
        diff_labels_safe = jnp.where(diff_active, labels, 0)
        diff_ce = optax.softmax_cross_entropy_with_integer_labels(logits, diff_labels_safe)
        diff_ce = jnp.where(diff_active, diff_ce, 0.0)
        diff_loss = (diff_ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)
    else:
        diff_loss = jnp.array(0.0, dtype=jnp.float32)

    # ---------------------------------------------------------------------------
    # Terms 3-6: KL divergences, hard agreement, and distillation
    # ---------------------------------------------------------------------------
    if (rho > 0.0) or (chi > 0.0) or (delta > 0.0) or (delta_masked > 0.0) or (eta > 0.0):
        (
            kl_fwd_loss,
            kl_rev_loss,
            hard_agree_loss,
            hard_agree_masked_loss,
            distill_loss,
        ) = _compute_alignment_losses(
            logits,
            mask_ntp,
            mask_diff,
            rho=rho,
            chi=chi,
            delta=delta,
            delta_masked=delta_masked,
            eta=eta,
            eta_T=eta_T,
        )
    else:
        kl_fwd_loss = jnp.array(0.0, dtype=jnp.float32)
        kl_rev_loss = jnp.array(0.0, dtype=jnp.float32)
        hard_agree_loss = jnp.array(0.0, dtype=jnp.float32)
        hard_agree_masked_loss = jnp.array(0.0, dtype=jnp.float32)
        distill_loss = jnp.array(0.0, dtype=jnp.float32)

    # ---------------------------------------------------------------------------
    # Term 7: Top-K set distillation (gamma)
    # ---------------------------------------------------------------------------
    if (gamma > 0.0) and (gamma_topk > 0):
        topk_loss = _compute_topk_set_distill_loss(
            logits,
            mask_ntp,
            mask_diff,
            gamma_topk=gamma_topk,
        )
    else:
        topk_loss = jnp.array(0.0, dtype=jnp.float32)

    # ---------------------------------------------------------------------------
    # Total loss
    # ---------------------------------------------------------------------------
    total_loss = (
        alpha * ar_loss
        + beta * diff_loss
        + rho * kl_fwd_loss
        + chi * kl_rev_loss
        + delta * hard_agree_loss
        + delta_masked * hard_agree_masked_loss
        + eta * distill_loss
        + gamma * topk_loss
    )

    # Report a single hard-agreement metric (full + masked) for logging.
    hard_agree_report = hard_agree_loss + hard_agree_masked_loss

    # ---------------------------------------------------------------------------
    # Acceptance metric (optional, for logging)
    # ---------------------------------------------------------------------------
    accept_rate = jax.lax.cond(
        compute_accept,
        lambda _: _compute_accept_rate(
            logits,
            mask_ntp,
            mask_diff,
            accept_top_k=accept_top_k,
            accept_max_positions=accept_max_positions,
        ),
        lambda _: jnp.array(0.0, dtype=jnp.float32),
        operand=None,
    )
    
    # Greedy acceptance rate (argmax match) - always compute for greedy decoding metric
    greedy_accept_rate = jax.lax.cond(
        compute_accept,
        lambda _: _compute_greedy_accept_rate(
            logits,
            mask_ntp,
            mask_diff,
            accept_max_positions=accept_max_positions,
        ),
        lambda _: jnp.array(0.0, dtype=jnp.float32),
        operand=None,
    )

    return total_loss, (
        ar_loss,
        diff_loss,
        kl_fwd_loss,
        kl_rev_loss,
        hard_agree_report,
        distill_loss,
        topk_loss,
        accept_rate,
        greedy_accept_rate,
    )


def loss_and_grad(
    params,
    batch,
    *,
    model,
    dropout_rng,
    alpha: float,
    beta: float,
    rho: float,
    chi: float,
    delta: float,
    eta: float,
    eta_T: float,
    gamma: float,
    gamma_topk: int,
    compute_accept,
    accept_top_k: int,
    accept_max_positions: int,
    delta_masked: float = 0.0,
):
    def loss_fn(p):
        return loss_and_metrics(
            p,
            batch,
            model=model,
            dropout_rng=dropout_rng,
            alpha=alpha,
            beta=beta,
            rho=rho,
            chi=chi,
            delta=delta,
            delta_masked=delta_masked,
            eta=eta,
            eta_T=eta_T,
            gamma=gamma,
            gamma_topk=gamma_topk,
            compute_accept=compute_accept,
            accept_top_k=accept_top_k,
            accept_max_positions=accept_max_positions,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


@partial(
    jax.jit,
    static_argnames=[
        "model",
        "optimizer",
        "alpha",
        "beta",
        "rho",
        "chi",
        "delta",
        "delta_masked",
        "eta",
        "eta_T",
        "gamma",
        "gamma_topk",
        "accept_top_k",
        "accept_max_positions",
    ],
)
def train_step(
    params,
    opt_state,
    batch,
    *,
    model,
    optimizer,
    dropout_rng,
    alpha: float = 1.0,
    beta: float = 1.0,
    rho: float = 0.0,
    chi: float = 0.0,
    delta: float = 0.0,
    delta_masked: float = 0.0,
    eta: float = 0.0,
    eta_T: float = 1.0,
    gamma: float = 0.0,
    gamma_topk: int = 0,
    compute_accept=False,
    accept_top_k: int = 64,
    accept_max_positions: int = 256,
):
    (
        loss,
        (
            ar_loss,
            diff_loss,
            kl_fwd,
            kl_rev,
            hard_agree,
            distill_loss,
            topk_loss,
            accept_rate,
            greedy_accept_rate,
        ),
    ), grads = loss_and_grad(
        params,
        batch,
        model=model,
        dropout_rng=dropout_rng,
        alpha=alpha,
        beta=beta,
        rho=rho,
        chi=chi,
        delta=delta,
        delta_masked=delta_masked,
        eta=eta,
        eta_T=eta_T,
        gamma=gamma,
        gamma_topk=gamma_topk,
        compute_accept=compute_accept,
        accept_top_k=accept_top_k,
        accept_max_positions=accept_max_positions,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return (
        new_params,
        opt_state,
        loss,
        ar_loss,
        diff_loss,
        kl_fwd,
        kl_rev,
        hard_agree,
        distill_loss,
        topk_loss,
        accept_rate,
        greedy_accept_rate,
    )


__all__ = ["loss_and_metrics", "loss_and_grad", "train_step"]
