from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=["model", "optimizer", "alpha", "agreement_lambda"])
def train_step(
    params,
    opt_state,
    batch,
    *,
    model,
    optimizer,
    dropout_rng,
    alpha: float = 1.0,
    agreement_lambda: float = 0.0,
) -> Tuple:
    """
    Single training step with optional agreement loss.

    Loss formulation:
        L_tidar = (alpha * L_AR_CE + L_Diff_CE) / (1 + alpha)
        L_total = L_tidar + agreement_lambda * KL(stopgrad(p_AR) || p_Diff)

    Args:
        params: Model parameters.
        opt_state: Optimizer state.
        batch: Dict with input_ids, labels, loss_mask_ntp, loss_mask_diff, attn_bias, position_ids.
        model: Flax model.
        optimizer: Optax optimizer.
        dropout_rng: PRNG key for dropout.
        alpha: Weight for AR loss relative to Diff loss. Default 1.0 (equal weight).
        agreement_lambda: Coefficient for agreement KL loss. 0.0 disables it.

    Returns:
        Tuple of (new_params, new_opt_state, total_loss, ntp_loss, diff_loss, agreement_loss).
        agreement_loss is 0.0 when agreement_lambda == 0.0.
    """

    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["input_ids"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
            attn_bias=batch["attn_bias"],
            position_ids=batch["position_ids"],
        )
        labels = batch["labels"]
        mask_ntp = batch["loss_mask_ntp"]
        mask_diff = batch["loss_mask_diff"]

        # Avoid NaN in cross-entropy for masked positions
        active = (mask_ntp + mask_diff) > 0
        labels_safe = jnp.where(active, labels, 0)

        # Per-token cross-entropy
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_safe)

        # Separate NTP (AR) and Diff losses
        ntp_loss = (ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
        diff_loss = (ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)

        # Combined TiDAR loss with alpha weighting
        tidar_loss = (alpha * ntp_loss + diff_loss) / (1.0 + alpha)

        # Agreement loss: KL(stopgrad(p_AR) || p_Diff)
        # Only computed when agreement_lambda > 0
        if agreement_lambda > 0.0:
            # Sequence layout: [clean (S) | diff (S)]
            S = logits.shape[1] // 2
            ar_logits = logits[:, :S]    # clean/AR half
            diff_logits = logits[:, S:]  # diffusion half

            # Teacher: AR distribution (detached)
            log_p_ar = jax.nn.log_softmax(ar_logits, axis=-1)
            p_ar = jax.nn.softmax(jax.lax.stop_gradient(ar_logits), axis=-1)
            log_p_ar_sg = jax.lax.stop_gradient(log_p_ar)

            # Student: Diff distribution
            log_p_diff = jax.nn.log_softmax(diff_logits, axis=-1)

            # KL divergence per position: sum over vocab
            # KL(p_ar || p_diff) = sum_v p_ar(v) * (log p_ar(v) - log p_diff(v))
            kl_per_pos = (p_ar * (log_p_ar_sg - log_p_diff)).sum(axis=-1)  # [B, S]

            # Mask: use diff mask for the second half (positions 0..S-1 in diff == positions S..2S-1 overall)
            # mask_diff is [B, 2S], we want the second half
            diff_mask_for_kl = mask_diff[:, S:]  # [B, S]

            # Average KL over valid diff positions
            agreement_loss = (kl_per_pos * diff_mask_for_kl).sum() / jnp.maximum(diff_mask_for_kl.sum(), 1.0)
            total_loss = tidar_loss + agreement_lambda * agreement_loss
        else:
            agreement_loss = jnp.array(0.0)
            total_loss = tidar_loss

        return total_loss, (ntp_loss, diff_loss, agreement_loss)

    (loss, (ntp_loss, diff_loss, agreement_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, ntp_loss, diff_loss, agreement_loss


__all__ = ["train_step"]
