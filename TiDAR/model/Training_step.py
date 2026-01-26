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
    # Theoretical acceptance rate on the last batch row.
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


def loss_and_metrics(
    params,
    batch,
    *,
    model,
    dropout_rng,
    loss_alpha: float,
    agreement_lambda: float,
    agreement_temperature: float,
    compute_accept,
    accept_top_k: int,
    accept_max_positions: int,
) -> Tuple:
    logits = model.apply(
        {"params": params},
        batch["input_ids"],
        rngs={"dropout": dropout_rng},
        deterministic=False,
        attn_bias=batch["attn_bias"],
        position_ids=batch["position_ids"],
    )
    labels = batch["labels"]
    mask_ntp = batch["loss_mask_ntp"]
    mask_diff = batch["loss_mask_diff"]

    active = (mask_ntp + mask_diff) > 0
    labels_safe = jnp.where(active, labels, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_safe)
    ce = jnp.where(active, ce, 0.0)
    ntp_loss = (ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
    diff_loss = (ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)
    tidar_loss = (loss_alpha * ntp_loss + diff_loss) / (1.0 + loss_alpha)

    if agreement_lambda > 0.0:
        # Sequence layout: [clean (S) | diff (S)]
        # Align AR (predict token t+1 at position t) with Diff (predict token t+1 at position t+1).
        S = logits.shape[1] // 2
        ar_logits = logits[:, : S - 1]
        diff_logits = logits[:, S + 1 :]
        temperature = jnp.asarray(agreement_temperature, dtype=jnp.float32)
        temperature = jnp.maximum(temperature, 1e-6)
        ar_logits_scaled = ar_logits / temperature
        diff_logits_scaled = diff_logits / temperature
        log_p_ar = jax.nn.log_softmax(ar_logits_scaled, axis=-1)
        p_ar = jax.nn.softmax(jax.lax.stop_gradient(ar_logits_scaled), axis=-1)
        log_p_ar_sg = jax.lax.stop_gradient(log_p_ar)
        log_p_diff = jax.nn.log_softmax(diff_logits_scaled, axis=-1)
        kl_terms = log_p_ar_sg - log_p_diff
        kl_per_pos = jnp.sum(jnp.where(p_ar > 0, p_ar * kl_terms, 0.0), axis=-1)
        diff_mask_for_kl = mask_diff[:, S + 1 :]
        kl_per_pos = jnp.where(diff_mask_for_kl > 0, kl_per_pos, 0.0)
        agreement_loss = kl_per_pos.sum() / jnp.maximum(diff_mask_for_kl.sum(), 1.0)
        total_loss = tidar_loss + agreement_lambda * agreement_loss
    else:
        agreement_loss = jnp.array(0.0)
        total_loss = tidar_loss

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

    return total_loss, (ntp_loss, diff_loss, agreement_loss, accept_rate)


def loss_and_grad(
    params,
    batch,
    *,
    model,
    dropout_rng,
    loss_alpha: float,
    agreement_lambda: float,
    agreement_temperature: float,
    compute_accept,
    accept_top_k: int,
    accept_max_positions: int,
):
    def loss_fn(p):
        return loss_and_metrics(
            p,
            batch,
            model=model,
            dropout_rng=dropout_rng,
            loss_alpha=loss_alpha,
            agreement_lambda=agreement_lambda,
            agreement_temperature=agreement_temperature,
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
        "loss_alpha",
        "agreement_lambda",
        "agreement_temperature",
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
    loss_alpha: float = 1.0,
    agreement_lambda: float = 0.0,
    agreement_temperature: float = 1.0,
    compute_accept=False,
    accept_top_k: int = 64,
    accept_max_positions: int = 256,
):
    (loss, (ntp_loss, diff_loss, agreement_loss, accept_rate)), grads = loss_and_grad(
        params,
        batch,
        model=model,
        dropout_rng=dropout_rng,
        loss_alpha=loss_alpha,
        agreement_lambda=agreement_lambda,
        agreement_temperature=agreement_temperature,
        compute_accept=compute_accept,
        accept_top_k=accept_top_k,
        accept_max_positions=accept_max_positions,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, ntp_loss, diff_loss, agreement_loss, accept_rate


__all__ = ["loss_and_metrics", "loss_and_grad", "train_step"]
