from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng) -> Tuple:
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
        active = (mask_ntp + mask_diff) > 0
        labels_safe = jnp.where(active, labels, 0)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_safe)
        ntp_loss = (ce * mask_ntp).sum() / jnp.maximum(mask_ntp.sum(), 1.0)
        diff_loss = (ce * mask_diff).sum() / jnp.maximum(mask_diff.sum(), 1.0)
        return ntp_loss + diff_loss, (ntp_loss, diff_loss)

    (loss, (ntp_loss, diff_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, ntp_loss, diff_loss


__all__ = ["train_step"]
