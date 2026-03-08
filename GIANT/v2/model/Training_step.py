from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax


def loss_and_grad(
    params,
    batch,
    *,
    model,
    dropout_rng,
    axis_name: Optional[str] = None,
):
    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
        mask = batch["mask"].astype(jnp.float32)
        loss_numer = (loss * mask).sum(dtype=jnp.float32)
        mask_sum = mask.sum(dtype=jnp.float32)
        if axis_name is not None:
            loss_numer = jax.lax.psum(loss_numer, axis_name)
            mask_sum = jax.lax.psum(mask_sum, axis_name)
        denom = jnp.maximum(mask_sum, 1.0)
        return loss_numer / denom

    return jax.value_and_grad(loss_fn)(params)


@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng, axis_name: Optional[str] = None):
    (loss, grads) = loss_and_grad(
        params,
        batch,
        model=model,
        dropout_rng=dropout_rng,
        axis_name=axis_name,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


__all__ = ["loss_and_grad", "train_step"]
