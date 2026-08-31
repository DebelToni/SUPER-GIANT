from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax

def _language_model_loss(logits, batch, *, axis_name: Optional[str]):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
    mask = batch["mask"].astype(jnp.float32)
    loss_numer = (loss * mask).sum(dtype=jnp.float32)
    mask_sum = mask.sum(dtype=jnp.float32)
    if axis_name is not None:
        loss_numer = jax.lax.psum(loss_numer, axis_name)
        mask_sum = jax.lax.psum(mask_sum, axis_name)
    return loss_numer / jnp.maximum(mask_sum, 1.0)


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
        return _language_model_loss(logits, batch, axis_name=axis_name)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)
    return loss, grads


def adapter_loss_and_grad(
    adapter_params,
    base_params,
    batch,
    *,
    model,
    dropout_rng,
    axis_name: Optional[str] = None,
):
    """Differentiate only the adapter collection while the base stays immutable."""

    def loss_fn(adapters):
        logits = model.apply(
            {"params": base_params, "adapters": adapters},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )
        return _language_model_loss(logits, batch, axis_name=axis_name)

    loss, grads = jax.value_and_grad(loss_fn)(adapter_params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)
    return loss, grads


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


__all__ = ["adapter_loss_and_grad", "loss_and_grad", "train_step"]
