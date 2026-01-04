from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(
    params,
    opt_state,
    batch,
    *,
    model,
    optimizer,
    dropout_rng,
    grad_mask: Optional[dict] = None,
):
    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["encoder_tokens"],
            batch["decoder_input"],
            encoder_mask=batch.get("encoder_mask"),
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["decoder_target"])
        loss = (loss * batch["decoder_mask"]).sum() / batch["decoder_mask"].sum()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    if grad_mask is not None:
        grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)

    grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, grad_norm


@partial(jax.jit, static_argnames=["model"])
def eval_step(params, batch, *, model):
    logits = model.apply(
        {"params": params},
        batch["encoder_tokens"],
        batch["decoder_input"],
        encoder_mask=batch.get("encoder_mask"),
        deterministic=True,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["decoder_target"])
    loss = (loss * batch["decoder_mask"]).sum() / batch["decoder_mask"].sum()
    return loss
