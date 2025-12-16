from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=("model", "optimizer", "supervision_steps"))
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng, supervision_steps: int):
    """
    batch:
      puzzle:   (B, 81) int32  (0 for blanks)
      solution: (B, 81) int32  (1..9)
      aug_ids:  (B,)    int32
    """

    def loss_fn(p):
        keys = jax.random.split(dropout_rng, supervision_steps + 1)

        x = model.apply(
            {"params": p},
            batch["puzzle"],
            deterministic=False,
            aug_ids=batch["aug_ids"],
            rngs={"dropout": keys[0]},
            method=model.encode,
        )
        y, z = model.apply({"params": p}, batch["puzzle"], method=model.initial_state)

        total = jnp.array(0.0, jnp.float32)
        total_ce = jnp.array(0.0, jnp.float32)
        total_halt = jnp.array(0.0, jnp.float32)

        last_match = jnp.zeros((batch["puzzle"].shape[0],), dtype=jnp.float32)
        last_token_acc = jnp.array(0.0, dtype=jnp.float32)

        for i in range(supervision_steps):
            y, z, logits, q_logit, pred = model.apply(
                {"params": p},
                x,
                y,
                z,
                deterministic=False,
                rngs={"dropout": keys[i + 1]},
                method=model.step_from_x,
            )

            ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["solution"]).mean()
            match = jnp.all(pred == batch["solution"], axis=-1).astype(jnp.float32)
            halt = optax.sigmoid_binary_cross_entropy(q_logit, match).mean()
            token_acc = (pred == batch["solution"]).mean().astype(jnp.float32)

            total = total + (ce + halt)
            total_ce = total_ce + ce
            total_halt = total_halt + halt

            last_match = match
            last_token_acc = token_acc
            y = jax.lax.stop_gradient(y)
            z = jax.lax.stop_gradient(z)

        inv = jnp.array(1.0 / float(supervision_steps), dtype=jnp.float32)
        loss = total * inv
        ce = total_ce * inv
        halt = total_halt * inv
        solved_acc = last_match.mean()
        return loss, (ce, halt, solved_acc, last_token_acc)

    (loss, (ce, halt, solved_acc, token_acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, (loss, ce, halt, solved_acc, token_acc)


@partial(jax.jit, static_argnames=("model", "supervision_steps"))
def eval_step(params, batch, *, model, supervision_steps: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    x = model.apply(
        {"params": params},
        batch["puzzle"],
        deterministic=True,
        aug_ids=batch["aug_ids"],
        method=model.encode,
    )
    y, z = model.apply({"params": params}, batch["puzzle"], method=model.initial_state)
    pred = jnp.zeros_like(batch["solution"])
    logits = jnp.zeros((batch["solution"].shape[0], batch["solution"].shape[1], model.vocab_size), dtype=jnp.float32)

    for _ in range(supervision_steps):
        y, z, logits, _q_logit, pred = model.apply(
            {"params": params},
            x,
            y,
            z,
            deterministic=True,
            method=model.step_from_x,
        )

    match = jnp.all(pred == batch["solution"], axis=-1)
    solved_acc = match.mean()
    token_acc = (pred == batch["solution"]).mean()
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["solution"]).mean()
    return solved_acc, token_acc, ce
