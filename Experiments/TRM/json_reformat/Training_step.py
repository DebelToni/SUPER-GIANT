from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_div(tree, denom: jnp.ndarray):
    return jax.tree_util.tree_map(lambda x: x / denom, tree)


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(mask.sum(), 1.0)
    return (values * mask).sum() / denom


@partial(jax.jit, static_argnames=("model", "optimizer", "supervision_steps", "microbatch_size"))
def train_step(
    params,
    opt_state,
    batch,
    *,
    model,
    optimizer,
    dropout_rng,
    supervision_steps: int,
    microbatch_size: int | None = None,
):
    """
    batch:
      input:  (B, L) int32
      target: (B, L) int32
      mask:   (B, L) float32
    """
    input_all = batch["input"]
    target_all = batch["target"]
    mask_all = batch["mask"]

    b = input_all.shape[0]
    if microbatch_size is None:
        microbatch_size = b
    microbatch_size = int(microbatch_size)
    n_micro = b // microbatch_size

    input_all = input_all.reshape(n_micro, microbatch_size, -1)
    target_all = target_all.reshape(n_micro, microbatch_size, -1)
    mask_all = mask_all.reshape(n_micro, microbatch_size, -1)

    def micro_grad_and_metrics(input_ids, target_ids, mask, key_micro):
        def loss_fn(p):
            key_x, key_steps = jax.random.split(key_micro)
            x = model.apply(
                {"params": p},
                input_ids,
                deterministic=False,
                aug_ids=None,
                rngs={"dropout": key_x},
                method=model.encode,
            )
            y, z = model.apply({"params": p}, input_ids, method=model.initial_state)

            bsz = input_ids.shape[0]
            halted = jnp.zeros((bsz,), dtype=jnp.bool_)
            steps = jnp.zeros((bsz,), dtype=jnp.int32)
            total_loss = jnp.array(0.0, dtype=jnp.float32)
            total_ce = jnp.array(0.0, dtype=jnp.float32)
            total_halt = jnp.array(0.0, dtype=jnp.float32)
            last_match = jnp.zeros((bsz,), dtype=jnp.float32)
            last_token_acc = jnp.zeros((bsz,), dtype=jnp.float32)
            step_count = jnp.array(0, dtype=jnp.int32)

            def body_fn(carry, _):
                (
                    step_count,
                    rng,
                    y,
                    z,
                    halted,
                    steps,
                    total_loss,
                    total_ce,
                    total_halt,
                    last_match,
                    last_token_acc,
                ) = carry

                rng, key_step, key_explore = jax.random.split(rng, 3)
                y_n, z_n, logits, q_halt, q_continue, pred = model.apply(
                    {"params": p},
                    x,
                    y,
                    z,
                    deterministic=False,
                    rngs={"dropout": key_step},
                    method=model.step_from_x,
                )

                ce_tokens = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
                token_denom = jnp.maximum(mask.sum(axis=-1), 1.0)
                ce = (ce_tokens * mask).sum(axis=-1) / token_denom

                mask_bool = mask > 0.0
                match = jnp.all(jnp.where(mask_bool, pred == target_ids, True), axis=-1)
                halt_loss = optax.sigmoid_binary_cross_entropy(q_halt, match.astype(jnp.float32))

                active = jnp.logical_not(halted)
                active_f = active.astype(jnp.float32)
                ce_mean = _masked_mean(ce, active_f)
                halt_mean = _masked_mean(halt_loss, active_f)
                step_loss = ce_mean + 0.5 * halt_mean

                total_loss = total_loss + step_loss
                total_ce = total_ce + ce_mean
                total_halt = total_halt + halt_mean
                last_match = match.astype(jnp.float32)

                token_acc = (pred == target_ids).astype(jnp.float32)
                last_token_acc = (token_acc * mask).sum(axis=-1) / token_denom

                steps_next = steps + active.astype(jnp.int32)
                halted_step = steps_next >= supervision_steps
                if model.enable_early_stop and supervision_steps > 1:
                    if model.no_act_continue:
                        halted_step = jnp.logical_or(halted_step, q_halt > model.halt_threshold_logit)
                    else:
                        halted_step = jnp.logical_or(halted_step, q_halt > q_continue)
                    if model.halt_exploration_prob > 0.0:
                        key_explore, key_rand = jax.random.split(key_explore)
                        rand = jax.random.uniform(key_explore, (bsz,))
                        min_halt = jnp.where(
                            rand < model.halt_exploration_prob,
                            jax.random.randint(key_rand, (bsz,), 2, supervision_steps + 1),
                            0,
                        )
                        halted_step = jnp.logical_and(halted_step, steps_next >= min_halt)

                halted = jnp.logical_or(halted, halted_step)
                y = jnp.where(active[:, None, None], y_n, y)
                z = jnp.where(active[:, None, None], z_n, z)
                y = jax.lax.stop_gradient(y)
                z = jax.lax.stop_gradient(z)

                any_active = jnp.any(active)
                step_count = step_count + any_active.astype(jnp.int32)

                new_carry = (
                    step_count,
                    rng,
                    y,
                    z,
                    halted,
                    steps_next,
                    total_loss,
                    total_ce,
                    total_halt,
                    last_match,
                    last_token_acc,
                )
                return new_carry, None

            init_carry = (
                step_count,
                key_steps,
                y,
                z,
                halted,
                steps,
                total_loss,
                total_ce,
                total_halt,
                last_match,
                last_token_acc,
            )
            (
                step_count,
                _rng,
                _y,
                _z,
                _halted,
                _steps,
                total_loss,
                total_ce,
                total_halt,
                last_match,
                last_token_acc,
            ), _ = jax.lax.scan(body_fn, init_carry, xs=None, length=supervision_steps)

            denom = jnp.maximum(step_count.astype(jnp.float32), 1.0)
            loss_mean = total_loss / denom
            ce_mean = total_ce / denom
            halt_mean = total_halt / denom
            solved_acc = last_match.mean()
            token_acc = last_token_acc.mean()
            return loss_mean, (loss_mean, ce_mean, halt_mean, solved_acc, token_acc)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads, metrics

    grads_acc = _tree_zeros_like(params)
    loss_acc = jnp.array(0.0, dtype=jnp.float32)
    ce_acc = jnp.array(0.0, dtype=jnp.float32)
    halt_acc = jnp.array(0.0, dtype=jnp.float32)
    solved_acc = jnp.array(0.0, dtype=jnp.float32)
    token_acc = jnp.array(0.0, dtype=jnp.float32)

    def micro_body(carry, micro_idx):
        grads_acc, loss_acc, ce_acc, halt_acc, solved_acc, token_acc = carry
        input_ids = input_all[micro_idx]
        target_ids = target_all[micro_idx]
        mask = mask_all[micro_idx]

        key_micro = jax.random.fold_in(dropout_rng, micro_idx)
        grads, (loss, ce, halt, solved, tok) = micro_grad_and_metrics(input_ids, target_ids, mask, key_micro)

        grads_acc = _tree_add(grads_acc, grads)
        loss_acc = loss_acc + loss
        ce_acc = ce_acc + ce
        halt_acc = halt_acc + halt
        solved_acc = solved_acc + solved
        token_acc = token_acc + tok
        return (grads_acc, loss_acc, ce_acc, halt_acc, solved_acc, token_acc), None

    (grads_acc, loss_acc, ce_acc, halt_acc, solved_acc, token_acc), _ = jax.lax.scan(
        micro_body,
        (grads_acc, loss_acc, ce_acc, halt_acc, solved_acc, token_acc),
        jnp.arange(n_micro),
    )

    denom = jnp.array(float(n_micro), dtype=jnp.float32)
    grads = _tree_div(grads_acc, denom)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    loss = loss_acc / denom
    ce = ce_acc / denom
    halt = halt_acc / denom
    solved_acc = solved_acc / denom
    token_acc = token_acc / denom
    return new_params, opt_state, (loss, ce, halt, solved_acc, token_acc)


@partial(jax.jit, static_argnames=("model", "supervision_steps"))
def eval_step(params, batch, *, model, supervision_steps: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    x = model.apply(
        {"params": params},
        batch["input"],
        deterministic=True,
        aug_ids=None,
        method=model.encode,
    )
    y, z = model.apply({"params": params}, batch["input"], method=model.initial_state)
    pred = jnp.zeros_like(batch["target"])
    logits = jnp.zeros((batch["target"].shape[0], batch["target"].shape[1], model.vocab_size), dtype=jnp.float32)

    for _ in range(supervision_steps):
        y, z, logits, _q_halt, _q_continue, pred = model.apply(
            {"params": params},
            x,
            y,
            z,
            deterministic=True,
            method=model.step_from_x,
        )

    mask = batch["mask"]
    mask_bool = mask > 0.0
    match = jnp.all(jnp.where(mask_bool, pred == batch["target"], True), axis=-1)
    solved_acc = match.mean()
    token_acc = _masked_mean((pred == batch["target"]).astype(jnp.float32), mask)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
    ce = _masked_mean(ce, mask)
    return solved_acc, token_acc, ce
