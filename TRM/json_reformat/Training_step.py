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


def _tree_scale(tree, scale: jnp.ndarray):
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


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

    inv_steps = jnp.array(1.0 / float(supervision_steps), dtype=jnp.float32)

    def micro_grad_and_metrics(input_ids, target_ids, mask, key_micro):
        keys = jax.random.split(key_micro, supervision_steps + 1)
        key_x = keys[0]
        step_keys = keys[1:]

        def step0_loss_fn(p):
            x = model.apply(
                {"params": p},
                input_ids,
                deterministic=False,
                aug_ids=None,
                rngs={"dropout": key_x},
                method=model.encode,
            )
            y, z = model.apply({"params": p}, input_ids, method=model.initial_state)
            y, z, logits, q_logit, pred = model.apply(
                {"params": p},
                x,
                y,
                z,
                deterministic=False,
                rngs={"dropout": step_keys[0]},
                method=model.step_from_x,
            )

            ce = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            ce = _masked_mean(ce, mask)

            mask_bool = mask > 0.0
            match = jnp.all(jnp.where(mask_bool, pred == target_ids, True), axis=-1).astype(jnp.float32)
            halt = optax.sigmoid_binary_cross_entropy(q_logit, match).mean()

            token_acc = _masked_mean((pred == target_ids).astype(jnp.float32), mask)
            step_loss = (ce + halt).astype(jnp.float32)

            aux = (y, z, step_loss, ce, halt, match, token_acc)
            return step_loss, aux

        (loss0, (y, z, step_loss0, ce0, halt0, match0, token_acc0)), grads0 = jax.value_and_grad(
            step0_loss_fn, has_aux=True
        )(params)

        total_loss = step_loss0
        total_ce = ce0.astype(jnp.float32)
        total_halt = halt0.astype(jnp.float32)
        last_match = match0
        last_token_acc = token_acc0.astype(jnp.float32)

        y = jax.lax.stop_gradient(y)
        z = jax.lax.stop_gradient(z)

        def body(carry, key_step):
            y, z, grads, total_loss, total_ce, total_halt = carry

            def step_loss_fn(p):
                x = model.apply(
                    {"params": p},
                    input_ids,
                    deterministic=False,
                    aug_ids=None,
                    rngs={"dropout": key_x},
                    method=model.encode,
                )
                y_n, z_n, logits, q_logit, pred = model.apply(
                    {"params": p},
                    x,
                    y,
                    z,
                    deterministic=False,
                    rngs={"dropout": key_step},
                    method=model.step_from_x,
                )

                ce = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
                ce = _masked_mean(ce, mask)

                mask_bool = mask > 0.0
                match = jnp.all(jnp.where(mask_bool, pred == target_ids, True), axis=-1).astype(jnp.float32)
                halt = optax.sigmoid_binary_cross_entropy(q_logit, match).mean()
                token_acc = _masked_mean((pred == target_ids).astype(jnp.float32), mask)
                step_loss = (ce + halt).astype(jnp.float32)
                aux = (y_n, z_n, ce, halt, match, token_acc)
                return step_loss, aux

            (step_loss, (y_n, z_n, ce, halt, match, token_acc)), grads_i = jax.value_and_grad(
                step_loss_fn, has_aux=True
            )(params)

            grads = _tree_add(grads, grads_i)
            total_loss = total_loss + step_loss
            total_ce = total_ce + ce.astype(jnp.float32)
            total_halt = total_halt + halt.astype(jnp.float32)

            y_n = jax.lax.stop_gradient(y_n)
            z_n = jax.lax.stop_gradient(z_n)
            y = y_n
            z = z_n
            return (y, z, grads, total_loss, total_ce, total_halt), (match, token_acc)

        if supervision_steps > 1:
            (y, z, grads_sum, total_loss, total_ce, total_halt), (matches, token_accs) = jax.lax.scan(
                body,
                (y, z, grads0, total_loss, total_ce, total_halt),
                step_keys[1:],
            )
            last_match = matches[-1]
            last_token_acc = token_accs[-1].astype(jnp.float32)
        else:
            grads_sum = grads0

        grads_mean = _tree_scale(grads_sum, inv_steps)
        loss_mean = total_loss * inv_steps
        ce_mean = total_ce * inv_steps
        halt_mean = total_halt * inv_steps
        solved_acc = last_match.mean()
        return grads_mean, (loss_mean, ce_mean, halt_mean, solved_acc, last_token_acc)

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
        y, z, logits, _q_logit, pred = model.apply(
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
