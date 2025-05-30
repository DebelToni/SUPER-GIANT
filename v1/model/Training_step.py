import jax, jax.numpy as jnp, optax
from functools import partial

@partial(jax.jit,static_argnames=['model', 'optimizer'])
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng):
    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["target"])
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        return loss

    (loss, grads) = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

