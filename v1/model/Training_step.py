# Training_step.py
import jax, jax.numpy as jnp, optax
from functools import partial


@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(
    params,
    opt_state,
    batch,
    *,
    model,          # GiantGPT(nn.Module) – treated as static
    optimizer,      # optax.GradientTransformation – treated as static
    dropout_rng,    # fresh RNG per step (root key for dropout)
):
    """One optimisation step. Returns (new_params, new_opt_state, loss)."""

    def loss_fn(p):
        # --------------------------------------------------------------
        # Pass the **root** dropout key via Flax’s standard rngs mechanism
        # --------------------------------------------------------------
        logits = model.apply(
            {"params": p},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["target"]
        )
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

