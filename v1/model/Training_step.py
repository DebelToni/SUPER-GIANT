# import jax
# import jax.numpy as jnp
# import optax
#
#
#
# # Get gradients
# grad_fn = jax.value_and_grad(loss_fn)
#
# # Single training step
# @jax.jit
# def train_step(params, opt_state, batch):
#     loss, grads = grad_fn(params, batch)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, opt_state, loss
#
# Training_step.py
import jax, jax.numpy as jnp, optax

def loss_fn(params, batch):
    logits = model.apply({'params': params}, batch['input'])  # forward pass
    targets = batch['target']
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    # Mask out padding positions (assuming pad token id = 0 for example)
    mask = batch['mask']  # 1 for real tokens, 0 for pad
    loss = (loss * mask).sum() / mask.sum()  # average only over real tokens
    return loss

@jax.jit(static_argnames=('model', 'optimizer'))
def train_step(params, opt_state, batch, *, model, optimizer):
    def loss_fn(p):
        logits  = model.apply({"params": p}, batch["input"])
        loss    = optax.softmax_cross_entropy_with_integer_labels(
                      logits, batch["target"])
        loss    = (loss * batch["mask"]).sum() / batch["mask"].sum()
        return loss

    (loss, grads) = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

