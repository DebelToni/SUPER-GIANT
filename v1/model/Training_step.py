
"""
Training_step.py – single optimisation step for GiantGPT.
Fixes:
  • Passes a fresh dropout key to the model every batch (rng argument, not rngs).
  • Compatible with the revised `block_*` parameter tree structure.
"""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, vocab_size, dtype=logits.dtype)
    return optax.softmax_cross_entropy(logits, one_hot).mean()

def train_step(state: train_state.TrainState, batch, rng):
    """Runs one optimiser step.

    Parameters
    ----------
    state
        Flax :pyclass:`~flax.training.train_state.TrainState`.
        ``state.apply_fn`` must be *giant_gpt_apply* (see ``GiantGPT.py``).
    batch
        Tuple ``(tokens, targets)`` – both ``int32`` with shape ``[B, T]``.
    rng
        PRNG key. The function will split it and return the *next* key.

    Returns
    -------
    new_state
        Updated :pyclass:`~flax.training.train_state.TrainState`.
    metrics
        ``dict`` with the scalar loss.
    new_rng
        The PRNG key that should be fed into the *next* call.
    """
    tokens, targets = batch

    # ------- RNG handling (fix #3) --------------------------------------
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(params):
        logits = state.apply_fn(
            params,
            None,                   # no KV‑cache during training
            tokens,
            rng=dropout_rng,        # <— correct key placement
            deterministic=False,
            enable_kv_cache=False,
        )
        loss = cross_entropy_loss(logits, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    new_state = state.apply_gradients(grads=grads)
    metrics = { "loss": loss }
    return new_state, metrics, new_rng
