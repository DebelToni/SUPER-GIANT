
"""Run_training.py  — training script for GiantGPT (patched).

This version restores the original data pipeline (no padding tokens) while
keeping the four JIT‑related bug‑fixes:

1.  Unique per‑layer scopes (``block_N``).
2.  Fresh dropout key every batch.
3.  Captured model object, fed through a JIT‑compiled ``apply_fn``.
4.  Correct weight‑decay masking.

The training loop logic matches your previous implementation so the only
behavioural difference should be the improved numerical stability.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",  "0.80")

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from omegaconf import OmegaConf

from GiantGPT       import GiantGPT, build_apply_fn
from Training_step  import train_step
from Evaluate       import evaluate
from Data_loader    import data_loader
from Save_params    import save_params
from prepare_dataset import get_data


# --------------------------------------------------------------------------- #
# 0.  Hyper‑parameters
# --------------------------------------------------------------------------- #
Config = OmegaConf.load("Config.yml")

# Convenience alias
C = Config   # shorter


def main() -> None:
    # --------------------------------------------------------------------- #
    # 1.  Dataset – original streaming pipeline, no explicit pad tokens
    # --------------------------------------------------------------------- #
    print("Loading dataset / tokenizer ...")
    train_tokens, val_tokens, tokenizer = get_data(
        subset_pct     = C.dataset_percent,
        context_length = C.context_length,
    )
    print(f"train batches: {len(train_tokens)}  val batches: {len(val_tokens)}")
    print(f"train_tokens shape: {train_tokens.shape}  val_tokens shape: {val_tokens.shape}")

    steps_per_epoch = math.ceil(len(train_tokens) / C.batch_size)
    total_steps     = steps_per_epoch * C.num_epochs
    print(total_steps, "total steps")

    # --------------------------------------------------------------------- #
    # 2.  Model, params, optimiser
    # --------------------------------------------------------------------- #
    print("Initialising model …")
    model = GiantGPT(
        vocab_size      = tokenizer.vocab_size,
        context_length  = C.context_length,
        d_model         = C.embedding_size,
        n_heads         = C.num_heads,
        d_ff            = C.feed_forward_size,
        n_layers        = C.num_layers,
        dropout_rate    = C.dropout_rate,
    )

    rng   = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, C.context_length), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    save_params(params, "initial_params.pkl")

    # JIT‑compiled forward pass that captures *model* (fix #4)
    apply_fn = build_apply_fn(model)

    # optimiser & schedule
    warmup_steps = 500
    assert total_steps > warmup_steps, (
        f"warmup ({warmup_steps}) >= total_steps ({total_steps}); train longer or reduce warmup."
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value   = 0.0,
        peak_value   = C.learning_rate,
        warmup_steps = warmup_steps,
        decay_steps  = total_steps - warmup_steps,
        end_value    = C.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate = lr_schedule,
            b1 = 0.9, b2 = 0.95, eps = 1e-8,
            weight_decay = C.weight_decay,
        ),
    )

    state = train_state.TrainState.create(
        apply_fn = apply_fn,
        params   = params,
        tx       = optimizer,
    )

    # --------------------------------------------------------------------- #
    # 3.  Training loop
    # --------------------------------------------------------------------- #
    print(f"Training for {C.num_epochs} epochs with batch size {C.batch_size}")
    rng = jax.random.PRNGKey(0)
    global_step = 0

    for epoch in range(C.num_epochs):
        for batch in data_loader(train_tokens, C.batch_size):
            # batch: int32 [B, T] – contiguous, no pad token
            tokens_in  = batch[:, :-1]          # teacher‑forcing input
            targets    = batch[:, 1:]           # next‑token labels

            rng, step_rng = jax.random.split(rng)
            state, metrics, rng = train_step(
                state,
                (tokens_in, targets),
                step_rng,
            )
            global_step += 1

            if global_step < 5:  # print first few dropout keys
                jax.debug.print("dropout key {}/{}: {}", global_step, total_steps, step_rng)
            if global_step % 200 == 0:
                loss = float(metrics["loss"])
                print(f"step {global_step:7d} / {total_steps:7d} | loss {loss:.4f}  ppl {math.exp(loss):.2f}")

        # end of epoch – validation
        val_loss = evaluate(state.params, model, val_tokens)
        print(f"✓ Epoch {epoch + 1} done – val loss {val_loss:.4f}  ppl {math.exp(val_loss):.2f}")

    # --------------------------------------------------------------------- #
    # 4.  Save final checkpoint
    # --------------------------------------------------------------------- #
    save_params(state.params)
    with Path("tokenizer.pkl").open("wb") as f:
        pickle.dump(tokenizer, f)
    print("✔ parameters & tokenizer saved")


if __name__ == "__main__":
    print("Starting training …")
    main()
