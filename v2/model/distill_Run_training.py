# distill_Run_training.py
import os, sys, math, pickle, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

# --------------------------------------------------------------------------- #
# JAX stack + project imports
# --------------------------------------------------------------------------- #
try:
    import jax
except ImportError:
    print("JAX not found – installing …")
    os.system("pip install jax[cuda12] transformers datasets flax")
    import jax

import jax.numpy as jnp
import optax
import numpy as np
from omegaconf import OmegaConf

from GiantGPT         import GiantGPT
from distill_Training_step    import train_step
from distill_prepare_dataset  import get_data, data_loader
from Save_params      import save_params
from checkpoint_manager import (
    save   as save_ckpt,
    load   as load_ckpt,
    latest as latest_ckpt,
)

from jax import config
config.update("jax_default_matmul_precision", "tensorfloat32")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
Config = OmegaConf.load("Config.yml")

# -------------------------- simple CLI parser ------------------------------ #
# Defaults
checkpoint_dir    = "checkpoints"
checkpoint_every  = 1_000         # optimiser steps
resume_request    = None          # "latest" | path | None

for arg in sys.argv[1:]:
    if arg.startswith("--checkpoint_dir="):
        checkpoint_dir = arg.split("=", 1)[1]
    elif arg.startswith("--checkpoint_every="):
        checkpoint_every = int(arg.split("=", 1)[1])
    elif arg == "--resume":
        resume_request = "latest"
    elif arg.startswith("--resume="):
        resume_request = arg.split("=", 1)[1]
# --------------------------------------------------------------------------- #

def main() -> None:
    # Echo the effective run‑time config (handy for logs)
    print("» Effective training configuration:")
    for k, v in Config.__dict__.items():
        if not k.startswith("__") and not callable(v):
            print(f"{k:>20} = {v}")
    print(f"{'checkpoint_dir':>20} = {checkpoint_dir}")
    print(f"{'checkpoint_every':>20} = {checkpoint_every}")
    print(f"{'resume_request':>20} = {resume_request}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    print("Preparing dataset …")
    train_it, val_it, tokenizer = get_data(
        subset_pct=Config.dataset_percent * 100 if Config.dataset_percent <= 1 else Config.dataset_percent,
        context_length=Config.context_length - 1,
        batch_size=Config.batch_size,
    )
    train_loader = data_loader(train_it)

    # Estimate dataset size for logging
    train_batches = sum(1 for _ in train_loader)
    val_batches = sum(1 for _ in data_loader(val_it)) if val_it else 0
    print(f"train batches: {train_batches}   val batches: {val_batches}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = GiantGPT(
        vocab_size     = len(tokenizer),
        context_length = Config.context_length - 1,
        d_model        = Config.embedding_size,
        n_heads        = Config.num_heads,
        d_ff           = Config.feed_forward_size,
        n_layers       = Config.num_layers,
        dropout_rate   = Config.dropout_rate,
    )
    rng     = jax.random.PRNGKey(0)
    dummy   = jnp.zeros((Config.batch_size, Config.context_length - 1), dtype=jnp.int32)
    params  = model.init(rng, dummy, deterministic=True)["params"]
    save_params(params, "initial_params.pkl")     # optional convenience dump

    # ------------------------------------------------------------------ #
    # Optimiser + LR scheduler
    # ------------------------------------------------------------------ #
    steps_per_epoch = train_batches
    total_steps     = steps_per_epoch * Config.num_epochs
    assert total_steps > 500, "total_steps must exceed warm‑up (500)"

    schedule = optax.warmup_cosine_decay_schedule(
        init_value   = 0.0,
        peak_value   = Config.learning_rate,
        warmup_steps = 500,
        decay_steps  = total_steps - 500,
        end_value    = Config.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate = schedule,
            b1 = 0.9, b2 = 0.95, eps = 1e-8, weight_decay=Config.weight_decay,
        ),
    )
    opt_state    = optimizer.init(params)
    global_step  = 0

    # ------------------------------------------------------------------ #
    # Resume logic
    # ------------------------------------------------------------------ #
    if resume_request:
        if resume_request == "latest":
            ckpt_path = latest_ckpt(checkpoint_dir)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found in '{checkpoint_dir}' to resume from.")
        else:
            ckpt_path = resume_request
        params, global_step = load_ckpt(ckpt_path)
        print(f"▶ Resumed from {ckpt_path}  (global_step={global_step})")
        opt_state = optimizer.init(params)   # re‑seed optimiser state

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    print(f"Training for {Config.num_epochs} epochs with batch size {Config.batch_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    rng = jax.random.PRNGKey(0)

    for epoch in range(Config.num_epochs):
        t0 = time.time()
        for batch in train_loader:
            rng, dropout_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, batch,
                model = model,
                optimizer = optimizer,
                dropout_rng = dropout_rng,
            )
            global_step += 1

            # Console logging
            if global_step % 200 == 0:
                est_total = Config.num_epochs * steps_per_epoch
                print(f"step {global_step:>7}/{est_total:<7} "
                      f"| loss {loss:.4f}  ppl {np.exp(loss):.2f}")

            # Periodic checkpoint
            if global_step % checkpoint_every == 0:
                ckpt_file = save_ckpt(params, global_step, checkpoint_dir)
                print(f"💾 checkpoint → {ckpt_file}")

        dt = time.time() - t0
        print(f"epoch {epoch+1} done in {dt:.1f}s")

    # ------------------------------------------------------------------ #
    # Final save
    # ------------------------------------------------------------------ #
    save_ckpt(params, global_step, checkpoint_dir)
    save_params(params)   # legacy pickle dump
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✔ final parameters & tokenizer saved")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Starting distillation training …")
    main()