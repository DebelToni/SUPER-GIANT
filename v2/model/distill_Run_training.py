# Run_training.py
import os, sys, time, pickle
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import jax, jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.training import train_state
from omegaconf import OmegaConf

from GiantGPT import GiantGPT
from distill_Training_step import train_step
from distill_prepare_dataset import get_data, data_loader

Config = OmegaConf.load("Config.yml")

def save_ckpt(params, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"params_step_{step:08d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(params, f)
    return path

def main():
    print("» Effective training configuration:")
    for k, v in Config.items():
        print(f"{k:>22} = {v}")

    # Dataset: iterators that yield static-shaped batches of dicts
    train_it, val_it, tokenizer = get_data(
        subset_pct=Config.dataset_percent * 100 if Config.dataset_percent <= 1 else Config.dataset_percent,
        context_length=Config.context_length - 1,  # we produce (T) after shift inside prep; keep as-is if you prefer
        batch_size=Config.batch_size,
    )
    train_loader = data_loader(train_it)

    # Model
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=Config.context_length - 1,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.feed_forward_size,
        n_layers=Config.num_layers,
        dropout_rate=Config.dropout_rate,
    )

    rng = jax.random.PRNGKey(0)
    x_init = jnp.zeros((Config.batch_size, Config.context_length - 1), dtype=jnp.int32)
    params = model.init(rng, x_init, deterministic=True)["params"]

    # Optimizer
    schedule = optax.cosine_decay_schedule(
        init_value=Config.learning_rate,
        decay_steps=1_000_000,
        alpha=0.1,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=Config.weight_decay)
    opt_state = optimizer.init(params)

    # Training
    steps = 0
    checkpoint_dir = "checkpoints"
    steps_per_epoch_guess = 1000  # purely for logging; adjust if you prefer

    for epoch in range(Config.num_epochs):
        t0 = time.time()
        for batch in train_loader:
            rng, dropout_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, batch,
                model=model, optimizer=optimizer, dropout_rng=dropout_rng
            )
            steps += 1
            if steps % 200 == 0:
                print(f"epoch {epoch+1}/{Config.num_epochs}  step {steps}  "
                      f"loss {float(loss):.4f}  ppl {np.exp(float(loss)):.2f}")
            if steps % 2000 == 0:
                ck = save_ckpt(params, steps, checkpoint_dir)
                print(f"💾 saved checkpoint → {ck}")
        dt = time.time() - t0
        print(f"epoch {epoch+1} done in {dt:.1f}s")

    # Final save
    ck = save_ckpt(params, steps, checkpoint_dir)
    print(f"✔ final checkpoint → {ck}")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    main()

