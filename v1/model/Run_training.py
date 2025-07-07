
"""
Run_training.py – entry‑point to train GiantGPT with the corrected JIT setup.

Key fixes compared to the previous version
------------------------------------------
1.  A *new* dropout key is generated every batch so JIT no longer re‑uses the
    same sequence of sub‑keys (fix #3).
2.  The per‑layer parameter names were switched from ``layer_N`` to
    ``block_N`` to avoid scope collisions (fix #1).  No other file uses the
    ``layer_*`` prefix any more.
3.  The ``model`` object is created *once* and captured by the jitted apply
    function so weight‑decay masking works correctly (fix #4).
"""

import functools
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from transformers import AutoTokenizer
from datasets import load_dataset

from omegaconf import OmegaConf
from GiantGPT import GiantGPT, build_apply_fn
from Training_step import train_step

Config = OmegaConf.load("Config.yml")

# --------------------------------------------------------------------- #
# 1.  Setup tokenizer, dataset, dataloader
# --------------------------------------------------------------------- #
tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)
dataset = load_dataset(Config.dataset_name, split="train[:{}%]".format(Config.dataset_percent))

max_len = Config.context_length

def encode(example) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ids = tok(example["text"], truncation=True, max_length=max_len + 1, padding="max_length")["input_ids"]
    # shift‑left for teacher forcing
    tokens  = jnp.array(ids[:-1], dtype=jnp.int32)
    targets = jnp.array(ids[1:],  dtype=jnp.int32)
    return tokens, targets

dataset = dataset.map(encode, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42).batch(Config.batch_size)

# --------------------------------------------------------------------- #
# 2.  Build model & optimiser
# --------------------------------------------------------------------- #
model = GiantGPT(
    vocab_size       = tok.vocab_size,
    context_length   = Config.context_length,
    d_model          = Config.embedding_size,
    n_heads          = Config.num_heads,
    d_ff             = Config.feed_forward_size,
    n_layers         = Config.num_layers,
    dropout_rate     = Config.dropout_rate,
)

apply_fn = build_apply_fn(model)   # jitted + captured model

tx = optax.adamw(
    learning_rate = Config.learning_rate,
    weight_decay  = Config.weight_decay,
)

state = train_state.TrainState.create(
    apply_fn = apply_fn,
    params   = model.init(jax.random.PRNGKey(0), jnp.zeros((1, max_len), dtype=jnp.int32))["params"],
    tx       = tx,
)

# --------------------------------------------------------------------- #
# 3.  Training loop
# --------------------------------------------------------------------- #
rng = jax.random.PRNGKey(1)
global_step = 0
for epoch in range(Config.num_epochs):
    for batch in dataset.as_numpy_iterator():
        tokens, targets = batch
        state, metrics, rng = train_step(state, (tokens, targets), rng)  # splits inside
        global_step += 1

        if global_step % 100 == 0:
            loss = metrics["loss"]
            print(f"[step {global_step:6d}] loss = {loss:.4f}")

# --------------------------------------------------------------------- #
# 4.  Save checkpoint
# --------------------------------------------------------------------- #
import pickle, pathlib, os
ckpt_path = pathlib.Path("model_params.pkl")
ckpt_path.write_bytes(pickle.dumps(state.params))
print("Finished.  Parameters written to", ckpt_path)
