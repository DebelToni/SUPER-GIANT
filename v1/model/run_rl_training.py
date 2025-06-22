# run_rl_training.py  — fixed version
"""Reinforcement‑learning fine‑tuning for GiantGPT on toy arithmetic.

Changes vs. first draft
-----------------------
* **Removed** the problematic `static_argnums` usage that made JAX try to
  hash the parameters dict.  The new code relies on JAX’s default pytree
  handling, so `params` is treated as a normal dynamic argument and no
  hash attempt is made.
* **Re‑organised** the jitted functions so that the only *static* value
  is the model’s `apply` method; all pytrees (params, tokens, RNG keys)
  are dynamic and safe.

You can drop this file straight into your repo, overwriting the previous
version.
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # nicer on GPU mem

import functools
import itertools
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core import FrozenDict

import yaml

# ---------------------------------------------------------------------------
# Local code – make sure these imports resolve inside your project layout
# ---------------------------------------------------------------------------
from math_env import sample_batch  # freshly added helper for arithmetic
from GiantGPT import GiantGPT       # your existing model definition
from math_tokenizer import MathTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("config_rl.yml")
_cfg = yaml.safe_load(CONFIG_PATH.read_text())

BATCH_SIZE   = _cfg.get("batch_size", 64)
NUM_UPDATES  = _cfg.get("num_updates", 50_000)
LR           = _cfg.get("learning_rate", 3e-4)
BASE_MOMENT  = _cfg.get("baseline_momentum", 0.9)
CTX_LEN      = _cfg.get("context_length", 32)
SEED         = _cfg.get("seed", 42)
LOG_EVERY    = _cfg.get("log_every", 200)
CHECK_EVERY  = _cfg.get("checkpoint_every", 5_000)
SAVE_DIR     = Path(_cfg.get("save_dir", "checkpoints"))
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: initialise model & optimiser
# ---------------------------------------------------------------------------
print("Building model…")
tokenizer = MathTokenizer.load("math_tokenizer_data")

model = GiantGPT(
    vocab_size=len(tokenizer),
    **_cfg["model"],  # e.g. emb_dim, n_heads, n_layers, …
)

def init_model(rng: jax.random.PRNGKey):
    """Creates initial params (Flax FrozenDict)."""
    dummy_tokens = jnp.zeros((1, CTX_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy_tokens)["params"]
    return params

rng = jax.random.PRNGKey(SEED)
params = init_model(rng)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(LR, weight_decay=_cfg.get("weight_decay", 0.01)),
)

# Running baseline (moving average of recent rewards)
baseline = jnp.array(0.0)

# ---------------------------------------------------------------------------
# Jitted helpers
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(0,))
def get_next_token_logits(apply_fn, params: FrozenDict, tokens: jnp.ndarray):
    """Returns logits for the *next* position (last index) for each example."""
    logits = apply_fn({"params": params}, tokens)     # (B, T, V)
    return logits[:, -1, :]                           # (B, V)

@jax.jit
def compute_loss_and_grads(params: FrozenDict,
                           tokens: jnp.ndarray,
                           actions: jnp.ndarray,
                           advantages: jnp.ndarray,
                           rng_key: jax.random.PRNGKey):
    """REINFORCE loss; returns (loss, grads)."""
    def loss_fn(p):
        logits = get_next_token_logits(model.apply, p, tokens)   # (B, V)
        log_probs = jax.nn.log_softmax(logits)
        # Gather log‑p of selected actions
        logp_action = jnp.take_along_axis(log_probs,
                                          actions[:, None],
                                          axis=1).squeeze(1)
        # REINFORCE objective (negative for gradient descent)
        return -(advantages * logp_action).mean()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print("Starting RL fine‑tuning…")
wall0 = time.time()

for step in range(1, NUM_UPDATES + 1):
    # ---------------------------------------------------------------------
    # 1.  Generate on‑the‑fly batch of problems
    # ---------------------------------------------------------------------
    expr_batch, truth_batch = sample_batch(BATCH_SIZE)  # list[str], list[int]
    prompt_tokens = tokenizer.encode_batch(expr_batch, pad_to=CTX_LEN)  # (B, T)
    prompt_tokens = jnp.array(prompt_tokens, dtype=jnp.int32)

    # ---------------------------------------------------------------------
    # 2.  Policy: sample an answer token from model distribution
    # ---------------------------------------------------------------------
    rng, sub = jax.random.split(rng)
    logits = get_next_token_logits(model.apply, state.params, prompt_tokens)
    action = jax.random.categorical(sub, logits)           # (B,)

    # ---------------------------------------------------------------------
    # 3.  Reward: 1 if action token == truth else 0
    # ---------------------------------------------------------------------
    action_int = np.vectorize(tokenizer.token_to_int.__getitem__)(action)
    rewards = (action_int == np.array(truth_batch)).astype(np.float32)
    rewards = jnp.array(rewards)

    # Moving‑average baseline (scalar)
    baseline = BASE_MOMENT * baseline + (1 - BASE_MOMENT) * rewards.mean()
    advantages = rewards - baseline  # broadcast

    # ---------------------------------------------------------------------
    # 4.  Compute loss and update params
    # ---------------------------------------------------------------------
    loss, grads = compute_loss_and_grads(state.params,
                                         prompt_tokens,
                                         action,
                                         advantages,
                                         rng)
    state = state.apply_gradients(grads=grads)

    # ---------------------------------------------------------------------
    # 5.  Logging / checkpoint
    # ---------------------------------------------------------------------
    if step % LOG_EVERY == 0:
        took = time.time() - wall0
        print(f"step {step:>6d} \t loss {loss:.4f} \t avgR {rewards.mean():.3f} "
              f"\t baseline {float(baseline):.3f} \t {took/LOG_EVERY:.3f}s/it")
        wall0 = time.time()

    if step % CHECK_EVERY == 0:
        ckpt_path = SAVE_DIR / f"ckpt_{step:06d}.npz"
        print(f"Saving → {ckpt_path}")
        with ckpt_path.open("wb") as f:
            for arr in jax.tree_util.tree_leaves(state.params):
                np.save(f, np.array(arr), allow_pickle=False)

print("Done!")

