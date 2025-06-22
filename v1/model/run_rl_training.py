# run_rl_training.py
"""Reinforcement‑learning fine‑tuning script.

This file mirrors *Run_training.py* but swaps cross‑entropy for a simple
REINFORCE objective with a moving‑average baseline.

The policy is the same GiantGPT decoder‑only model; we only change the
loss and the data pipeline.
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import math
import pickle
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from math_tokenizer import build_math_tokenizer
from math_env import sample_problem
from GiantGPT import GiantGPT

# -------------------------------------------------------------------------
# Configuration -----------------------------------------------------------
# -------------------------------------------------------------------------
CFG = OmegaConf.load("config_rl.yml")

# for reproducibility
SEED = 42

# -------------------------------------------------------------------------
# Utilities ---------------------------------------------------------------
# -------------------------------------------------------------------------

def pad_to_length(ids: list[int], *, pad: int, length: int) -> list[int]:
    """Right‑pad ``ids`` with ``pad`` up to ``length`` elements."""
    if len(ids) > length:
        # keep the right‑most tokens (GPT sees the most recent context)
        ids = ids[-length:]
    return ids + [pad] * (length - len(ids))


def log_prob_of_action(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Return log‑probability of *action* under unnormalised *logits*."""
    log_probs = jax.nn.log_softmax(logits)
    batch_idx = jnp.arange(action.shape[0])
    return log_probs[batch_idx, action]

# -------------------------------------------------------------------------
# Build objects -----------------------------------------------------------
# -------------------------------------------------------------------------
print("Building tokenizer…")
tokenizer = build_math_tokenizer()
PAD = tokenizer.pad_token_id

print("Initialising model…")
model = GiantGPT(
    vocab_size=tokenizer.vocab_size,
    context_length=CFG.context_length,
    d_model=CFG.embedding_size,
    n_heads=CFG.num_heads,
    d_ff=CFG.feed_forward_size,
    n_layers=CFG.num_layers,
    dropout_rate=CFG.dropout_rate,
)

rng = jax.random.PRNGKey(SEED)
params = model.init(rng, jnp.zeros((1, CFG.context_length), dtype=jnp.int32))["params"]

# Adam + weight‑decay (there is no LR schedule because RL updates are noisy)
optimizer = optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay)
opt_state = optimizer.init(params)

# Moving‑average reward baseline -----------------------------------------
baseline = 0.0
mom = CFG.baseline_momentum

# JIT‑compiled helper to get logits for the *next* token -------------------
@functools.partial(jax.jit, static_argnums=0)
def next_token_logits(params: dict, inputs: jax.Array) -> jax.Array:
    """Return unnormalised logits for position *len(inputs)-1*."""
    logits = model.apply({"params": params}, inputs, deterministic=True)
    return logits[:, inputs.shape[1] - 1, :]  # (batch, vocab)

# -------------------------------------------------------------------------
# Main update loop --------------------------------------------------------
# -------------------------------------------------------------------------
print("Starting RL fine‑tuning…")

for step in range(1, CFG.num_updates + 1):
    # ------------------------------------------------------------------
    # 1. Generate a batch of prompts and ground truths
    # ------------------------------------------------------------------
    exprs: list[str] = []
    truths: list[int] = []
    for _ in range(CFG.batch_size):
        expr, truth = sample_problem()  # two‑operand expression
        exprs.append(expr)
        truths.append(truth)

    token_ids = [pad_to_length(tokenizer(expr, add_special_tokens=False).input_ids,
                               pad=PAD, length=CFG.context_length) for expr in exprs]
    tokens = jnp.asarray(token_ids, dtype=jnp.int32)  # (B, L)

    # ------------------------------------------------------------------
    # 2. Sample an answer token & compute log‑probabilities
    # ------------------------------------------------------------------
    rng, sub = jax.random.split(rng)
    subkeys = jax.random.split(sub, CFG.batch_size)

    logits = next_token_logits(params, tokens)      # (B, V)
    sampled = jax.vmap(jax.random.categorical)(subkeys, logits)  # (B,)
    logp = log_prob_of_action(logits, sampled)      # (B,)

    # ------------------------------------------------------------------
    # 3. Reward (Python side, not inside the TPU/GPU graph)
    # ------------------------------------------------------------------
    pred_nums = [int(tokenizer.convert_ids_to_tokens(int(t))) for t in np.asarray(sampled)]
    rewards = np.array([1.0 if p == t else 0.0 for p, t in zip(pred_nums, truths)], dtype=np.float32)
    avg_reward = rewards.mean()

    # moving baseline
    baseline = mom * baseline + (1 - mom) * avg_reward
    advantages = rewards - baseline  # broadcasting OK (scalar baseline)

    # ------------------------------------------------------------------
    # 4. Compute REINFORCE loss & update parameters
    # ------------------------------------------------------------------
    def loss_fn(params_: dict, tokens_: jax.Array, actions_: jax.Array, adv_: jax.Array) -> jax.Array:
        lgt = next_token_logits(params_, tokens_)
        lp  = log_prob_of_action(lgt, actions_)
        return -jnp.mean(adv_ * lp)

    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, sampled, jnp.asarray(advantages))
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # ------------------------------------------------------------------
    # 5. Logging & checkpointing
    # ------------------------------------------------------------------
    if step % 100 == 0:
        print(f"step {step:>6} | loss {loss:.4f} | reward {avg_reward:.3f} | baseline {baseline:.3f}")

    if step % 5000 == 0:
        ckpt_path = Path(f"rl_checkpoint_step{step}.pkl")
        with ckpt_path.open("wb") as f:
            pickle.dump(params, f)
        print(f"✔ saved checkpoint → {ckpt_path}")

