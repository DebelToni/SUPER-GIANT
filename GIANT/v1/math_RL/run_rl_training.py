

from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from math_env import sample_batch, encode_batch
from GiantGPT import GiantGPT
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from checkpoint_io import save_npz

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")
RL_Config = OmegaConf.load("config_rl.yml")

BATCH_SIZE   = RL_Config.batch_size
NUM_UPDATES  = RL_Config.num_updates
LR           = RL_Config.learning_rate
BASE_MOMENT  = RL_Config.baseline_momentum
CTX_LEN      = RL_Config.context_length
SEED         = 42
LOG_EVERY    = RL_Config.print_every
CHECK_EVERY  = RL_Config.save_every
SAVE_DIR     = Path("checkpoints_rl")
SAVE_DIR.mkdir(exist_ok=True)

print("Building model…")
if Config.use_custom_tokenizer:
    _tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Config.custom_tokenizer_path
    )
else:
    _tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer = _tokenizer

id2num = np.full(tokenizer.vocab_size, -1, dtype=np.int32)
for tok, idx in tokenizer.get_vocab().items():
    if len(tok) == 3 and tok.isdigit():
        id2num[idx] = int(tok)

id2num = jnp.array(id2num)

model = GiantGPT(
        vocab_size = tokenizer.vocab_size,
        context_length    = Config.context_length,
        d_model    = Config.embedding_size,
        n_heads    = Config.num_heads,
        d_ff       = Config.feed_forward_size,
        n_layers   = Config.num_layers,
        dropout_rate = Config.dropout_rate,
    )

def init_model(rng: jax.random.PRNGKey):
    dummy_tokens = jnp.zeros((1, CTX_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy_tokens)["params"]
    return params

rng = jax.random.PRNGKey(SEED)
params = init_model(rng)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(LR, weight_decay=RL_Config.weight_decay),
)

baseline = jnp.array(0.0)

@functools.partial(jax.jit, static_argnums=(0,))
def get_next_token_logits(apply_fn,
                          params: FrozenDict,
                          tokens: jnp.ndarray):
    logits = apply_fn({"params": params},
                      tokens,
                      deterministic=True)
    return logits[:, -1, :]

print("Starting RL fine‑tuning…")
wall0 = time.time()





@jax.jit
def compute_loss_and_grads_no_enthropy(params,
                           tokens: jnp.ndarray,
                           idx:    jnp.ndarray,
                           actions: jnp.ndarray,
                           advantages: jnp.ndarray,
                           ):
    def loss_fn(p):
        logits_full = model.apply({"params": p},
                                  tokens,
                                  deterministic=True)

        logits = jnp.take_along_axis(logits_full,
                                     idx[:, None, None],
                                     axis=1).squeeze(1)

        log_probs   = jax.nn.log_softmax(logits)
        logp_action = jnp.take_along_axis(log_probs,
                                          actions[:, None], 1).squeeze(1)
        return -(advantages * logp_action).mean()

    return jax.value_and_grad(loss_fn)(params)



ENTROPY_COEF = 0.01

@jax.jit
def compute_loss_and_grads(params,
                           tokens: jnp.ndarray,
                           idx:    jnp.ndarray,
                           actions: jnp.ndarray,
                           advantages: jnp.ndarray):
    def loss_fn(p):
        logits_full = model.apply({"params": p},
                                  tokens,
                                  deterministic=True)

        logits = jnp.take_along_axis(logits_full,
                                     idx[:, None, None],
                                     axis=1).squeeze(1)

        log_probs  = jax.nn.log_softmax(logits)
        act_logp   = jnp.take_along_axis(log_probs,
                                         actions[:, None], 1).squeeze(1)
        policy_L   = -(advantages * act_logp).mean()

        entropy    = -(log_probs * jnp.exp(log_probs)).sum(axis=1).mean()
        return policy_L - ENTROPY_COEF * entropy

    return jax.value_and_grad(loss_fn)(params)

for step in range(1, NUM_UPDATES + 1):
    exprs, truths   = sample_batch(BATCH_SIZE)
    tok_batch, lens = encode_batch(tokenizer, exprs, CTX_LEN)
    tokens          = jnp.array(tok_batch, dtype=jnp.int32)
    idx             = lens - 1

    logits_full = model.apply({"params": state.params},
                              tokens,
                              deterministic=True)
    logits = jnp.take_along_axis(logits_full,
                                 idx[:, None, None],
                                 axis=1).squeeze(1)

    rng, sub  = jax.random.split(rng)
    actions   = jax.random.categorical(sub, logits)

    action_int = jnp.take(id2num, actions)
    error     = jnp.abs(action_int - jnp.array(truths))
    rewards   = 1.0 - (error / 121.0)
    baseline   = BASE_MOMENT * baseline + (1 - BASE_MOMENT) * rewards.mean()
    advantages = rewards - baseline

    loss, grads = compute_loss_and_grads(state.params,
                                         tokens,
                                         idx,
                                         actions,
                                         advantages)
    state       = state.apply_gradients(grads=grads)

    if step % LOG_EVERY == 0:
        took = time.time() - wall0
        print(f"step {step:>6d}  loss {loss:+.6f}  "
              f"avgR {rewards.mean():.3f}  baseline {float(baseline):.3f}  "
              f"{took/LOG_EVERY:.3f}s/it")
        wall0 = time.time()

    if step == 1:
        print("Example train expression:", exprs[0], " → ", truths[0])


    if step % CHECK_EVERY == 0:
        ckpt_path = SAVE_DIR / f"ckpt_{step:06d}.npz"
        print(f"Saving → {ckpt_path}")
        save_npz(state.params, ckpt_path)



print("Done!")

