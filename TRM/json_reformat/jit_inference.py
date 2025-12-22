from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRM import TRM

PyTree = Dict[str, Any]
Array = jnp.ndarray


def init_params(
    model: TRM,
    rng: jax.Array,
    *,
    batch_size: int,
    pad_token_id: int = 0,
) -> PyTree:
    dummy_tokens = jnp.full((batch_size, model.context_length), pad_token_id, dtype=jnp.int32)
    variables = model.init({"params": rng}, dummy_tokens, deterministic=True, aug_ids=None)
    return variables["params"]


def make_jitted_step(model: TRM):
    @jax.jit
    def step(
        params: PyTree,
        tokens: Array,  # (B, L) int32
    ):
        logits, q_logit, pred, (y, z) = model.apply(
            {"params": params},
            tokens,
            deterministic=True,
            aug_ids=None,
        )
        return logits, q_logit, pred, (y, z)

    return step


def make_jitted_inference(model: TRM):
    max_steps = int(model.max_supervision_steps)
    enable_early_stop = bool(model.enable_early_stop)
    threshold = float(model.halt_threshold_logit)

    @jax.jit
    def infer(
        params: PyTree,
        tokens: Array,  # (B, L) int32
    ):
        x = model.apply(
            {"params": params},
            tokens,
            deterministic=True,
            aug_ids=None,
            method=model.encode,
        )
        y, z = model.apply({"params": params}, tokens, method=model.initial_state)

        b = tokens.shape[0]
        halted = jnp.zeros((b,), dtype=jnp.bool_)
        logits = jnp.zeros((b, model.context_length, model.vocab_size), dtype=jnp.float32)
        q_logit = jnp.full((b,), -jnp.inf, dtype=jnp.float32)
        pred = jnp.zeros((b, model.context_length), dtype=jnp.int32)
        step = jnp.array(0, dtype=jnp.int32)

        def cond_fn(carry):
            step, _y, _z, halted, _logits, _q, _pred = carry
            if enable_early_stop:
                return jnp.logical_and(step < max_steps, jnp.logical_not(jnp.all(halted)))
            return step < max_steps

        def body_fn(carry):
            step, y, z, halted, logits, q_logit, pred = carry

            y_n, z_n, logits_n, q_n, pred_n = model.apply(
                {"params": params},
                x,
                y,
                z,
                deterministic=True,
                method=model.step_from_x,
            )

            if enable_early_stop:
                y = jnp.where(halted[:, None, None], y, y_n)
                z = jnp.where(halted[:, None, None], z, z_n)
                logits = jnp.where(halted[:, None, None], logits, logits_n)
                q_logit = jnp.where(halted, q_logit, q_n)
                pred = jnp.where(halted[:, None], pred, pred_n)
                halted = jnp.logical_or(halted, q_logit > threshold)
            else:
                y, z, logits, q_logit, pred = y_n, z_n, logits_n, q_n, pred_n

            return (step + 1, y, z, halted, logits, q_logit, pred)

        step, y, z, halted, logits, q_logit, pred = jax.lax.while_loop(
            cond_fn, body_fn, (step, y, z, halted, logits, q_logit, pred)
        )

        return logits, q_logit, pred, (y, z), step, halted

    return infer
