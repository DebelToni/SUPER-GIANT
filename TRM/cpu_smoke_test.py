from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRM import TRM
from sudoku.Training_step import train_step, eval_step


def main() -> None:
    print("backend:", jax.default_backend())

    model = TRM(
        vocab_size=8,
        context_length=4,
        d_model=16,
        tiny_layers=1,
        variant="attn",
        num_heads=4,
        rope_dim=4,
        d_ff=32,
        mixer_hidden=32,
        dropout_rate=0.0,
        activation="silu",
        add_positional_embedding=False,
        L_cycles=2,
        H_cycles=2,
        max_supervision_steps=4,
        enable_early_stop=True,
        halt_threshold_logit=0.0,
        halt_exploration_prob=0.0,
        no_act_continue=True,
        aug_enabled=False,
        aug_num_embeddings=1,
        aug_default_id=0,
    )

    rng = jax.random.PRNGKey(0)
    init_tokens = jnp.zeros((1, model.context_length), dtype=jnp.int32)
    init_aug = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, init_tokens, deterministic=False, aug_ids=init_aug)
    params = variables["params"]

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(params)

    puzzle = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    solution = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    aug_ids = jnp.zeros((1,), dtype=jnp.int32)
    batch = {"puzzle": puzzle, "solution": solution, "aug_ids": aug_ids}

    for step in range(3):
        params, opt_state, (loss, ce, halt, solved, token) = train_step(
            params,
            opt_state,
            batch,
            model=model,
            optimizer=optimizer,
            dropout_rng=jax.random.fold_in(rng, step),
            supervision_steps=3,
            microbatch_size=None,
        )
        jax.block_until_ready(loss)
        print(
            f"step={step} loss={float(loss):.4f} ce={float(ce):.4f} "
            f"halt={float(halt):.4f} solved={float(solved):.3f} token={float(token):.3f}"
        )

    solved_acc, token_acc, ce = eval_step(params, batch, model=model, supervision_steps=3)
    jax.block_until_ready((solved_acc, token_acc, ce))
    print(
        f"eval solved={float(solved_acc):.3f} token={float(token_acc):.3f} ce={float(ce):.4f}"
    )


if __name__ == "__main__":
    main()
