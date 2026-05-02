from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRM import TRM
from jit_inference import init_params, make_jitted_inference


def load_cfg():
    return OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
        OmegaConf.load(PROJECT_ROOT / "model" / "Config.yml"),
        OmegaConf.load(Path(__file__).resolve().parent / "Config.yml"),
    )


def main():
    cfg = load_cfg()
    mcfg = cfg.model

    model = TRM(
        vocab_size=int(mcfg.vocab_size),
        context_length=int(mcfg.context_length),
        d_model=int(mcfg.embedding_size),
        tiny_layers=int(mcfg.tiny_layers),
        variant=str(mcfg.variant),
        num_heads=int(mcfg.num_heads),
        rope_dim=int(mcfg.rope_dim),
        d_ff=int(mcfg.feed_forward_size),
        mixer_hidden=int(mcfg.mixer_hidden),
        dropout_rate=float(mcfg.dropout_rate),
        activation=str(mcfg.activation),
        add_positional_embedding=bool(getattr(mcfg, "add_positional_embedding", True)),
        L_cycles=int(mcfg.recursion.L_cycles),
        H_cycles=int(mcfg.recursion.H_cycles),
        max_supervision_steps=int(mcfg.recursion.max_supervision_steps),
        enable_early_stop=bool(mcfg.recursion.enable_early_stop),
        halt_threshold_logit=float(mcfg.recursion.halt_threshold_logit),
        aug_enabled=bool(mcfg.augmentation.enabled),
        aug_num_embeddings=int(mcfg.augmentation.num_embeddings),
        aug_default_id=int(mcfg.augmentation.default_id),
    )

    B = 2
    key = jax.random.PRNGKey(0)
    params = init_params(model, key, batch_size=B, pad_token_id=0)

    x_key = jax.random.PRNGKey(1)
    tokens = jax.random.randint(
        x_key,
        (B, int(mcfg.context_length)),
        minval=0,
        maxval=int(mcfg.vocab_size),
        dtype=jnp.int32,
    )
    aug_ids = jnp.zeros((B,), dtype=jnp.int32)

    infer = make_jitted_inference(model)

    t0 = time.time()
    logits, q_logit, pred, _state, steps, halted = infer(params, tokens, aug_ids)
    jax.block_until_ready((logits, q_logit, pred, steps, halted))
    t1 = time.time()

    logits2, q_logit2, pred2, _state2, steps2, halted2 = infer(params, tokens, aug_ids)
    jax.block_until_ready((logits2, q_logit2, pred2, steps2, halted2))
    t2 = time.time()

    print("tokens:", tokens.shape, tokens.dtype)
    print("logits:", logits.shape, logits.dtype)
    print("q_logit:", q_logit.shape, q_logit.dtype)
    print("pred:", pred.shape, pred.dtype)
    print("steps:", steps, steps.dtype)
    print("halted:", halted, halted.dtype)
    print(f"first call (compile+run): {t1 - t0:.3f}s")
    print(f"second call (run):        {t2 - t1:.3f}s")

    # Sanity: outputs stable across runs
    print("pred identical:", bool(jnp.all(pred == pred2)))


if __name__ == "__main__":
    main()
