
"""
Generate_text_fast.py – greedy decoding script for GiantGPT.

Supports KV‑cache and the new `block_*` param structure.
"""
import argparse
import pickle
from typing import List

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from omegaconf import OmegaConf
from GiantGPT import GiantGPT, build_apply_fn

Config = OmegaConf.load("Config.yml")

def greedy_generate(model, apply_fn, params, prompt: str, max_steps: int = 50):
    tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    input_ids = jnp.array(tok(prompt)["input_ids"], dtype=jnp.int32)[None, :]  # [1, L]

    # Allocate KV‑cache (Python ''dict'' is fine; Flax expects an identical tree)
    cache = None
    rng = jax.random.PRNGKey(0)

    cur_index = 0
    for _ in range(max_steps):
        logits, cache = apply_fn(
            params,
            cache,
            input_ids,
            rng=rng,
            deterministic=True,
            enable_kv_cache=True,
            cur_index=cur_index,
        )
        next_id = jnp.argmax(logits[0, -1]).astype(jnp.int32)
        input_ids = jnp.concatenate([input_ids, next_id[None, None]], axis=1)
        cur_index += 1

    decoded = tok.batch_decode(input_ids[0].tolist(), skip_special_tokens=True)[0]
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="model_params.pkl")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    model = GiantGPT(
        vocab_size       = tok.vocab_size,
        context_length   = Config.context_length,
        d_model          = Config.embedding_size,
        n_heads          = Config.num_heads,
        d_ff             = Config.feed_forward_size,
        n_layers         = Config.num_layers,
        dropout_rate     = Config.dropout_rate,
    )
    apply_fn = build_apply_fn(model)

    out = greedy_generate(model, apply_fn, params, args.prompt, args.steps)
    print(out)

if __name__ == "__main__":
    main()
