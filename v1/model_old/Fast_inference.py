"""
Standalone, **jit‑compiled** inference script with no Python in the token loop.

Usage
-----
$ python fast_generate.py \
        --params  model_params.pkl \
        --prompt  "The quick brown fox" \
        --steps   64 \
        --temperature 0.8 --top_k 40

The script auto‑detects CPU/GPU and honours the dtypes in `Config.py`.
"""

import argparse, pickle, functools
from pathlib import Path
from typing import Optional, Tuple, Any

import jax, jax.numpy as jnp
from transformers import AutoTokenizer

import Config
from GiantGPT import GiantGPT

# ------------- I/O helpers ----------------------------------------------------
def load_params(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def initialise_model(tokenizer) -> Tuple[Any, Any]:
    """Initialise model variables and an *empty* KV cache."""
    model = GiantGPT(
        vocab_size     = Config.vocab_size,
        context_length = Config.context_length,
        d_model        = Config.embedding_size,
        n_heads        = Config.num_heads,
        d_ff           = Config.feed_forward_size,
        n_layers       = Config.num_layers,
        dropout_rate   = 0.0,
        use_cache      = True,          # <‑‑ enable caching
    )
    dummy = jnp.zeros((1, 1), jnp.int32)
    variables = model.init(jax.random.PRNGKey(0),
                           dummy,
                           deterministic=True,
                           init_cache=True)
    params, cache = variables["params"], variables["cache"]
    return model, params, cache

# ------------- sampling -------------------------------------------------------
def sample_next_id(rng, logits, temperature: float, top_k: Optional[int], greedy: bool):
    logits = logits.astype(jnp.float32) / jnp.maximum(1e-8, temperature)
    if greedy:
        return jnp.argmax(logits)
    if top_k and top_k > 0:
        kth = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    return jax.random.categorical(rng, logits)

# ------------- compiled generation loop --------------------------------------
def compile_generate(model, steps, temperature, top_k, greedy):
    @functools.partial(jax.jit, static_argnums=(5,6,7))
    def _generate(params, cache, tokens, rng, steps, temperature, top_k, greedy):
        """KV‑cached autoregressive generation via `lax.scan`."""

        def one_step(state, _):
            cache, last_id, rng = state
            logits, new_cache = model.apply(
                {"params": params, "cache": cache},
                jnp.array([[last_id]], jnp.int32),
                deterministic=True,
                mutable=["cache"])
            logits = logits[0, 0]                  # (vocab,)
            rng, sub = jax.random.split(rng)
            next_id  = sample_next_id(sub, logits, temperature, top_k, greedy)
            return (new_cache["cache"], next_id, rng), next_id

        rng = jax.random.PRNGKey(42)
        init_state = (cache, tokens[-1], rng)
        (_, _, _), generated = jax.lax.scan(one_step, init_state, None, length=steps)
        return generated
    return _generate

# ------------- main -----------------------------------------------------------
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--prompt", type=str, default="The quick brown fox jumped over the lazy")
    cli.add_argument("--params", type=Path, default=Path("model_params.pkl"))
    cli.add_argument("--steps", type=int, default=50)
    cli.add_argument("--temperature", type=float, default=0.8)
    cli.add_argument("--top_k", type=int, default=40)
    cli.add_argument("--greedy", action="store_true")
    args = cli.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, params, cache = initialise_model(tokenizer)

    # overwrite fresh params with the trained ones
    params = load_params(args.params)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    # prime model with the prompt *once* (updates KV cache)
    _, cache = model.apply({"params": params, "cache": cache},
                           jnp.array([prompt_ids], jnp.int32),
                           deterministic=True,
                           mutable=["cache"])

    generate_fn = compile_generate(model,
                                   steps=args.steps,
                                   temperature=args.temperature,
                                   top_k=(None if args.top_k in (0, None) else args.top_k),
                                   greedy=args.greedy)

    out_ids = generate_fn(params, cache["cache"], jnp.array(prompt_ids, jnp.int32),
                          None,                          # rng seeded inside
                          args.steps,
                          args.temperature,
                          args.top_k,
                          args.greedy)

    full = prompt_ids + list(map(int, out_ids))
    text = tokenizer.decode(full, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)
    print(text)

if __name__ == "__main__":
    main()
