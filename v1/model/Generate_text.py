# generate_text.py
"""Generate text from a TinyGPT checkpoint saved with `save_params`.

Usage (greedy one‑token demo):
    python generate_text.py --prompt "The quick brown fox jumped over the lazy" --steps 1

Usage (sample 20 tokens with temperature):
    python generate_text.py --prompt "Once upon a time" --steps 20 --temperature 0.8 --top_k 40
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# -------- project‑local imports ------------------------------------------------
import Config                              # hyper‑parameters you trained with
from GiantGPT import GiantGPT            # the wrapper we built earlier


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_params(path: Path):
    """Load Flax params dict pickled by `save_params`."""
    with path.open("rb") as f:
        return pickle.load(f)


def build_model():
    """Re‑instantiate TinyGPT with the same hyper‑params as during training."""
    return GiantGPT(
        vocab_size     = Config.vocab_size,
        context_length = Config.context_length,
        d_model        = Config.embedding_size,
        n_heads        = Config.num_heads,
        d_ff           = Config.feed_forward_size,
        n_layers       = Config.num_layers,
        dropout_rate   = 0.0,            # no dropout at inference
    )


def _select_logits_sampling(logits: jnp.ndarray, rng: jax.random.PRNGKey,
                             temperature: float = 1.0, top_k: Optional[int] = None):
    """Sample one token from the distribution at `logits`."""
    logits = logits / jnp.maximum(1e-8, temperature)

    if top_k is not None:
        # keep only the largest k probs
        top_vals = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < top_vals, -jnp.inf, logits)

    probs = jax.nn.softmax(logits)
    return jax.random.choice(rng, logits.shape[-1], p=probs)


def generate(model, params, tokenizer, prompt: str, steps: int = 1,
             temperature: float = 1.0, top_k: Optional[int] = None,
             greedy: bool = False) -> str:
    """Return `prompt` plus `steps` newly generated tokens."""
    rng = jax.random.PRNGKey(0)

    ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)

    for _ in range(steps):
        # keep only the last `context_length` tokens
        ctx = ids[-Config.context_length:]
        x = jnp.asarray(ctx, dtype=jnp.int32)[None, :]   # (1, seq_len)

        logits = model.apply({"params": params}, x, deterministic=True)
        last_logits = logits[:, -1, :]                   # (1, vocab)
        last_logits = jnp.squeeze(last_logits, axis=0)   # (vocab,)

        if greedy:
            next_id = int(jnp.argmax(last_logits))
        else:
            rng, sub = jax.random.split(rng)
            next_id = int(_select_logits_sampling(last_logits, sub,
                                                  temperature, top_k))

        ids.append(next_id)

        # optional early‑stop on EOS token (if your tokenizer has one)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids, clean_up_tokenization_spaces=True)


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--prompt", type=str, default="The quick brown fox jumped over the lazy")
    cli.add_argument("--params", type=Path, default=Path("model_params.pkl"))
    cli.add_argument("--steps", type=int, default=1, help="how many tokens to generate")
    cli.add_argument("--temperature", type=float, default=1.0)
    cli.add_argument("--top_k", type=int, default=None, help="top‑k sampling; 0 or None for no top‑k")
    cli.add_argument("--greedy", action="store_true", help="force argmax instead of sampling")
    args = cli.parse_args()

    print("Loading tokenizer …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    print("Building model …", flush=True)
    model = build_model()

    print(f"Loading params from {args.params} …", flush=True)
    params = load_params(args.params)

    print("Generating …", flush=True)
    text = generate(model, params, tokenizer,
                    prompt=args.prompt,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_k=(None if args.top_k in (0, None) else args.top_k),
                    greedy=args.greedy)
    print("\n=== RESULT ===\n")
    print(text)


if __name__ == "__main__":
    main()

