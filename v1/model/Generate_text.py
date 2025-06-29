# generate_text.py
"""
My old inference script that does not use JIT or KV caching. Do not refer to this.
"""

import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# import Config  
from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

from GiantGPT import GiantGPT  

def load_params(path: Path):
    """Load Flax params dict pickled by `save_params`."""
    print(f"Attempting to load parameters from: {path.resolve()}")
    if not path.exists():
        print(f"Error: Parameter file not found at {path.resolve()}")
        exit(1)
    with path.open("rb") as f:
        return pickle.load(f)


def build_model() -> GiantGPT:
    """Re‑instantiate GiantGPT with tokenizer‑aware vocab size."""

    if Config.use_custom_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    return GiantGPT(
        # vocab_size=Config.vocab_size+1,
        vocab_size=tok.vocab_size,
        context_length=Config.context_length,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.feed_forward_size,
        n_layers=Config.num_layers,
        dropout_rate=0.0,
    )


def _select_logits_sampling(logits: jnp.ndarray, rng: jax.random.PRNGKey,
                            temperature: float = 1.0, top_k: Optional[int] = None):
    """Sample one token from the distribution at `logits`."""
    logits = logits.astype(jnp.float32)
    
    logits = logits / jnp.maximum(1e-8, temperature)

    if top_k is not None:
        top_vals = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < top_vals, -jnp.inf, logits)

    probs = jax.nn.softmax(logits)
    return jax.random.choice(rng, logits.shape[-1], p=probs)


@jax.jit
def _apply_model(params, x, model):
    """JIT-compiled function for model application."""
    return model.apply({"params": params}, x, deterministic=True)


def generate(model, params, tokenizer, prompt: str, steps: int = 1,
             temperature: float = 1.0, top_k: Optional[int] = None,
             greedy: bool = False) -> str:
    """Return `prompt` plus `steps` newly generated tokens."""
    rng = jax.random.PRNGKey(0)

    print(f"Input prompt: '{prompt}'")
    ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Input IDs: {ids}")

    apply_fn = jax.jit(lambda p, i: model.apply({"params": p}, i, deterministic=True))

    for i in range(steps):
        print(f"\n--- Step {i+1}/{steps} ---")
        
        effective_context_length = Config.context_length - 1
        ctx = ids[-effective_context_length:]
        print(f"Context (last {len(ctx)} tokens): {ctx}")

        x = jnp.asarray(ctx, dtype=jnp.int32)[None, :]
        print(f"Model input shape: {x.shape}")

        logits = apply_fn(params, x) 
        print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")

        last_logits = logits[:, -1, :]  # (1, vocab)
        last_logits = jnp.squeeze(last_logits, axis=0)  # (vocab,)
        print(f"Last logits shape: {last_logits.shape}")

        if greedy:
            next_id = int(jnp.argmax(last_logits.astype(jnp.float32))) 
        else:
            rng, sub = jax.random.split(rng)
            next_id = int(_select_logits_sampling(last_logits, sub,
                                                temperature, top_k))
        
        print(f"Selected token ID: {next_id} ('{tokenizer.decode([next_id])}')")
        ids.append(next_id)

        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            print("EOS token reached. Stopping generation.")
            break

    print("\n--- Generation Complete ---")
    return tokenizer.decode(ids, clean_up_tokenization_spaces=True)



def main():
    cli = argparse.ArgumentParser(description="Generate text using a trained GiantGPT model.")
    cli.add_argument("--prompt", type=str, default="The quick brown fox jumped over the lazy",
                     help="The initial text to start generation from.")
    cli.add_argument("--params", type=Path, default=Path("model_params.pkl"),
                     help="Path to the saved model parameters file (e.g., model_params.pkl).")
    cli.add_argument("--steps", type=int, default=50,
                     help="Number of new tokens to generate.")
    cli.add_argument("--temperature", type=float, default=0.8,
                     help="Sampling temperature. Higher values increase randomness.")
    cli.add_argument("--top_k", type=int, default=40,
                     help="Top-k sampling; considers only the k most likely tokens. 0 or None for no top-k.")
    cli.add_argument("--greedy", action="store_true",
                     help="If set, force argmax sampling (ignore temperature/top_k).")
    args = cli.parse_args()

    print("--- Setup ---")
    print(f"Using JAX device: {jax.default_backend()} ({jax.devices()})")
    tok_src = Config.custom_tokenizer_path if Config.use_custom_tokenizer else Config.tokenizer_name
    print(f"Loading tokenizer ({tok_src}) …", flush=True)
    if Config.use_custom_tokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model …", flush=True)
    model = build_model()

    print(f"Loading params from {args.params} …", flush=True)
    params = load_params(args.params)

    print("--- Starting Generation ---")
    text = generate(model, params, tokenizer,
                    prompt=args.prompt,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_k=(None if args.top_k in (0, None) else args.top_k),
                    greedy=args.greedy)
    
    print("\n" + "="*20 + " RESULT " + "="*20)
    print(text)
    print("="*48)


if __name__ == "__main__":
    main()
