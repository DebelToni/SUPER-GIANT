# Generate_text_fast.py – KV-cached, fully‑JIT text generation for GiantGPT
# ------------------------------------------------------------------------
"""Drop‑in replacement for the old `Generate_text.py` that
  • JIT‑compiles the whole autoregressive loop (no Python per‑token step)
  • Stores/updates KV caches in Flax’s \"cache\" collection on device
  • Supports greedy, temperature and top‑k sampling

▪ **2025‑05‑31 update** – more robust checkpoint loader
  The script now accepts Pickle/NPZ/Numpy checkpoints without assuming
  `numpy.load(...).item()`.  This fixes the `AttributeError: 'dict' object …`
  you hit when the checkpoint is already a Python dict.  See `load_checkpoint()`
  below.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

import Config
from GiantGPT import GiantGPT

# ---------------------------------------------------------------------------
# Helper: build model exactly as in training (but dropout off by default)
# ---------------------------------------------------------------------------

def build_model() -> GiantGPT:
    return GiantGPT(
        vocab_size=Config.vocab_size,
        context_length=Config.context_length,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.ffn_size,
        n_layers=Config.num_layers,
        dropout_rate=0.0,  # dropout disabled for inference
    )

# ---------------------------------------------------------------------------
# Robust checkpoint loader (handles .pkl, .npz, .npy)
# ---------------------------------------------------------------------------

def _numpy_or_jax_array(x):
    """Ensure leaves are JAX arrays – helpful if checkpoint stored NumPy."""
    return jnp.asarray(x) if not isinstance(x, jax.Array) else x


def load_checkpoint(path: Path):
    """Return a PyTree of JAX arrays living on *CPU* (device_put later)."""
    ext = path.suffix.lower()
    if ext in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            params = pickle.load(f)
    elif ext == ".npz":
        params = dict(np.load(path, allow_pickle=True))
    else:  # .npy OR anything saved with np.save / jnp.save
        arr = np.load(path, allow_pickle=True)
        params = arr.item() if hasattr(arr, "item") else arr
    return jax.tree_util.tree_map(_numpy_or_jax_array, params)


# ---------------------------------------------------------------------------
# JIT‑compiled generation with KV caching
# ---------------------------------------------------------------------------

def init_caches(model: GiantGPT, params: dict, batch_size: int = 1):
    """Initialise empty `cache` collection with correct shapes on device."""
    dummy_token = jnp.ones((batch_size, 1), jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(0),
        dummy_token,
        deterministic=True,
        decode=True,
        cur_index=jnp.array(0, jnp.int32),
    )
    return variables.pop("params")  # -> cache dict


def preprocess_prompt(tokenizer, prompt: str, max_len: int):
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if ids.shape[0] >= max_len:
        ids = ids[-max_len:]
    return ids.astype("int32")


# ––––– Single‑step helper (JIT) ––––––––––––––––––––––––––––––––––––––––––––

def make_step_fn(model: GiantGPT, temperature: float, top_k: Optional[int]):
    """Returns a *pure* JIT‑able step function closed over params/constants."""

    @jax.jit(donate_argnums=(1,))  # donate cache to avoid copies
    def step_fn(
        params: dict,
        cache: dict,
        prev_token: jnp.ndarray,  # (B, 1)
        cur_index: jnp.ndarray,   # () scalar int32
        rng: jax.random.KeyArray,
    ):
        # Apply model, updating KV cache inside the mutable collection.
        logits, new_vars = model.apply(
            {"params": params, "cache": cache},
            prev_token,
            deterministic=True,
            decode=True,
            cur_index=cur_index,
            rngs={"dropout": rng},  # still supply but deterministic=True
            mutable=["cache"],
        )
        cache = new_vars["cache"]
        logits = logits[:, 0]  # (B, vocab)

        # Greedy or sampling --------------------------------------------------
        if temperature == 0.0:
            next_token = jnp.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                kth = jnp.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            next_token = jax.random.categorical(rng, logits, axis=-1)
        next_token = next_token.astype(jnp.int32)[:, None]  # (B,1)
        return next_token, cache

    return step_fn

# ––––– Main generation ––––––––––––––––––––––––––––––––––––––––––––––––––––

def generate(
    params: dict,
    model: GiantGPT,
    tokenizer,
    prompt_ids: jnp.ndarray,  # (P,)
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
):
    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    # Build empty caches & prime them with the prompt -----------------------
    cache = init_caches(model, params)

    # Broadcast prompt into batch dim 1
    tokens = prompt_ids[None, :]  # (1, P)
    rng = jax.random.PRNGKey(42)

    step_fn = make_step_fn(model, temperature, top_k)

    # Feed the prompt tokens except the last one to build up the cache ------
    if tokens.shape[1] > 1:
        def warm_body(state, token_and_idx):
            cache, _ = state
            tok, idx = token_and_idx  # scalar token, scalar idx
            _, cache = step_fn(params, cache, tok[None, None], idx, rng)
            return (cache, None), None

        idxs = jnp.arange(tokens.shape[1]-1, dtype=jnp.int32)
        toks = tokens[:, :-1].squeeze(0)
        (cache, _), _ = jax.lax.scan(
            warm_body,
            (cache, None),
            (toks, idxs),
        )

    # Append room for new tokens --------------------------------------------
    pad_len = max_new_tokens
    max_len = tokens.shape[1] + pad_len
    tokens = jnp.pad(tokens, ((0, 0), (0, pad_len)))  # shape (1, max_len)

    def generation_body(state, _):
        tokens_buf, cache, rng, idx = state
        rng, step_rng = jax.random.split(rng)
        prev_token = tokens_buf[:, idx-1:idx]  # last generated/input token
        next_token, cache = step_fn(params, cache, prev_token, idx-1, step_rng)
        tokens_buf = tokens_buf.at[:, idx].set(next_token.squeeze(1))
        return (tokens_buf, cache, rng, idx+1), None

    init_state = (tokens, cache, rng, tokens.shape[1] - pad_len + 1)
    (tokens, _, _, _), _ = jax.lax.scan(
        generation_body,
        init_state,
        None,
        length=max_new_tokens,
    )
    return tokenizer.decode(tokens[0, :tokens.shape[1]-pad_len + max_new_tokens], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Entry‑point CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    temperature = 0.0 if args.greedy else args.temperature

    print("\nLoading checkpoint…")
    params_cpu = load_checkpoint(args.checkpoint)

    print("Building model…")
    model = build_model()
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    prompt_ids = preprocess_prompt(tokenizer, args.prompt, Config.context_length)

    print("Generating… (first call will JIT‑compile)")
    text = generate(
        params_cpu,
        model,
        tokenizer,
        prompt_ids,
        args.steps,
        temperature,
        args.top_k,
    )

    print("\n" + "="*20 + " RESULT " + "="*20)
    print(text)
    print("="*48)


if __name__ == "__main__":
    main()

