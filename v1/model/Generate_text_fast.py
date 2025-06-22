# generate_text_fast.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from checkpoint_io import load_npz

import jax.lax as lax

from functools import partial

# import Config
from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

from GiantGPT import GiantGPT

def build_model() -> GiantGPT:
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
        params = load_npz(path)
    else:  
        arr = np.load(path, allow_pickle=True)
        params = arr.item() if hasattr(arr, "item") else arr
    return jax.tree_util.tree_map(_numpy_or_jax_array, params)

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
    # return variables.pop("params")  
    return variables["cache"]

def preprocess_prompt_no_EOS(tokenizer, prompt: str, max_len: int):
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]

    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.array(ids, dtype="int32")

def preprocess_prompt(tokenizer, prompt: str, max_len: int):
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if ids.shape[0] >= max_len:
        ids = ids[-max_len:]
    return ids.astype("int32")

def make_step_fn(model: GiantGPT, temperature: float, top_k: Optional[int]):
    """Returns a *pure* JIT‑able step function closed over params/constants."""

    # @jax.jit(donate_argnums=(1,))  
    @partial(jax.jit, donate_argnums=(1,))
    def step_fn(
        params: dict,
        cache: dict,
        prev_token: jnp.ndarray,  
        cur_index: jnp.ndarray,   
        rng: jax.random.KeyArray,
    ):

        logits, new_vars = model.apply(
            {"params": params, "cache": cache},
            prev_token,
            deterministic=True,
            decode=True,
            cur_index=cur_index,
            rngs={"dropout": rng},  
            mutable=["cache"],
        )
        cache = new_vars["cache"]
        logits = logits[:, 0]  

        if temperature == 0.0:
            next_token = jnp.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            # if top_k is not None and top_k > 0:
            #     kth = jnp.sort(logits, axis=-1)[:, -top_k][:, None]
            #     logits = jnp.where(logits < kth, -jnp.inf, logits)
            if top_k and top_k > 0:
                values, _ = jax.lax.top_k(logits, top_k)
                kth = values[:, -1][:, None]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            next_token = jax.random.categorical(rng, logits, axis=-1)
        next_token = next_token.astype(jnp.int32)[:, None]  
        return next_token, cache

    return step_fn

def generate(
    params: dict,
    model: GiantGPT,
    tokenizer,
    prompt_ids: jnp.ndarray,  
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
):
    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    cache = init_caches(model, params)

    tokens = prompt_ids[None, :]  
    rng = jax.random.PRNGKey(42)

    step_fn = make_step_fn(model, temperature, top_k)

    if tokens.shape[1] > 1:
        def warm_body(state, token_and_idx):
            cache, _ = state
            tok, idx = token_and_idx  
            _, cache = step_fn(params, cache, tok[None, None], idx, rng)
            return (cache, None), None

        idxs = jnp.arange(tokens.shape[1]-1, dtype=jnp.int32)
        toks = tokens[:, :-1].squeeze(0)
        (cache, _), _ = jax.lax.scan(
            warm_body,
            (cache, None),
            (toks, idxs),
        )

    pad_len = max_new_tokens
    max_len = tokens.shape[1] + pad_len
    tokens = jnp.pad(tokens, ((0, 0), (0, pad_len)))  

    def generation_body(state, _):
        tokens_buf, cache, rng, idx = state
        rng, step_rng = jax.random.split(rng)

        # prev_token = tokens_buf[:, idx-1:idx]  
        prev_token = lax.dynamic_slice_in_dim(tokens_buf, idx - 1, 1, axis=1)  # (1,1)

        # next_token, cache = step_fn(params, cache, prev_token, idx-1, step_rng)
        next_token, cache = step_fn(params, cache, prev_token, idx - 1, step_rng)  # (1,1)

        # tokens_buf = tokens_buf.at[:, idx].set(next_token.squeeze(1))
        tokens_buf = lax.dynamic_update_slice(tokens_buf, next_token, (0, idx))

        return (tokens_buf, cache, rng, idx + 1), None

    start_idx = jnp.array(tokens.shape[1] - pad_len, dtype=jnp.int32)

    # init_state = (tokens, cache, rng, tokens.shape[1] - pad_len + 1)
    init_state = (tokens, cache, rng, start_idx)
    (tokens, _, _, _), _ = jax.lax.scan(
        generation_body,
        init_state,
        None,
        length=max_new_tokens,
    )

    return tokenizer.decode(tokens[0, :tokens.shape[1]-pad_len + max_new_tokens], skip_special_tokens=True)

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
    if Config.use_custom_tokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
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
