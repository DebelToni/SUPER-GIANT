from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from checkpoint_io import load_npz

import jax.lax as lax
from functools import partial

from omegaconf import OmegaConf
CONFIG_PATH = Path(__file__).resolve().parent / "Config.yml"
Config = OmegaConf.load(CONFIG_PATH)

from GiantGPT import GiantGPT

from jax import config
config.update("jax_default_matmul_precision", "tensorfloat32")


def build_model() -> GiantGPT:
    if Config.use_custom_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(Config.tokenizer_name, use_fast=True)

    return GiantGPT(
        vocab_size=len(tok),
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
        use_kv_cache=True,
        cur_index=jnp.array(0, jnp.int32),
    )
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
    """Returns a *pure* JIT-able step function closed over params/constants."""

    @partial(jax.jit, donate_argnums=(1,))
    def step_fn(
        params: dict,
        cache: dict,
        prev_token: jnp.ndarray,   # (B,1)
        cur_index: jnp.ndarray,    # scalar int32
        rng: jax.random.KeyArray,
    ):
        logits, new_vars = model.apply(
            {"params": params, "cache": cache},
            prev_token,
            deterministic=True,
            use_kv_cache=True,
            cur_index=cur_index,
            rngs={"dropout": rng},
            mutable=["cache"],
        )
        cache = new_vars["cache"]
        logits = logits[:, 0]  # (B,V)

        if temperature == 0.0:
            next_token = jnp.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                values, _ = jax.lax.top_k(logits, top_k)
                kth = values[:, -1][:, None]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            next_token = jax.random.categorical(rng, logits, axis=-1)
        next_token = next_token.astype(jnp.int32)[:, None]  # (B,1)
        return next_token, cache

    return step_fn


def _block_until_ready_tree(x):
    """Blocks on all JAX arrays inside a PyTree."""
    leaves = jax.tree_util.tree_leaves(x)
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def generate(
    params: dict,
    model: GiantGPT,
    tokenizer,
    prompt_ids: jnp.ndarray,  # (L,)
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    *,
    return_stats: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    """
    Generates text. If return_stats is True, also returns a dict with:
      - prefill_tokens
      - generated_tokens
      - prefill_time_s
      - decode_time_s
    """
    device = jax.devices(Config.device)[0]
    params = jax.device_put(params, device)

    cache = init_caches(model, params)

    tokens = prompt_ids[None, :]  # (1, L_prompt)
    rng = jax.random.PRNGKey(42)

    step_fn = make_step_fn(model, temperature, top_k)

    # --- Prefill over the prompt (does not count toward tokens/sec) ---
    prefill_time_s = 0.0
    if tokens.shape[1] > 1:
        def warm_body(state, token_and_idx):
            cache, _ = state
            tok, idx = token_and_idx
            _, cache = step_fn(params, cache, tok[None, None], idx, rng)
            return (cache, None), None

        idxs = jnp.arange(tokens.shape[1] - 1, dtype=jnp.int32)
        toks = tokens[:, :-1].squeeze(0)

        t0 = time.perf_counter()
        (cache, _), _ = jax.lax.scan(
            warm_body,
            (cache, None),
            (toks, idxs),
        )
        _block_until_ready_tree(cache)
        prefill_time_s = time.perf_counter() - t0

    # --- Decode loop over max_new_tokens (we report tokens/sec for this only) ---
    pad_len = max_new_tokens
    tokens = jnp.pad(tokens, ((0, 0), (0, pad_len)))  # (1, L_prompt + pad)

    def generation_body(state, _):
        tokens_buf, cache, rng, idx = state
        rng, step_rng = jax.random.split(rng)
        prev_token = lax.dynamic_slice_in_dim(tokens_buf, idx - 1, 1, axis=1)
        next_token, cache = step_fn(params, cache, prev_token, idx - 1, step_rng)
        tokens_buf = lax.dynamic_update_slice(tokens_buf, next_token, (0, idx))
        return (tokens_buf, cache, rng, idx + 1), None

    start_idx = jnp.array(tokens.shape[1] - pad_len, dtype=jnp.int32)
    init_state = (tokens, cache, rng, start_idx)

    t1 = time.perf_counter()
    (tokens, _, _, _), _ = jax.lax.scan(
        generation_body,
        init_state,
        None,
        length=max_new_tokens,
    )
    # Force device sync so timing reflects actual compute time
    tokens.block_until_ready()
    decode_time_s = time.perf_counter() - t1

    out = tokenizer.decode(
        tokens[0, :tokens.shape[1] - pad_len + max_new_tokens],
        skip_special_tokens=True,
    )

    if return_stats:
        return out, {
            "prefill_tokens": int(prompt_ids.shape[0]),
            "generated_tokens": int(max_new_tokens),
            "prefill_time_s": float(prefill_time_s),
            "decode_time_s": float(decode_time_s),
        }
    return out


generate_jit = jax.jit(generate, static_argnames=("model", "tokenizer", "temperature", "top_k"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default="model_params.pkl")
    ap.add_argument("--prompt", type=str, default="Once upon")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="Print timing and tokens/sec for the decode phase only.")
    args = ap.parse_args()

    temperature = 0.0 if args.greedy else args.temperature

    print("\nLoading checkpoint…")
    params_cpu = load_checkpoint(args.checkpoint)

    print("Building model…")
    model = build_model()
    if Config.use_custom_tokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name, use_fast=True)

    prompt_ids = preprocess_prompt(tokenizer, args.prompt, Config.context_length)

    print("Generating… (first decode call may include JIT compile)")
    if args.verbose:
        text, stats = generate(
            params_cpu,
            model,
            tokenizer,
            prompt_ids,
            args.steps,
            temperature,
            args.top_k,
            return_stats=True,
        )
    else:
        text = generate(
            params_cpu,
            model,
            tokenizer,
            prompt_ids,
            args.steps,
            temperature,
            args.top_k,
            return_stats=False,
        )

    print("\n" + "="*20 + " RESULT " + "="*20)
    print(text)
    print("="*48)

    if args.verbose:
        gen_tok = stats["generated_tokens"]
        dec_s = stats["decode_time_s"]
        toks_per_s = (gen_tok / dec_s) if dec_s > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {stats['prefill_tokens']}")
        print(f"generated_tokens: {gen_tok}")
        print(f"prefill_time_s: {stats['prefill_time_s']:.6f}")
        print(f"decode_time_s:  {dec_s:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
