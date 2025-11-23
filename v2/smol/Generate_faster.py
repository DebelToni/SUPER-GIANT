from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from omegaconf import OmegaConf

SMOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SMOL_DIR.parent

CONFIG_PATH = SMOL_DIR / "Config.yml"
Config = OmegaConf.load(CONFIG_PATH)

import sys
sys.path.append(str(PROJECT_ROOT / "model"))
sys.path.append(str(SMOL_DIR))
from checkpoint_io import load_npz  # noqa: E402
from GiantGPT import GiantGPT  # noqa: E402,F401


def build_model() -> GiantGPT:
    tok = AutoTokenizer.from_pretrained(Config.tokenizer.name, use_fast=True, cache_dir=Config.tokenizer.cache_dir)
    return GiantGPT(
        vocab_size=len(tok),
        context_length=Config.model.context_length,
        d_model=Config.model.embedding_size,
        n_heads=Config.model.num_heads,
        d_ff=Config.model.feed_forward_size,
        n_layers=Config.model.num_layers,
        dropout_rate=0.0,
    )


def _numpy_or_jax_array(x):
    return jnp.asarray(x) if not isinstance(x, jax.Array) else x


def load_checkpoint(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(_numpy_or_jax_array, params)


def init_caches(model: GiantGPT, params: dict, batch_size: int = 1):
    dummy_token = jnp.ones((batch_size, 1), jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(0),
        dummy_token,
        deterministic=True,
        use_kv_cache=True,
        cur_index=jnp.array(0, jnp.int32),
    )
    return variables["cache"]


def preprocess_prompt(tokenizer, prompt: str, max_len: int):
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if ids.shape[0] >= max_len:
        ids = ids[-max_len:]
    return ids.astype("int32")


def make_step_fn(model: GiantGPT, temperature: float, top_k: Optional[int]):
    @jax.jit
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
            use_kv_cache=True,
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
            if top_k and top_k > 0:
                values, _ = jax.lax.top_k(logits, top_k)
                kth = values[:, -1][:, None]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            next_token = jax.random.categorical(rng, logits, axis=-1)
        next_token = next_token.astype(jnp.int32)[:, None]
        return next_token, cache

    return step_fn


def _select_device():
    target = getattr(Config.model, "device", "auto")
    if target == "cpu":
        return jax.devices("cpu")[0]
    if target == "gpu":
        try:
            gpus = jax.devices("gpu")
            if gpus:
                return gpus[0]
        except RuntimeError:
            pass
        print("⚠ Requested GPU but none available; falling back to CPU.")
        return jax.devices("cpu")[0]
    try:
        gpus = jax.devices("gpu")
        if gpus:
            return gpus[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0]


def generate(
    params: dict,
    model: GiantGPT,
    tokenizer,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    *,
    return_stats: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    device = _select_device()
    params = jax.device_put(params, device)

    cache = init_caches(model, params)

    tokens = prompt_ids[None, :]
    rng = jax.random.PRNGKey(42)

    step_fn = make_step_fn(model, temperature, top_k)

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
        cache = jax.tree_util.tree_map(lambda x: x.block_until_ready(), cache)
        prefill_time_s = time.perf_counter() - t0

    pad_len = max_new_tokens
    tokens = jnp.pad(tokens, ((0, 0), (0, pad_len)))

    def generation_body(state, _):
        tokens_buf, cache, rng, idx = state
        rng, step_rng = jax.random.split(rng)
        prev_token = jax.lax.dynamic_slice_in_dim(tokens_buf, idx - 1, 1, axis=1)
        next_token, cache = step_fn(params, cache, prev_token, idx - 1, step_rng)
        tokens_buf = jax.lax.dynamic_update_slice(tokens_buf, next_token, (0, idx))
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
    tokens.block_until_ready()
    decode_time_s = time.perf_counter() - t1

    full_tokens = np.asarray(tokens[0, :tokens.shape[1] - pad_len + max_new_tokens])
    decoded = tokenizer.decode(full_tokens, skip_special_tokens=True)

    stop_on_eos = bool(getattr(Config.inference, "stop_on_eos", False))
    eos_id = tokenizer.eos_token_id
    if stop_on_eos and eos_id is not None:
        idx = np.where(full_tokens == eos_id)[0]
        if idx.size > 0:
            cut = int(idx[0])
            decoded = tokenizer.decode(full_tokens[:cut], skip_special_tokens=True) + "<EOS>"

    if return_stats:
        return decoded, {
            "prefill_tokens": int(prompt_ids.shape[0]),
            "generated_tokens": int(max_new_tokens),
            "prefill_time_s": float(prefill_time_s),
            "decode_time_s": float(decode_time_s),
        }
    return decoded


def resolve_checkpoint(path_arg: str | None) -> Path:
    if path_arg and path_arg != "latest":
        p = Path(path_arg)
        return p
    ckpt_dir = Path(Config.paths.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (PROJECT_ROOT / ckpt_dir).resolve()
    candidates = sorted(ckpt_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint path (.npz). Defaults to latest under Config.paths.checkpoint_dir.")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    checkpoint_path = resolve_checkpoint(str(args.checkpoint) if args.checkpoint else "latest")

    temperature = 0.0 if args.greedy else (args.temperature if args.temperature is not None else Config.generate.temperature)
    top_k = args.top_k if args.top_k is not None else Config.generate.top_k
    steps = args.steps if args.steps is not None else Config.generate.max_new_tokens

    params_cpu = load_checkpoint(checkpoint_path)

    model = build_model()
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer.name, use_fast=True, cache_dir=Config.tokenizer.cache_dir)

    prompt_ids = preprocess_prompt(tokenizer, args.prompt, Config.model.context_length)

    if args.verbose:
        text, stats = generate(
            params_cpu,
            model,
            tokenizer,
            prompt_ids,
            steps,
            temperature,
            top_k,
            return_stats=True,
        )
    else:
        text = generate(
            params_cpu,
            model,
            tokenizer,
            prompt_ids,
            steps,
            temperature,
            top_k,
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
