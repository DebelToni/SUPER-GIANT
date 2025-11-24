from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from checkpoint_io import load_npz
from jit_inference import init_inference_state, make_prefill_and_decode_fns


def load_config() -> Dict[str, Any]:
    """Load the local smol config."""
    cfg = OmegaConf.load(Path(__file__).resolve().parent / "Config.yml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]


def load_tokenizer(cfg: Dict[str, Any]):
    tok_cfg = cfg.get("tokenizer", {}) or {}
    tok_name = tok_cfg.get("name")
    if not tok_name:
        raise ValueError("tokenizer.name must be set in smol/Config.yml")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        use_fast=True,
        cache_dir=tok_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def build_model(cfg: Dict[str, Any], vocab_size: int, context_length: int) -> GiantGPT:
    model_cfg = cfg["model"]
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg["embedding_size"],
        n_heads=model_cfg["num_heads"],
        d_ff=model_cfg["feed_forward_size"],
        n_layers=model_cfg["num_layers"],
        dropout_rate=0.0,
    )


def tokenize_prompt(tokenizer, prompt: str, max_len: int, *, strip_eos: bool) -> np.ndarray:
    if strip_eos:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
    else:
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()

    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.asarray(ids, dtype=np.int32)


def load_params(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast text generation using the current SUPER-GIANT layout.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/smollm-135m.npz",
                        help="Path to a checkpoint (.npz).")
    parser.add_argument("--prompt", type=str, default="Once upon",
                        help="Prompt to feed the model.")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Zero switches to greedy decoding.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Optional top-k sampling cutoff (0 disables it).")
    parser.add_argument("--greedy", action="store_true",
                        help="Shortcut for --temperature 0.0.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for sampling.")
    parser.add_argument("--max_context", type=int, default=None,
                        help="Override context length from config.")
    parser.add_argument("--strip_eos", "--no_eos", action="store_true", dest="strip_eos",
                        help="Drop a trailing EOS token from the prompt before generation.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing stats.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    jax.config.update("jax_default_matmul_precision", cfg["model"]["compute_dtype"])

    temperature = 0.0 if args.greedy else max(args.temperature, 0.0)
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.top_k < 0:
        raise ValueError("top_k must be >= 0")
    do_sample = temperature > 0.0
    top_k = int(args.top_k)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
    checkpoint_path = checkpoint_path.resolve()
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    context_length = args.max_context or int(cfg["model"]["context_length"])
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens. Provide non-empty text.")

    model = build_model(cfg, len(tokenizer), context_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    params = load_params(checkpoint_path)

    rng = jax.random.PRNGKey(args.seed)
    key_params, key_dropout, key_sample = jax.random.split(rng, 3)
    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=pad_token_id,
        use_kv_cache=True,
    )

    params = jax.device_put(params)
    nonparam = jax.device_put(nonparam)

    prompt = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    compiled_prefill = prefill_fn.lower(params, nonparam, prompt).compile()
    compiled_decode = decode_fn.lower(
        params,
        nonparam,
        jnp.zeros((1, 1), jnp.int32),
        jnp.array(0, jnp.int32),
        steps=args.steps,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        rng_key=(key_sample if do_sample else None),
    ).compile()

    prefill_start = time.perf_counter()
    nonparam_filled, t_cur, last_tok = compiled_prefill(params, nonparam, prompt)
    block_until_ready(nonparam_filled)
    prefill_time = time.perf_counter() - prefill_start

    decode_start = time.perf_counter()
    tokens_new, _ = compiled_decode(
        params,
        nonparam_filled,
        last_tok,
        t_cur,
        temperature=temperature,
        rng_key=(key_sample if do_sample else None),
    )
    tokens_new.block_until_ready()
    decode_time = time.perf_counter() - decode_start

    generated = jnp.concatenate([prompt, tokens_new], axis=1)
    full_tokens = np.asarray(generated[0])
    text = tokenizer.decode(full_tokens, skip_special_tokens=True)

    inference_cfg = cfg.get("inference", {}) or {}
    stop_on_eos = bool(inference_cfg.get("stop_on_eos", False))
    eos_id = tokenizer.eos_token_id
    if stop_on_eos and eos_id is not None:
        idx = np.where(full_tokens == eos_id)[0]
        if idx.size > 0:
            cut = int(idx[0])
            text = tokenizer.decode(full_tokens[:cut], skip_special_tokens=True) + "<EOS>"

    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")

    if args.verbose:
        toks_per_s = (args.steps / decode_time) if decode_time > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {prompt.shape[1]}")
        print(f"generated_tokens: {args.steps}")
        print(f"prefill_time_s: {prefill_time:.6f}")
        print(f"decode_time_s:  {decode_time:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
