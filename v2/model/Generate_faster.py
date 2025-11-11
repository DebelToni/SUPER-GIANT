from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omegaconf import OmegaConf

from GiantGPT import GiantGPT
from checkpoint_manager import latest as latest_checkpoint
from checkpoint_io import load_npz
from jit_inference import init_inference_state, make_prefill_and_decode_fns

from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", "tensorfloat32")


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent


def load_configs() -> OmegaConf:
    global_cfg = OmegaConf.load(PROJECT_ROOT / "Global_Config.yml")
    local_cfg = OmegaConf.load(MODEL_DIR / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or not cfg.paths.get("data_root"):
            return str(path)
        return str(Path(cfg.paths.data_root) / path)

    base_root = Path(cfg.paths.get("data_root", PROJECT_ROOT))
    cfg.paths.data_root = str(base_root)

    for key in ("processed_data_root", "dataloader_state_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            cfg.paths[key] = resolve_path(cfg.paths[key]) or cfg.paths[key]

    if cfg.qa_finetune.get("answers_arrow"):
        cfg.qa_finetune.answers_arrow = resolve_path(cfg.qa_finetune.answers_arrow)

    if cfg.qa_finetune.get("checkpoint_dir"):
        cfg.qa_finetune.checkpoint_dir = resolve_path(cfg.qa_finetune.checkpoint_dir)

    return cfg


def build_model(cfg: OmegaConf, vocab_size: int) -> GiantGPT:
    model_cfg = cfg.model
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=model_cfg.context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
    )


def preprocess_prompt_no_EOS(tokenizer, prompt: str, max_len: int):
    """Tokenize prompt without EOS token, matching Generate_text_fast.py"""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.array(ids, dtype="int32")


def preprocess_prompt(tokenizer, prompt: str, max_len: int):
    """Tokenize prompt with special tokens, matching Generate_text_fast.py"""
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if ids.shape[0] >= max_len:
        ids = ids[-max_len:]
    return ids.astype("int32")


def main():
    ap = argparse.ArgumentParser(description="Fast JIT-compiled text generation using GiantGPT")
    ap.add_argument("--checkpoint", type=Path, default="model_params.pkl",
                    help="Path to model checkpoint (.pkl, .pickle, .npz)")
    ap.add_argument("--prompt", type=str, default="Once upon",
                    help="Text prompt for generation")
    ap.add_argument("--steps", type=int, default=20,
                    help="Number of tokens to generate (same as max_new_tokens)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 = greedy)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="Top-k sampling (None = no top-k)")
    ap.add_argument("--greedy", action="store_true",
                    help="Force greedy decoding (temperature=0.0)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print timing and tokens/sec for the decode phase")
    ap.add_argument("--no_eos", action="store_true",
                    help="Remove EOS token from prompt preprocessing")

    args = ap.parse_args()

    # Handle temperature/greedy settings
    temperature = 0.0 if args.greedy else args.temperature
    do_sample = temperature > 0.0
    top_k_int = 0 if args.top_k is None else int(args.top_k)

    print("\nLoading checkpoint…")
    params_cpu = load_checkpoint(args.checkpoint)

    print("Building model…")
    model = build_model()

    # Load tokenizer (matching Generate_text_fast.py exactly)
    if Config.use_custom_tokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(Config.custom_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    # Preprocess prompt
    if args.no_eos:
        prompt_ids = preprocess_prompt_no_EOS(tokenizer, args.prompt, Config.context_length)
    else:
        prompt_ids = preprocess_prompt(tokenizer, args.prompt, Config.context_length)

    prompt = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)  # [1, Lp]

    # Initialize inference state for JIT (creates cache structure)
    key = jax.random.PRNGKey(42)
    k_params, k_drop, k_sample = jax.random.split(key, 3)
    _, nonparam = init_inference_state(model, k_params, k_drop, batch_size=1, pad_token_id=0, use_kv_cache=True)

    # Replace dummy params with loaded params
    params = params_cpu

    # Put state on default device once
    params = jax.device_put(params)
    nonparam = jax.device_put(nonparam)

    # Build JIT-ed prefill & decode functions (capture `model` statically)
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    # --- Compile both ahead-of-time for these exact shapes/flags ---
    compiled_prefill = prefill_fn.lower(params, nonparam, prompt).compile()
    compiled_decode = decode_fn.lower(
        params, nonparam, jnp.zeros((1,1), jnp.int32), jnp.array(0, jnp.int32),
        steps=args.steps, do_sample=do_sample, top_k=top_k_int, temperature=temperature,
        rng_key=(k_sample if do_sample else None),
    ).compile()

    # --- Run prefill (not included in decode timing) ---
    nonparam2, t_cur, last_tok_2d = compiled_prefill(params, nonparam, prompt)
    # Ensure prefill finished
    jax.tree_util.tree_leaves(nonparam2)[0].block_until_ready()

    # --- Decode & time ---
    t0 = time.perf_counter()
    tokens_new, nonparam_out = compiled_decode(
        params, nonparam2, last_tok_2d, t_cur,
        temperature=temperature,
        rng_key=(k_sample if do_sample else None),
    )

    tokens_new.block_until_ready()
    dt = time.perf_counter() - t0

    # Decode to text (not timed)
    full_tokens = jnp.concatenate([prompt, tokens_new], axis=1)
    text = tokenizer.decode(np.asarray(full_tokens[0]), skip_special_tokens=True)

    print("\n" + "="*20 + " RESULT " + "="*20)
    print(text)
    print("="*48)

    if args.verbose:
        toks_per_s = args.steps / dt if dt > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {prompt.shape[1]}")
        print(f"generated_tokens: {args.steps}")
        print(f"decode_time_s:  {dt:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
