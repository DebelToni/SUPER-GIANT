from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from checkpoint_io import load_npz
from checkpoint_manager import latest as latest_ckpt
from jit_inference import init_inference_state, make_prefill_and_decode_fns


def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    global_cfg = OmegaConf.load(project_root / "Global_Config.yml")
    local_cfg = OmegaConf.load(model_dir / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str((base_prefix / path).resolve())

    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)

    for key in ("processed_data_root", "dataloader_state_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "answers_arrow" in cfg.qa_finetune and cfg.qa_finetune.answers_arrow is not None:
        resolved = resolve_path(cfg.qa_finetune.answers_arrow)
        if resolved is not None:
            cfg.qa_finetune.answers_arrow = resolved

    if "checkpoint_dir" in cfg.qa_finetune and cfg.qa_finetune.checkpoint_dir is not None:
        resolved = resolve_path(cfg.qa_finetune.checkpoint_dir)
        if resolved is not None:
            cfg.qa_finetune.checkpoint_dir = resolved

    return cfg


def resolve_with_data_root(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def resolve_checkpoint_path(cfg: OmegaConf, checkpoint: Optional[str], checkpoint_dir: str) -> Path:
    base_root = Path(cfg.paths.data_root)
    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = resolve_with_data_root(base_root, checkpoint)
        if path.is_dir():
            latest = latest_ckpt(str(path))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {path}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

    ckpt_dir = resolve_with_data_root(base_root, checkpoint_dir)
    latest = latest_ckpt(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return Path(latest)


def load_tokenizer(cfg: OmegaConf):
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def build_model(cfg: OmegaConf, vocab_size: int, context_length: int) -> GiantGPT:
    model_cfg = cfg.model
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
    )


def load_params(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def clone_state(tree):
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def enforce_context_limit(buffer: List[int], max_len: int) -> None:
    overflow = len(buffer) - max_len
    if overflow > 0:
        del buffer[:overflow]


def encode_prompt(tokenizer, text: str, max_len: int, *, strip_eos: bool) -> np.ndarray:
    if strip_eos:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
    else:
        ids = tokenizer(text, return_tensors="np").input_ids[0].tolist()
    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.asarray(ids, dtype=np.int32)


def encode_segment(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def trim_generated(ids: np.ndarray, eos_id: Optional[int], stop_on_eos: bool) -> Tuple[np.ndarray, bool]:
    if ids.size == 0 or eos_id is None or not stop_on_eos:
        return ids, False
    idx = np.where(ids == eos_id)[0]
    if idx.size == 0:
        return ids, False
    return ids[: idx[0]], True


def block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def generate_tokens(
    *,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    prompt_ids: np.ndarray,
    steps: int,
    temperature: float,
    top_k: int,
    do_sample: bool,
    rng_key: jax.random.KeyArray,
) -> Tuple[np.ndarray, float, float, jax.random.KeyArray]:
    state = clone_state(base_state)
    prompt = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)

    prefill_start = time.perf_counter()
    nonparam, t_cur, last_tok = prefill_fn(params, state, prompt)
    block_tree((nonparam, last_tok))
    prefill_time = time.perf_counter() - prefill_start

    rng = None
    new_rng = rng_key
    if do_sample:
        new_rng, rng = jax.random.split(rng_key)

    decode_start = time.perf_counter()
    tokens_new, _ = decode_fn(
        params,
        nonparam,
        last_tok,
        t_cur,
        steps=steps,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        rng_key=rng,
    )
    tokens_new.block_until_ready()
    decode_time = time.perf_counter() - decode_start
    return np.asarray(tokens_new[0]), prefill_time, decode_time, new_rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive text generation for SUPER-GIANT with optional chat history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="latest",
                        help="Path to a checkpoint (.npz). Defaults to the newest file in --checkpoint_dir.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory (relative to data_root) when --checkpoint is omitted or set to 'latest'.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional first prompt to run before entering interactive mode.")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of tokens to generate per turn.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling cutoff (0 disables it).")
    parser.add_argument("--greedy", action="store_true",
                        help="Shortcut for --temperature 0.0.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed.")
    parser.add_argument("--max_context", type=int, default=None,
                        help="Override context length from the model config.")
    parser.add_argument("--strip_eos", "--no_eos", action="store_true", dest="strip_eos",
                        help="Strip trailing EOS token from standalone prompts.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing for prefill/decode per turn.")
    parser.add_argument("--chat", action="store_true",
                        help="Preserve conversation history between turns.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.top_k < 0:
        raise ValueError("top_k must be >= 0")

    temperature = 0.0 if args.greedy else max(args.temperature, 0.0)
    do_sample = temperature > 0.0
    top_k = int(args.top_k)

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    context_length = args.max_context or int(cfg.model.context_length)

    model = build_model(cfg, len(tokenizer), context_length)
    params = load_params(checkpoint_path)

    rng = jax.random.PRNGKey(args.seed)
    key_params, key_dropout, sample_key = jax.random.split(rng, 3)

    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0),
        use_kv_cache=True,
    )

    params = jax.device_put(params)
    base_state = jax.device_put(nonparam)

    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    history_tokens: List[int] = []
    newline_tokens = encode_segment(tokenizer, "\n")

    def run_turn(user_text: str) -> None:
        nonlocal sample_key
        text = user_text.strip()
        if not text:
            return

        if args.chat:
            segment = f"User: {text}\nGIANT:"
            segment_tokens = encode_segment(tokenizer, segment)
            history_tokens.extend(segment_tokens)
            enforce_context_limit(history_tokens, context_length)
            if not history_tokens:
                print("Skipped turn: conversation window truncated everything.")
                return
            prompt_ids = np.asarray(history_tokens, dtype=np.int32)
        else:
            prompt_ids = encode_prompt(tokenizer, text, context_length, strip_eos=args.strip_eos)
            if prompt_ids.size == 0:
                print("Prompt produced zero tokens; please enter different text.")
                return

        tokens_new, prefill_time, decode_time, sample_key = generate_tokens(
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            prompt_ids=prompt_ids,
            steps=args.steps,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            rng_key=sample_key,
        )

        stop_on_eos = bool(getattr(cfg.inference, "stop_on_eos", False)) if hasattr(cfg, "inference") else False
        trimmed, hit_eos = trim_generated(tokens_new, tokenizer.eos_token_id, stop_on_eos)
        response_text = tokenizer.decode(trimmed if hit_eos else tokens_new, skip_special_tokens=True).strip()
        if hit_eos:
            response_text = response_text + "<EOS>"

        if args.chat:
            history_tokens.extend(trimmed.tolist())
            history_tokens.extend(newline_tokens)
            enforce_context_limit(history_tokens, context_length)

        print(f"GIANT: {response_text}\n")

        if args.verbose:
            toks_per_s = (args.steps / decode_time) if decode_time > 0 else float("inf")
            print(
                f"[perf] prefill={prefill_time:.4f}s decode={decode_time:.4f}s "
                f"tokens/s={toks_per_s:.2f}"
            )

    print("Enter text to chat with the model. Empty line or Ctrl+D exits.\n")
    echo_user_input = not sys.stdin.isatty()

    if args.prompt:
        print(f"User: {args.prompt}")
        run_turn(args.prompt)

    while True:
        try:
            user_text = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        user_clean = user_text.strip()
        if not user_clean:
            print("Exiting.")
            break

        if echo_user_input:
            print(f"User: {user_clean}")
        run_turn(user_clean)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
