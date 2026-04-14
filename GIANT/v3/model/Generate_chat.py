from __future__ import annotations

import argparse
import re
from typing import List

import jax
import numpy as np

from GIANT.v3.model.Generate_faster import (
    align_tokenizer_and_params_vocab,
    build_model,
    choose_bucketed_context_length,
    default_auto_buckets,
    load_configs,
    load_params,
    load_tokenizer,
    normalize_buckets,
    parse_bool_flag,
    parse_int_list,
    resolve_checkpoint_path,
)
from GIANT.v3.model.Chat import generate_tokens
from GIANT.v3.model.jit_inference import init_inference_state, make_prefill_and_decode_fns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat-style generation for SUPER-GIANT checkpoints.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Optional first user turn.")
    parser.add_argument("--user", type=str, action="append", default=[])
    parser.add_argument(
        "--message",
        type=str,
        action="append",
        default=[],
        help="Explicit message in the form role:text. Can be passed multiple times.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_context", "--context_length", type=int, default=None, dest="max_context")
    parser.add_argument("--stop_on_eos", type=str, default=None)
    parser.add_argument("--kv_cache_buckets", type=str, default=None)
    parser.add_argument("--disable_kv_buckets", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Continue in stdin/stdout chat mode after any initial turns.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_messages(args: argparse.Namespace) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    for user_text in args.user:
        messages.append({"role": "user", "content": user_text})
    for raw in args.message:
        role, sep, content = raw.partition(":")
        if not sep:
            raise ValueError(f"Invalid --message value {raw!r}; expected role:text")
        role = role.strip()
        content = content.strip()
        if not role or not content:
            raise ValueError(f"Invalid --message value {raw!r}; expected role:text")
        messages.append({"role": role, "content": content})
    return messages


def _base_messages_for_new_context(args: argparse.Namespace) -> List[dict[str, str]]:
    base_messages: List[dict[str, str]] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    return base_messages


def _render_chat_prompt(tokenizer, messages: List[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ImportError:
            pass
    parts: List[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _truncate_prompt_ids(prompt_ids: List[int], context_length: int) -> np.ndarray:
    if not prompt_ids:
        raise ValueError("Chat prompt produced zero tokens.")
    if len(prompt_ids) >= context_length:
        prompt_ids = prompt_ids[-context_length:]
    return np.asarray(prompt_ids, dtype=np.int32)


def _trim_chat_response(tokenizer, generated_tokens: np.ndarray, stop_on_eos: bool) -> np.ndarray:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and int(im_end_id) >= 0:
        stop_positions = np.where(generated_tokens == int(im_end_id))[0]
        if stop_positions.size > 0:
            return generated_tokens[: int(stop_positions[0])]
    if stop_on_eos and tokenizer.eos_token_id is not None:
        stop_positions = np.where(generated_tokens == int(tokenizer.eos_token_id))[0]
        if stop_positions.size > 0:
            return generated_tokens[: int(stop_positions[0])]
    return generated_tokens


def _render_text_for_stdout(text: str) -> str:
    rendered = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("<0x0A>", "\n")
        .replace("<0x09>", "\t")
    )
    fence_pattern = re.compile(r"```([A-Za-z0-9_+.-]*)\s+(.+?)\s+```", re.DOTALL)

    def _format_code_body(lang: str, body: str) -> str:
        formatted = body.strip().replace("; ", ";\n")
        if "\n" not in formatted and lang.lower() in {"python", "py"}:
            formatted = re.sub(r":\s+(?=\S)", ":\n    ", formatted)
        return formatted

    def _fence_repl(match: re.Match[str]) -> str:
        lang = match.group(1)
        body = _format_code_body(lang, match.group(2))
        header = f"```{lang}" if lang else "```"
        return f"{header}\n{body}\n```"

    rendered = fence_pattern.sub(_fence_repl, rendered)
    rendered = re.sub(r"(?<!\n)(\d+\.\s+\*\*)", r"\n\1", rendered)
    rendered = re.sub(r"(?<!\n)(-\s+)", r"\n\1", rendered)
    return rendered


def _run_turn(
    *,
    tokenizer,
    messages: List[dict[str, str]],
    context_length: int,
    steps: int,
    stop_on_eos: bool,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    temperature: float,
    top_k: int,
    do_sample: bool,
    sample_key,
) -> tuple[str, float, float, object]:
    prompt_text = _render_chat_prompt(tokenizer, messages)
    prompt_ids = _truncate_prompt_ids(tokenizer.encode(prompt_text, add_special_tokens=False), context_length)
    generated_tokens, prefill_time, decode_time, sample_key = generate_tokens(
        params=params,
        base_state=base_state,
        prefill_fn=prefill_fn,
        decode_fn=decode_fn,
        prompt_ids=prompt_ids,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        rng_key=sample_key,
    )
    trimmed = _trim_chat_response(tokenizer, generated_tokens, stop_on_eos)
    response_text = _render_text_for_stdout(tokenizer.decode(trimmed, skip_special_tokens=True).strip())
    return response_text, prefill_time, decode_time, sample_key


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    cfg_temperature = float(getattr(cfg.inference, "temperature", 0.0))
    cfg_top_k = int(getattr(cfg.inference, "top_k", 0))
    cfg_steps = int(getattr(cfg.inference, "max_decode_steps", 128))
    cfg_stop_on_eos = bool(getattr(cfg.inference, "stop_on_eos", True))

    max_steps = args.steps if args.steps is not None else cfg_steps
    input_temperature = args.temperature if args.temperature is not None else cfg_temperature
    temperature = 0.0 if args.greedy else max(float(input_temperature), 0.0)
    top_k = int(args.top_k) if args.top_k is not None else cfg_top_k
    stop_on_eos = parse_bool_flag(args.stop_on_eos, default=cfg_stop_on_eos)
    do_sample = temperature > 0.0

    messages = _parse_messages(args)
    base_messages = _base_messages_for_new_context(args)
    has_non_system_seed = any(message.get("role") != "system" for message in messages)
    interactive = bool(args.interactive or not has_non_system_seed)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)

    model_context_length = int(cfg.model.context_length)
    requested_context_length = args.max_context or model_context_length
    context_length = requested_context_length

    params = load_params(checkpoint_path)
    rng = jax.random.PRNGKey(args.seed)
    rng, vocab_align_key = jax.random.split(rng)
    tokenizer, params, _, _ = align_tokenizer_and_params_vocab(tokenizer, params, rng_key=vocab_align_key)

    model = build_model(cfg, len(tokenizer), context_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    device = jax.devices()[0]
    key_params, key_dropout, sample_key = jax.random.split(rng, 3)
    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=pad_token_id,
        use_kv_cache=True,
    )
    params = jax.device_put(params, device)
    base_state = jax.device_put(nonparam, device)
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    def print_response(response_text: str, prefill_time: float, decode_time: float) -> None:
        print(f"GIANT: {response_text}\n")
        if args.verbose:
            toks_per_s = (max_steps / decode_time) if decode_time > 0 else float("inf")
            print("[perf]")
            print(f"prefill_time_s: {prefill_time:.6f}")
            print(f"decode_time_s:  {decode_time:.6f}")
            print(f"tokens_per_second_decode: {toks_per_s:.6f}\n")

    if messages and has_non_system_seed:
        response_text, prefill_time, decode_time, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        print_response(response_text, prefill_time, decode_time)
        messages.append({"role": "assistant", "content": response_text})
        if not interactive:
            return

    print("Enter text to chat with the model. Empty line or Ctrl+D exits.\n")
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

        if user_clean == "/new":
            messages = list(base_messages)
            print("Started new conversation.\n")
            continue

        if user_clean == "/clear":
            print("\033[2J\033[H", end="")
            continue

        messages.append({"role": "user", "content": user_clean})
        response_text, prefill_time, decode_time, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        messages.append({"role": "assistant", "content": response_text})
        print_response(response_text, prefill_time, decode_time)


if __name__ == "__main__":
    main()
