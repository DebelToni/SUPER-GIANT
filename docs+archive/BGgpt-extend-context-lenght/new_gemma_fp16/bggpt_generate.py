# bggpt_generate.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from bggpt_config import BgGPTConfig
from bggpt_model_jax import (
    KVCache,
    build_rope_cache,
    forward_decode_one,
    forward_prefill,
    init_kv_cache,
)


@dataclass
class GenerationConfigJax:
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_k: int = 25
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    eos_token_ids: Tuple[int, ...] = (1, 107)


def apply_repetition_penalty(
    logits: jnp.ndarray,
    prev_tokens: jnp.ndarray,
    penalty: float,
) -> jnp.ndarray:
    """
    HF-style repetition penalty.

    Args:
        logits: (vocab,)
        prev_tokens: (seq_len,) int32
        penalty: float

    Returns:
        new_logits: (vocab,)
    """
    if penalty == 1.0:
        return logits

    # Gather logits for previously generated tokens
    gathered = logits[prev_tokens]  # (seq_len,)
    # Positive logits divided, negative multiplied
    penalized = jnp.where(
        gathered > 0, gathered / penalty, gathered * penalty
    )
    logits = logits.at[prev_tokens].set(penalized)
    return logits


def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -1e9,
) -> jnp.ndarray:
    """
    Apply top-k and/or top-p (nucleus) filtering to logits.

    Args:
        logits: (vocab,)
    """
    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        thresh = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < thresh, filter_value, logits)

    # Top-p
    if top_p < 1.0:
        # Sort by descending logit
        sorted_logits = jnp.sort(logits)[::-1]
        sorted_indices = jnp.argsort(logits)[::-1]
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumprobs = jnp.cumsum(probs, axis=-1)
        mask = cumprobs > top_p
        # Always keep at least 1 token
        mask = mask.at[0].set(False)
        filtered_logits = jnp.where(mask, filter_value, sorted_logits)
        # Scatter back to original order
        logits = jnp.full_like(logits, filter_value)
        logits = logits.at[sorted_indices].set(filtered_logits)

    return logits


def sample_token(
    logits: jnp.ndarray,
    rng_key: jax.Array,
) -> Tuple[int, jax.Array]:
    """
    Sample a token given logits.

    Args:
        logits: (vocab,)
        rng_key: PRNG key.

    Returns:
        (token_id, new_rng_key)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    rng_key, subkey = jax.random.split(rng_key)
    token_id = int(jax.random.categorical(subkey, jnp.log(probs)))
    return token_id, rng_key


def generate(
    params,
    config: BgGPTConfig,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_cfg: GenerationConfigJax,
    rng_key: jax.Array,
    max_seq_len: Optional[int] = None,
) -> str:
    """
    High-level generate function. Handles:

    - chat-template formatting
    - prompt prefill
    - auto-regressive decode
    """
    model_name = tokenizer.name_or_path

    # Use Gemma 2 chat template if defined
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(
        formatted,
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids_list: List[int] = enc["input_ids"]
    input_ids = jnp.array(input_ids_list, dtype=jnp.int32)[None, :]  # (1,T)

    if max_seq_len is None:
        max_seq_len = config.max_position_embeddings

    if input_ids.shape[1] >= max_seq_len:
        raise ValueError(
            f"Prompt length {input_ids.shape[1]} >= max_seq_len={max_seq_len}"
        )

    # Build RoPE cache
    rope_cache = build_rope_cache(config, max_seq_len=max_seq_len)

    # Prefill
    logits_prefill = forward_prefill(params, config, input_ids, rope_cache)
    # last token logits
    last_logits = logits_prefill[:, -1, :]  # (1,V)

    # Init KV cache and re-run decode on all prompt tokens to fill cache
    # (simple path: forward_prefill doesn't maintain cache)
    B = 1
    kv_cache = init_kv_cache(
        config, batch_size=B, max_seq_len=max_seq_len, dtype=jnp.float16
    )

    # Build cache via decode path over prompt tokens
    seq = input_ids
    for pos in range(seq.shape[1]):
        token_step = seq[:, pos : pos + 1]
        _, kv_cache = forward_decode_one(
            params, config, token_step, rope_cache, kv_cache, pos
        )

    generated: List[int] = []
    cur_seq = input_ids_list[:]  # python list of ints

    cur_pos = seq.shape[1]
    eos_ids = set(gen_cfg.eos_token_ids)

    for step in range(gen_cfg.max_new_tokens):
        # Decode next token
        last_token = jnp.array([[cur_seq[-1]]], dtype=jnp.int32)
        logits, kv_cache = forward_decode_one(
            params, config, last_token, rope_cache, kv_cache, cur_pos
        )
        logits = logits[0]  # (V,)

        # Apply temperature
        if gen_cfg.temperature != 1.0:
            logits = logits / gen_cfg.temperature

        # Apply repetition penalty
        prev_tokens = jnp.array(cur_seq, dtype=jnp.int32)
        logits = apply_repetition_penalty(
            logits, prev_tokens, gen_cfg.repetition_penalty
        )

        # Top-k/top-p
        logits = top_k_top_p_filtering(
            logits, top_k=gen_cfg.top_k, top_p=gen_cfg.top_p
        )

        # Sample
        token_id, rng_key = sample_token(logits, rng_key)
        cur_seq.append(int(token_id))
        generated.append(int(token_id))
        cur_pos += 1

        if token_id in eos_ids:
            break

        if cur_pos >= max_seq_len:
            break

    full_ids = cur_seq
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    # For convenience, return only generated tail after the original prompt
    full_text = text
    return full_text

