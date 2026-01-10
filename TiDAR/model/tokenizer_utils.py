from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import core as flax_core


def _lecun_embedding_init(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
    fan_in = shape[-1]
    std = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
    return jax.random.normal(key, shape, dtype) * std


def ensure_tidar_mask_token(
    tokenizer,
    *,
    base_token: str = "[MASK]",
    alt_prefix: str = "[TIDAR_MASK]",
) -> Tuple[str, int, int]:
    """Ensure a fresh mask token exists, returning (token_str, token_id, added_count)."""
    vocab = tokenizer.get_vocab()
    if base_token in vocab:
        candidate = alt_prefix
        suffix = 0
        while candidate in vocab:
            suffix += 1
            candidate = f"{alt_prefix}_{suffix}"
    else:
        candidate = base_token

    old_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [candidate]})
    try:
        tokenizer.add_special_tokens({"mask_token": candidate})
    except Exception:
        pass
    new_size = len(tokenizer)
    added = new_size - old_size

    token_id = tokenizer.convert_tokens_to_ids(candidate)
    if token_id is None or token_id < 0:
        raise ValueError(f"Failed to resolve mask token id for '{candidate}'")
    return candidate, int(token_id), int(added)


def resize_embedding_params(
    params,
    new_vocab_size: int,
    *,
    key: jax.Array,
    init_fn=None,
):
    """Resize embedding matrix to new_vocab_size; return (params, added_rows)."""
    init_fn = init_fn or _lecun_embedding_init
    is_frozen = isinstance(params, flax_core.FrozenDict)
    params_mut = flax_core.unfreeze(params) if is_frozen else params

    embedding = params_mut["Embed_0"]["embedding"]
    old_vocab_size, hidden = embedding.shape
    if new_vocab_size <= old_vocab_size:
        return params, 0

    add_rows = new_vocab_size - old_vocab_size
    new_rows = init_fn(key, (add_rows, hidden), embedding.dtype)
    new_embedding = jnp.concatenate([embedding, new_rows], axis=0)
    params_mut["Embed_0"]["embedding"] = new_embedding
    params_out = flax_core.freeze(params_mut) if is_frozen else params_mut
    return params_out, add_rows


def init_mask_embedding_row(
    params,
    mask_id: int,
    *,
    key: jax.Array,
    init_fn=None,
):
    """Initialize the mask token embedding row with LeCun weights."""
    init_fn = init_fn or _lecun_embedding_init
    is_frozen = isinstance(params, flax_core.FrozenDict)
    params_mut = flax_core.unfreeze(params) if is_frozen else params

    embedding = params_mut["Embed_0"]["embedding"]
    if mask_id < 0 or mask_id >= embedding.shape[0]:
        raise ValueError(f"mask_id {mask_id} out of bounds for embedding size {embedding.shape[0]}")
    new_row = init_fn(key, (1, embedding.shape[1]), embedding.dtype)[0]
    params_mut["Embed_0"]["embedding"] = embedding.at[mask_id].set(new_row)
    params_out = flax_core.freeze(params_mut) if is_frozen else params_mut
    return params_out
