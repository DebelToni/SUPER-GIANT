from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from model.GiantGPT import GiantGPT
from model.tidar_masks import build_tidar_prefill_bias_cached, build_tidar_decode_bias_cached
from model.tidar_utils import sample_from_logits, rejection_sample


def _as_batch(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim == 1:
        return tokens[None, :]
    return tokens


def init_kv_cache(model: GiantGPT, *, batch_size: int, pad_token_id: int = 0):
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(0),
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
    )
    return variables["cache"]


def prefill_prompt_cache(
    model: GiantGPT,
    params,
    cache_vars,
    prompt_ids: np.ndarray,
) -> Tuple[object, int]:
    prompt_ids = _as_batch(prompt_ids)
    batch_size, prompt_len = prompt_ids.shape
    if prompt_len == 0:
        return cache_vars, 0
    position_ids = np.arange(prompt_len, dtype=np.int32)
    position_ids = np.broadcast_to(position_ids[None, :], (batch_size, prompt_len))

    _, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        jnp.asarray(prompt_ids),
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=0,
        position_ids=jnp.asarray(position_ids),
        mutable=["cache"],
    )
    return mutated["cache"], prompt_len


def tidar_prefill_draft_cached(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    prefix_len: int,
    batch_size: int,
    mask_id: int,
    draft_len: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    masks = np.full((batch_size, draft_len), mask_id, dtype=np.int32)
    position_ids = np.arange(prefix_len, prefix_len + draft_len, dtype=np.int32)
    position_ids = np.broadcast_to(position_ids[None, :], (batch_size, draft_len))

    attn_bias = build_tidar_prefill_bias_cached(prefix_len=prefix_len, draft_len=draft_len)
    logits = model.apply(
        {"params": params, "cache": cache_vars},
        jnp.asarray(masks),
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        prefix_len=prefix_len,
        attn_bias=jnp.asarray(attn_bias),
        position_ids=jnp.asarray(position_ids),
    )

    logits_np = np.asarray(logits)
    draft_logits = logits_np[:, :draft_len, :]
    draft_tokens = sample_from_logits(
        draft_logits[0],
        rng=rng,
        temperature=temperature,
        top_k=top_k,
    )
    return np.asarray(draft_tokens, dtype=np.int32), draft_logits[0]


def _build_decode_step_inputs_cached(
    verify_ids: np.ndarray,
    *,
    mask_id: int,
    draft_len: int,
    prefix_len: int,
):
    verify_ids = _as_batch(verify_ids)
    batch_size = verify_ids.shape[0]

    predraft = np.full((batch_size, draft_len * draft_len), mask_id, dtype=np.int32)
    step_tokens = np.concatenate([verify_ids, predraft], axis=1)

    pos_verify = np.arange(prefix_len, prefix_len + draft_len, dtype=np.int32)
    pos_predraft = []
    for r in range(1, draft_len + 1):
        pos_predraft.extend(np.arange(prefix_len + r, prefix_len + r + draft_len, dtype=np.int32))
    pos_predraft = np.asarray(pos_predraft, dtype=np.int32)

    position_ids = np.concatenate(
        [
            np.broadcast_to(pos_verify[None, :], (batch_size, draft_len)),
            np.broadcast_to(pos_predraft[None, :], (batch_size, draft_len * draft_len)),
        ],
        axis=1,
    )

    attn_bias = build_tidar_decode_bias_cached(prefix_len=prefix_len, draft_len=draft_len)
    return step_tokens, position_ids, attn_bias


def commit_tokens_to_cache(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    committed: np.ndarray,
    prefix_len: int,
) -> Tuple[object, int]:
    committed = np.asarray(committed, dtype=np.int32)
    if committed.size == 0:
        return cache_vars, prefix_len
    tokens = _as_batch(committed)
    batch_size, step_len = tokens.shape
    position_ids = np.arange(prefix_len, prefix_len + step_len, dtype=np.int32)
    position_ids = np.broadcast_to(position_ids[None, :], (batch_size, step_len))

    _, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        jnp.asarray(tokens),
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=prefix_len,
        position_ids=jnp.asarray(position_ids),
        mutable=["cache"],
    )
    return mutated["cache"], prefix_len + step_len


def tidar_decode_step_cached(
    model: GiantGPT,
    params,
    cache_vars,
    *,
    prefix_len: int,
    prefix_ids: np.ndarray,
    verify_ids: np.ndarray,
    mask_id: int,
    draft_len: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
    draft_logits: Optional[np.ndarray] = None,
    always_accept: bool = False,
    max_commit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, object, int]:
    step_tokens, position_ids, attn_bias = _build_decode_step_inputs_cached(
        verify_ids,
        mask_id=mask_id,
        draft_len=draft_len,
        prefix_len=prefix_len,
    )

    logits = model.apply(
        {"params": params, "cache": cache_vars},
        jnp.asarray(step_tokens),
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        prefix_len=prefix_len,
        attn_bias=jnp.asarray(attn_bias),
        position_ids=jnp.asarray(position_ids),
    )

    logits_np = np.asarray(logits)
    verify_logits = logits_np[:, :draft_len, :]

    verify_ids_1d = verify_ids[0] if verify_ids.ndim == 2 else verify_ids
    if always_accept:
        if max_commit is None:
            r = draft_len
        else:
            r = int(min(draft_len, max_commit))
            if r < 1:
                raise ValueError("max_commit must be >= 1 when always_accept=True")
        committed = np.asarray(verify_ids_1d[:r], dtype=np.int32)
    else:
        draft_logits_1d = None
        if draft_logits is not None:
            draft_logits_1d = draft_logits[0] if draft_logits.ndim == 3 else draft_logits

        r, committed = rejection_sample(
            verify_ids_1d,
            verify_logits[0],
            rng=rng,
            draft_logits=draft_logits_1d,
        )

    cand_start = draft_len + (r - 1) * draft_len
    cand_end = cand_start + draft_len
    cand_logits = logits_np[:, cand_start:cand_end, :]
    next_verify = sample_from_logits(
        cand_logits[0],
        rng=rng,
        temperature=temperature,
        top_k=top_k,
    )

    cache_vars, prefix_len = commit_tokens_to_cache(
        model,
        params,
        cache_vars,
        committed=committed,
        prefix_len=prefix_len,
    )

    new_prefix = np.concatenate([_as_batch(prefix_ids)[0], committed], axis=0)
    return (
        new_prefix.astype(np.int32),
        np.asarray(next_verify, dtype=np.int32),
        cand_logits[0],
        r,
        cache_vars,
        prefix_len,
    )
