from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from model.tidar_masks import build_tidar_train_bias


IGNORE_INDEX = -100


def build_train_batch(
    tokens: jnp.ndarray,
    lengths: Optional[jnp.ndarray],
    *,
    mask_id: int,
    block_len: int,
    bias_value: float = -1.0e10,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, jnp.ndarray]:
    """Build TiDAR training inputs for a batch of token sequences."""
    batch_size, seq_len = tokens.shape
    clean = tokens
    diff = jnp.full_like(clean, mask_id)

    input_ids = jnp.concatenate([clean, diff], axis=1)

    pos = jnp.arange(seq_len, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(pos[None, :], (batch_size, seq_len))
    position_ids = jnp.concatenate([position_ids, position_ids], axis=1)

    labels = jnp.full((batch_size, seq_len * 2), ignore_index, dtype=jnp.int32)
    labels = labels.at[:, : seq_len - 1].set(clean[:, 1:])
    labels = labels.at[:, seq_len:].set(clean)

    loss_mask_ntp = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)
    loss_mask_diff = jnp.zeros((batch_size, seq_len * 2), dtype=jnp.float32)

    if lengths is None:
        valid = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    else:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        valid = (positions < lengths[:, None]).astype(jnp.float32)

    if seq_len > 1:
        loss_mask_ntp = loss_mask_ntp.at[:, : seq_len - 1].set(valid[:, 1:])
    loss_mask_diff = loss_mask_diff.at[:, seq_len:].set(valid)

    token_types = jnp.concatenate(
        [jnp.zeros(seq_len, dtype=jnp.int32), jnp.ones(seq_len, dtype=jnp.int32)]
    )
    attn_bias = build_tidar_train_bias(
        position_ids,
        token_types,
        block_len=block_len,
        bias_value=bias_value,
    )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask_ntp": loss_mask_ntp,
        "loss_mask_diff": loss_mask_diff,
        "attn_bias": attn_bias,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-9)


def sample_from_logits(
    logits: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    if temperature <= 0:
        return np.argmax(logits, axis=-1)
    scaled = logits / max(temperature, 1e-6)
    if top_k > 0:
        top_indices = np.argpartition(-scaled, top_k - 1, axis=-1)[..., :top_k]
        mask = np.full_like(scaled, -np.inf)
        np.put_along_axis(mask, top_indices, np.take_along_axis(scaled, top_indices, axis=-1), axis=-1)
        scaled = mask
    probs = _softmax(scaled)
    if probs.ndim == 1:
        return rng.choice(probs.shape[0], p=probs)
    out = np.zeros(probs.shape[0], dtype=np.int32)
    for i in range(probs.shape[0]):
        out[i] = rng.choice(probs.shape[1], p=probs[i])
    return out


def rejection_sample(
    verify_ids: np.ndarray,
    verify_logits: np.ndarray,
    *,
    rng: np.random.Generator,
    draft_logits: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """Return (r, committed_tokens) using speculative-style rejection sampling."""
    verify_ids = np.asarray(verify_ids)
    verify_logits = np.asarray(verify_logits)
    k = verify_ids.shape[0]
    p = _softmax(verify_logits)

    if draft_logits is None:
        q = p
    else:
        q = _softmax(np.asarray(draft_logits))

    committed = []
    r = 0
    for i in range(k):
        token = verify_ids[i]
        p_tok = p[i, token]
        q_tok = q[i, token]
        accept_prob = min(1.0, float(p_tok / max(q_tok, 1e-9)))
        if rng.random() < accept_prob:
            committed.append(token)
            r += 1
            continue
        new_tok = rng.choice(p.shape[1], p=p[i])
        committed.append(new_tok)
        r += 1
        break

    if r == 0:
        new_tok = rng.choice(p.shape[1], p=p[0])
        committed = [new_tok]
        r = 1
    return r, np.asarray(committed, dtype=np.int32)
