from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import jax
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
    key_padding_mask = jnp.concatenate([valid, valid], axis=1) > 0
    attn_bias = build_tidar_train_bias(
        position_ids,
        token_types,
        block_len=block_len,
        key_padding_mask=key_padding_mask,
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


def _apply_temperature_top_k(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    if top_k > 0:
        top_indices = np.argpartition(-scaled, top_k - 1, axis=-1)[..., :top_k]
        masked = np.full_like(scaled, -np.inf)
        np.put_along_axis(masked, top_indices, np.take_along_axis(scaled, top_indices, axis=-1), axis=-1)
        scaled = masked
    return scaled


def rejection_sample(
    verify_ids: np.ndarray,
    verify_logits: np.ndarray,
    *,
    rng: np.random.Generator,
    draft_logits: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[int, np.ndarray]:
    """Return (r, committed_tokens) using speculative-style rejection sampling."""
    verify_ids = np.asarray(verify_ids)
    verify_logits = _apply_temperature_top_k(
        np.asarray(verify_logits), temperature=temperature, top_k=top_k
    )
    k = verify_ids.shape[0]
    p = _softmax(verify_logits)

    if draft_logits is None:
        q = p
    else:
        q = _softmax(
            _apply_temperature_top_k(np.asarray(draft_logits), temperature=temperature, top_k=top_k)
        )

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


def jax_topk_mask(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    if top_k <= 0:
        return logits
    values, indices = jax.lax.top_k(logits, top_k)
    masked = jnp.full_like(logits, -jnp.inf)
    masked = masked.at[..., indices].set(values)
    return masked


def jax_sample(
    logits: jnp.ndarray,
    *,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
) -> jnp.ndarray:
    scaled = logits / jnp.maximum(temperature, 1e-6)
    scaled = jax_topk_mask(scaled, top_k)
    token = jax.random.categorical(key, scaled, axis=-1)
    return token.astype(jnp.int32)


def jax_rejection_sample(
    verify_ids: jnp.ndarray,
    verify_logits: jnp.ndarray,
    *,
    key: jax.Array,
    draft_logits: Optional[jnp.ndarray] = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[jax.Array, jnp.ndarray]:
    k = verify_ids.shape[0]
    scaled_verify = verify_logits / jnp.maximum(temperature, 1e-6)
    scaled_verify = jax_topk_mask(scaled_verify, top_k)
    probs = jax.nn.softmax(scaled_verify, axis=-1)
    if draft_logits is None:
        draft_probs = probs
    else:
        scaled_draft = draft_logits / jnp.maximum(temperature, 1e-6)
        scaled_draft = jax_topk_mask(scaled_draft, top_k)
        draft_probs = jax.nn.softmax(scaled_draft, axis=-1)

    key_accept, key_resample = jax.random.split(key)
    accept_u = jax.random.uniform(key_accept, (k,))
    resample_keys = jax.random.split(key_resample, k)

    committed = jnp.zeros((k,), dtype=jnp.int32)

    def step(carry, i):
        done, r, committed = carry
        tok = verify_ids[i]
        p_tok = probs[i, tok]
        q_tok = draft_probs[i, tok]
        accept_prob = jnp.minimum(1.0, p_tok / jnp.maximum(q_tok, 1e-9))
        accept = accept_u[i] < accept_prob

        def do_accept(state):
            done, r, committed = state
            committed = committed.at[i].set(tok)
            return done, i + 1, committed

        def do_reject(state):
            done, r, committed = state
            new_tok = jax.random.categorical(resample_keys[i], verify_logits[i], axis=-1)
            committed = committed.at[i].set(new_tok.astype(jnp.int32))
            return True, i + 1, committed

        def do_skip(state):
            return state

        carry = jax.lax.cond(done, do_skip, lambda s: jax.lax.cond(accept, do_accept, do_reject, s), carry)
        return carry, None

    (done, r, committed), _ = jax.lax.scan(step, (False, jnp.array(0, jnp.int32), committed), jnp.arange(k))
    r = jnp.maximum(r, 1)
    return r, committed
