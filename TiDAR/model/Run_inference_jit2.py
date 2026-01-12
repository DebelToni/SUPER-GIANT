from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TiDAR.model.TiDAR import TiDAR
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from GIANT.v2.model.checkpoint_manager import load_npz, latest as latest_ckpt

DEFAULT_CACHE_BUCKETS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)


# =========================
# Config / IO helpers
# =========================
def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(model_dir / "Config.yml"),
    )

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

    for key in ("checkpoints_root", "hf_cache_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "tokenizer" in cfg:
        cache_dir = cfg.tokenizer.get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cfg.tokenizer.cache_dir = str(Path(cfg.paths.data_root) / cache_path)
        custom_path = cfg.tokenizer.get("custom_path")
        if custom_path:
            custom_path = Path(str(custom_path))
            if not custom_path.is_absolute():
                cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path)

    return cfg


def resolve_checkpoint_path(cfg: OmegaConf, checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    base_root = Path(cfg.paths.data_root)
    ckpt_dir = Path(checkpoint_dir or cfg.paths.checkpoints_root)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()

    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        if path.is_dir():
            latest = latest_ckpt(str(path))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {path}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

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


def build_model(cfg: OmegaConf, vocab_size: int, context_length: int) -> TiDAR:
    model_cfg = cfg.model
    return TiDAR(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
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


# =========================
# Bucketing
# =========================
def parse_cache_buckets(raw: Optional[str], *, context_length: int) -> Tuple[int, ...]:
    if raw is None:
        return tuple(b for b in DEFAULT_CACHE_BUCKETS if b <= context_length)
    buckets = tuple(sorted({int(v) for v in raw.split(",") if v.strip()}))
    return tuple(b for b in buckets if b <= context_length)


def select_cache_bucket(required_len: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if bucket >= required_len:
            return bucket
    raise ValueError(f"No cache bucket >= {required_len} (available: {buckets})")


# =========================
# KV-cache init / prefix prefill
# =========================
def init_kv_cache(model: TiDAR, *, batch_size: int, pad_token_id: int) -> object:
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
    )
    return variables["cache"]


def prefill_prompt_cache(
    model: TiDAR,
    params,
    cache_vars,
    prompt_ids: np.ndarray,
    *,
    kv_cache_len: int,
) -> Tuple[object, int]:
    prompt_ids = np.asarray(prompt_ids, dtype=np.int32)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
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
        kv_cache_len=kv_cache_len,
        mutable=["cache"],
    )
    return mutated["cache"], prompt_len


# =========================
# TiDAR masks / position ids
# =========================
def build_decode_position_ids(prefix_len: jnp.ndarray, draft_len: int) -> jnp.ndarray:
    """
    JAX version (static draft_len): positions for TiDAR decode layout [VERIFY(K) | PREDRAFT(K*K)].
    VERIFY:   [L .. L+K-1]
    PREDRAFT: for r=1..K, block r predicts [L+r .. L+r+K-1]
    """
    # verify
    pos_verify = prefix_len + jnp.arange(draft_len, dtype=jnp.int32)

    # predraft offsets: shape (K,K): r + t, where r in [1..K], t in [0..K-1]
    r = jnp.arange(1, draft_len + 1, dtype=jnp.int32)[:, None]
    t = jnp.arange(draft_len, dtype=jnp.int32)[None, :]
    offsets = (r + t).reshape(-1)  # length K*K
    pos_predraft = prefix_len + offsets

    return jnp.concatenate([pos_verify, pos_predraft], axis=0)  # length K + K*K


def build_decode_position_ids_template(draft_len: int) -> jnp.ndarray:
    """Build position offsets template for decode step (0-indexed)."""
    pos_verify = jnp.arange(draft_len, dtype=jnp.int32)
    r = jnp.arange(1, draft_len + 1, dtype=jnp.int32)[:, None]
    t = jnp.arange(draft_len, dtype=jnp.int32)[None, :]
    offsets = (r + t).reshape(-1)
    pos_predraft = offsets
    return jnp.concatenate([pos_verify, pos_predraft], axis=0)


@lru_cache(maxsize=None)
def build_decode_bias_template(cache_len: int, draft_len: int, bias_value: float) -> jnp.ndarray:
    """
    Constant TiDAR decode bias for KV-cache mode:
      queries are step tokens (q_len = K + K^2),
      keys are [prefix-cache of length cache_len] + [step tokens of length q_len].

    Dynamic "prefix_len validity" masking is applied inside attention (your model code),
    so this template can treat all cache positions as allowed for step->prefix and rely
    on prefix-valid masking to block inactive cache slots.
    """
    q_len = draft_len + (draft_len * draft_len)
    key_len = cache_len + q_len

    q_idx = jnp.arange(q_len)[:, None]          # [q_len, 1]
    k_idx = jnp.arange(key_len)[None, :]        # [1, key_len]

    is_verify_q = q_idx < draft_len
    is_cand_q = q_idx >= draft_len

    is_prefix_k = k_idx < cache_len
    is_step_k = k_idx >= cache_len
    step_k_idx = k_idx - cache_len

    is_verify_k = is_step_k & (step_k_idx < draft_len)
    is_cand_k = is_step_k & (step_k_idx >= draft_len)

    # Verify queries:
    allow_verify_to_prefix = is_verify_q & is_prefix_k
    allow_verify_to_verify = is_verify_q & is_verify_k & (step_k_idx <= q_idx)

    # Candidate queries:
    cand_q_offset = q_idx - draft_len
    cand_k_offset = step_k_idx - draft_len

    cand_q_block = cand_q_offset // draft_len   # [0..K-1]
    cand_k_block = cand_k_offset // draft_len   # [0..K-1]
    cand_r = cand_q_block + 1                   # r in [1..K]

    allow_cand_to_prefix = is_cand_q & is_prefix_k
    allow_cand_to_verify = is_cand_q & is_verify_k & (step_k_idx < cand_r)
    allow_cand_to_cand = is_cand_q & is_cand_k & (cand_q_block == cand_k_block)

    allow = (
        allow_verify_to_prefix
        | allow_verify_to_verify
        | allow_cand_to_prefix
        | allow_cand_to_verify
        | allow_cand_to_cand
    )

    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]  # [1,1,q_len,key_len]


# =========================
# JAX sampling utilities (JIT-friendly)
# =========================
def _mask_top_k_2d(logits_2d: jnp.ndarray, top_k: int) -> jnp.ndarray:
    """logits_2d: [N, V]. top_k must be static (Python int) when jitted."""
    if top_k <= 0:
        return logits_2d
    top_vals, top_idx = jax.lax.top_k(logits_2d, top_k)  # [N,K], [N,K]
    masked = jnp.full_like(logits_2d, -jnp.inf)
    n = logits_2d.shape[0]
    rows = jnp.arange(n, dtype=jnp.int32)[:, None]
    masked = masked.at[rows, top_idx].set(top_vals)
    return masked


def _prepare_logits_2d(logits_2d: jnp.ndarray, *, temperature: float, top_k: int) -> jnp.ndarray:
    # temperature <= 0 handled by caller (argmax path)
    scaled = logits_2d / jnp.maximum(jnp.asarray(temperature, dtype=logits_2d.dtype), 1e-6)
    scaled = _mask_top_k_2d(scaled, top_k)
    return scaled


def sample_tokens(
    key: jax.Array,
    logits: jnp.ndarray,
    *,
    temperature: float,
    top_k: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    """
    Sample tokens from logits with optional temperature/top_k.

    logits: [..., V]
    returns tokens: [...]
    """
    v = logits.shape[-1]
    flat = logits.reshape((-1, v))  # [N, V]

    def do_sample(k):
        prepared = _prepare_logits_2d(flat, temperature=temperature, top_k=top_k)
        # One key is enough: categorical is vectorized across batch
        toks = jax.random.categorical(k, prepared, axis=-1).astype(jnp.int32)  # [N]
        return toks

    def do_argmax(_k):
        return jnp.argmax(flat, axis=-1).astype(jnp.int32)

    toks_flat = jax.lax.cond(jnp.asarray(temperature) > 0.0, do_sample, do_argmax, key)
    return key, toks_flat.reshape(logits.shape[:-1])


def rejection_sample_jax(
    key: jax.Array,
    *,
    draft_ids: jnp.ndarray,        # [K]
    verify_logits: jnp.ndarray,    # [K, V]
    draft_logits: jnp.ndarray,     # [K, V]
    temperature: float,
    top_k: int,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    TiDAR rejection sampling in JAX:
      - sequentially accept draft token i with prob min(1, p_i(tok)/q_i(tok))
      - if rejected at i: sample new token from p_i and stop
    Returns:
      key, r (int32 scalar), committed_full ([K] int32)
    """
    k = draft_ids.shape[0]

    # Prepare p and q logits consistently with sampling distribution
    p_logits = _prepare_logits_2d(verify_logits, temperature=temperature, top_k=top_k)  # [K,V]
    q_logits = _prepare_logits_2d(draft_logits, temperature=temperature, top_k=top_k)  # [K,V]

    # log-probs for ratio
    p_log = jax.nn.log_softmax(p_logits, axis=-1)
    q_log = jax.nn.log_softmax(q_logits, axis=-1)

    idx = jnp.arange(k, dtype=jnp.int32)
    p_log_tok = p_log[idx, draft_ids]
    q_log_tok = q_log[idx, draft_ids]
    ratio = jnp.exp(p_log_tok - q_log_tok)
    accept_prob = jnp.minimum(1.0, ratio).astype(jnp.float32)  # [K]

    key_u, key_resample = jax.random.split(key, 2)
    u = jax.random.uniform(key_u, (k,), dtype=jnp.float32)
    # Pre-sample replacement tokens from p for each position (only used at first rejection)
    resampled = jax.random.categorical(key_resample, p_logits, axis=-1).astype(jnp.int32)  # [K]

    def step(carry, x):
        stopped, r = carry
        draft_tok, repl_tok, u_i, ap_i = x

        accept = u_i < ap_i
        do_accept = (~stopped) & accept
        do_reject = (~stopped) & (~accept)

        out_tok = jnp.where(
            stopped,
            draft_tok,  # ignored beyond r
            jnp.where(accept, draft_tok, repl_tok),
        )
        stopped2 = stopped | do_reject
        r2 = r + (~stopped).astype(jnp.int32)
        return (stopped2, r2), out_tok

    (stopped_fin, r_fin), committed = jax.lax.scan(
        step,
        init=(jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32)),
        xs=(draft_ids, resampled, u, accept_prob),
        length=k,
    )
    # r_fin is guaranteed >= 1 for k>0
    return key, r_fin, committed


# =========================
# JITted TiDAR generation (NO python loops)
# =========================
def make_tidar_generate_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    pad_token_id: int,
    eos_id: int,
    stop_on_eos: bool,
    always_accept: bool,
    temperature: float,
    top_k: int,
    bias_value: float,
):
    """
    Returns a single jitted function that:
      - takes an already-prefilled KV cache (prompt committed)
      - takes an output buffer aligned with cache positions (length cache_len)
      - prefill-samples initial draft (K tokens) in JAX
      - runs TiDAR decode using lax.while_loop (no Python loops)
      - commits to KV cache using fixed-length K writes per iteration
        (prefix_len increases by r, and prefix-valid attention masking blocks junk cache slots)
    """
    q_len = draft_len + (draft_len * draft_len)
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))

    # Prefill bias in KV-cache mode: only mask queries exist, so allow everything.
    # Shape must match [1,1,K, cache_len + K]
    prefill_bias = jnp.zeros((1, 1, draft_len, cache_len + draft_len), dtype=decode_bias.dtype)
    prefill_bias = jax.device_put(prefill_bias)

    step_position_offsets = build_decode_position_ids_template(draft_len)
    step_position_offsets = jax.device_put(step_position_offsets)

    predraft_masks = jnp.full((draft_len * draft_len,), jnp.asarray(mask_id, dtype=jnp.int32), dtype=jnp.int32)
    predraft_masks = jax.device_put(predraft_masks)

    idx_k = jnp.arange(draft_len, dtype=jnp.int32)
    idx_k = jax.device_put(idx_k)

    def decode_apply(params, cache_vars, step_tokens, step_pos_ids, prefix_len):
        # step_tokens: [1, q_len]
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=step_pos_ids,
            kv_cache_len=cache_len,
        )
        return logits

    def prefill_apply(params, cache_vars, mask_tokens, mask_pos_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            mask_tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=prefill_bias,
            position_ids=mask_pos_ids,
            kv_cache_len=cache_len,
        )
        return logits

    def write_cache_apply(params, cache_vars, tokens_k, pos_ids_k, cur_index):
        # tokens_k: [1,K]
        _, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens_k,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=cur_index,
            position_ids=pos_ids_k,
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return mutated["cache"]

    def next_logit_apply(params, cache_vars, token_1, cur_index):
        # token_1: [1,1]
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            token_1,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=cur_index,
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0, 0], mutated["cache"]

    @jax.jit
    def generate(
        params,
        cache_vars,
        out_ids,
        prefix_len_init: jnp.ndarray,
        max_steps: jnp.ndarray,
        prev_logit_init: jnp.ndarray,
        rng_key: jax.Array,
    ):
        """
        params: model params pytree
        cache_vars: KV cache pytree (already contains prompt)
        out_ids: [cache_len] int32 buffer, with prompt already written at [0:prefix_len_init]
        prefix_len_init: scalar int32 (prompt length)
        max_steps: scalar int32 (#new tokens to generate)
        prev_logit_init: [V] logit for token after prompt
        rng_key: PRNGKey
        Returns: out_ids, final_prefix_len, generated_count
        """
        prefix_len = prefix_len_init.astype(jnp.int32)
        prev_logit = prev_logit_init
        generated = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        # ---- Prefill draft: run K mask tokens and sample draft tokens
        mask_tokens = jnp.full((1, draft_len), jnp.asarray(mask_id, dtype=jnp.int32), dtype=jnp.int32)
        mask_pos_ids = (prefix_len + jnp.arange(draft_len, dtype=jnp.int32))[None, :]
        prefill_logits = prefill_apply(params, cache_vars, mask_tokens, mask_pos_ids, prefix_len)[0]  # [K,V]

        rng_key, sub = jax.random.split(rng_key)
        _, draft_tokens = sample_tokens(sub, prefill_logits, temperature=temperature, top_k=top_k)  # [K]
        draft_tokens = draft_tokens.astype(jnp.int32)
        draft_logits = prefill_logits  # q for the first rejection step

        # Optional: write initial draft tokens into out_ids? (Not committed yet) -> NO.
        # They are proposals only.

        def cond_fn(state):
            _rng, _cache, _out, _prefix_len, _generated, _draft_toks, _draft_logits, _prev_logit, _done = state
            return (_generated < max_steps) & (~_done)

        def body_fn(state):
            rng, cache, out, prefix_len, generated, draft_toks, draft_lgts, prev_logit, done = state

            # Step tokens = [VERIFY(K)=draft_toks | PREDRAFT(K*K)=mask]
            step_tokens = jnp.concatenate([draft_toks, predraft_masks], axis=0).astype(jnp.int32)  # [q_len]
            step_pos_ids = (prefix_len + step_position_offsets)[None, :]                          # [1,q_len]

            logits = decode_apply(
                params,
                cache,
                step_tokens[None, :],
                step_pos_ids.astype(jnp.int32),
                prefix_len,
            )[0]  # [q_len, V]

            verify_logits_raw = logits[:draft_len, :]  # [K,V]
            verify_logits = jnp.concatenate(
                [prev_logit[None, :], verify_logits_raw[:-1, :]], axis=0
            )

            if always_accept:
                r = jnp.asarray(draft_len, dtype=jnp.int32)
                committed_full = draft_toks

                # Sample only the last candidate block (r = K)
                rng, sub_cand = jax.random.split(rng)
                start = draft_len + (draft_len - 1) * draft_len
                end = draft_len + draft_len * draft_len
                cand_logits_last = logits[start:end, :].reshape((draft_len, -1))
                _, draft_toks2 = sample_tokens(sub_cand, cand_logits_last, temperature=temperature, top_k=top_k)
                draft_toks2 = draft_toks2.astype(jnp.int32)
                draft_lgts2 = cand_logits_last
            else:
                cand_logits = logits[draft_len:, :].reshape((draft_len, draft_len, -1))  # [K,K,V]

                # Sample candidates (vectorized): sample for all K*K positions in one call
                rng, sub_cand = jax.random.split(rng)
                flat_cand = cand_logits.reshape((-1, cand_logits.shape[-1]))  # [K*K,V]
                _, flat_cand_tokens = sample_tokens(sub_cand, flat_cand, temperature=temperature, top_k=top_k)  # [K*K]
                cand_tokens = flat_cand_tokens.reshape((draft_len, draft_len)).astype(jnp.int32)  # [K,K]

                # Rejection sampling to choose r and committed tokens (in JAX)
                rng, r, committed_full = rejection_sample_jax(
                    rng,
                    draft_ids=draft_toks,
                    verify_logits=verify_logits,
                    draft_logits=draft_lgts,
                    temperature=temperature,
                    top_k=top_k,
                )

                # Next draft comes from candidate block (r-1), not eff_r
                block_idx = jnp.clip(r - 1, 0, draft_len - 1)
                draft_toks2 = cand_tokens[block_idx]          # [K]
                draft_lgts2 = cand_logits[block_idx]          # [K,V]

            remaining = (max_steps - generated).astype(jnp.int32)
            eff_r = jnp.minimum(r, remaining)

            # EOS early stop inside committed tokens
            has_eos = jnp.asarray(False)
            if stop_on_eos and eos_id >= 0:
                eos = jnp.asarray(eos_id, dtype=jnp.int32)
                pos = idx_k
                eos_mask = (committed_full == eos) & (pos < eff_r)
                first_eos = jnp.min(jnp.where(eos_mask, pos, jnp.asarray(draft_len, dtype=jnp.int32)))
                has_eos = first_eos < draft_len
                eff_r = jnp.where(has_eos, jnp.minimum(eff_r, first_eos + 1), eff_r)

            # Build fixed-length commit tokens (K) so cache write has static shape
            commit_fixed = jnp.where(idx_k < eff_r, committed_full, jnp.asarray(pad_token_id, dtype=jnp.int32))
            commit_fixed = commit_fixed.astype(jnp.int32)

            # Write to cache at current prefix_len with fixed K tokens
            # NOTE: requires prefix_len + K <= cache_len always (we enforce via bucket selection + slack).
            pos_ids_k = (prefix_len + idx_k)[None, :]  # [1,K]
            cache = write_cache_apply(params, cache, commit_fixed[None, :], pos_ids_k.astype(jnp.int32), prefix_len)

            # Update output buffer aligned with cache positions
            out = jax.lax.dynamic_update_slice(out, commit_fixed, (prefix_len,))

            # Advance prefix_len by eff_r (NOT by K)
            prefix_len2 = prefix_len + eff_r
            generated2 = generated + eff_r

            def _update_prev(args):
                cache_in, eff_r_in, prefix_len_in, committed_in = args
                last_tok = jnp.take(committed_in, eff_r_in - 1)
                last_index = prefix_len_in - 1
                token_1 = last_tok[None, None]
                next_logit, cache_next = next_logit_apply(params, cache_in, token_1, last_index)
                return cache_next, next_logit

            def _keep_prev(args):
                cache_in, _eff_r_in, _prefix_len_in, _committed_in = args
                return cache_in, prev_logit

            cache, prev_logit2 = jax.lax.cond(
                eff_r > 0,
                _update_prev,
                _keep_prev,
                (cache, eff_r, prefix_len2, committed_full),
            )

            done2 = done | (generated2 >= max_steps) | has_eos

            return (
                rng,
                cache,
                out,
                prefix_len2,
                generated2,
                draft_toks2,
                draft_lgts2,
                prev_logit2,
                done2,
            )

        state0 = (
            rng_key,
            cache_vars,
            out_ids,
            prefix_len,
            generated,
            draft_tokens,
            draft_logits,
            prev_logit,
            done,
        )
        stateF = jax.lax.while_loop(cond_fn, body_fn, state0)

        rngF, cacheF, outF, prefix_lenF, generatedF, _draft_toksF, _draft_logitsF, _prev_logitF, doneF = stateF
        return outF, prefix_lenF, generatedF

    return generate


# =========================
# CLI
# =========================
def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TiDAR inference (KV cache, jitted decode)")
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=_parse_bool, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    parser.add_argument("--always_accept", action="store_true")
    parser.add_argument("--cache_buckets", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()
    cfg = load_configs()

    # Keep your existing config update (even if name is a bit odd)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)

    stop_on_eos = args.stop_on_eos if args.stop_on_eos is not None else bool(cfg.inference.stop_on_eos)

    model_context_length = int(cfg.model.context_length)
    context_length = args.context_length if args.context_length is not None else model_context_length

    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if draft_len <= 0:
        raise ValueError("draft_len must be > 0")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")
    if context_length > model_context_length:
        raise ValueError(f"context_length {context_length} exceeds model context_length {model_context_length}.")

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens. Provide non-empty text.")

    prompt_len = int(prompt_ids.shape[0])

    # IMPORTANT CHANGE:
    # We write K tokens into cache every TiDAR iteration (fixed-shape write),
    # even though prefix_len advances by r. This requires extra slack so that
    # prefix_len+K never exceeds cache_len during the run.
    required_cache_len = prompt_len + max_steps + draft_len
    if required_cache_len > context_length:
        raise ValueError(
            f"prompt_len + max_steps + draft_len ({required_cache_len}) exceeds context_length {context_length}."
        )

    cache_buckets = parse_cache_buckets(args.cache_buckets, context_length=context_length)
    if not cache_buckets:
        raise ValueError("No valid cache buckets available within context length.")
    cache_len = select_cache_bucket(required_cache_len, cache_buckets)
    print(f"Using cache bucket: {cache_len}")

    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"[mask] added token '{mask_token}' (id={mask_id})")
    else:
        print(f"[mask] using existing token '{mask_token}' (id={mask_id})")

    model = build_model(cfg, len(tokenizer), context_length)
    params = load_params(checkpoint_path)

    rng = jax.random.PRNGKey(args.seed)
    rng, resize_key = jax.random.split(rng)
    params, added_rows = resize_embedding_params(params, len(tokenizer), key=resize_key)
    if added_rows:
        print(f"[checkpoint] expanded embeddings by {added_rows} rows for TiDAR mask token")

    params = jax.device_put(params)

    prompt_ids_jax = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)
    prompt_logits = model.apply({"params": params}, prompt_ids_jax, deterministic=True)
    prev_logit = jax.device_put(prompt_logits[0, -1])

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eos_id = tokenizer.eos_token_id
    eos_id_for_jit = int(eos_id) if eos_id is not None else -1

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)

    # Prefill committed prefix (prompt) into KV cache
    cache_vars, cache_index = prefill_prompt_cache(
        model,
        params,
        cache_vars,
        prompt_ids,
        kv_cache_len=cache_len,
    )

    # Output buffer aligned with cache positions (length = cache_len).
    # IMPORTANT: do this on host to avoid dynamic slice sizes inside jit.
    out_host = np.full((required_cache_len,), pad_token_id, dtype=np.int32)
    out_host[:prompt_len] = prompt_ids
    out_ids = jax.device_put(jnp.asarray(out_host))

    # Build a single jitted generator (NO python decode loop)
    generate_fn = make_tidar_generate_fn(
        model,
        cache_len=cache_len,
        draft_len=draft_len,
        mask_id=int(mask_id),
        pad_token_id=int(pad_token_id),
        eos_id=eos_id_for_jit,
        stop_on_eos=bool(stop_on_eos),
        always_accept=bool(args.always_accept),
        temperature=float(temperature),
        top_k=int(top_k),  # static for top_k masking/top_k op
        bias_value=float(bias_value),
    )

    # Run generation
    decode_start = time.perf_counter()
    out_ids_f, final_len, generated = generate_fn(
        params,
        cache_vars,
        out_ids,
        jnp.asarray(cache_index, dtype=jnp.int32),
        jnp.asarray(max_steps, dtype=jnp.int32),
        prev_logit,
        rng,
    )
    out_ids_f.block_until_ready()
    decode_time = time.perf_counter() - decode_start

    final_len = int(np.asarray(final_len))
    out_tokens = np.asarray(out_ids_f[:final_len], dtype=np.int32)

    # Stop-on-eos pretty printing (match your previous behavior)
    text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    if stop_on_eos and eos_id is not None:
        eos_hits = np.where(out_tokens == eos_id)[0]
        if eos_hits.size > 0:
            cut = int(eos_hits[0])
            text = tokenizer.decode(out_tokens[:cut], skip_special_tokens=True) + "<EOS>"

    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")

    if args.verbose:
        gen_count = int(np.asarray(generated))
        toks_per_s = (gen_count / decode_time) if decode_time > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {prompt_len}")
        print(f"generated_tokens: {gen_count}")
        print(f"decode_time_s:  {decode_time:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()

