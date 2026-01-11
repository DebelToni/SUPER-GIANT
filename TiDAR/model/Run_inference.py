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
from v2.model.checkpoint_manager import load_npz, latest as latest_ckpt

DEFAULT_CACHE_BUCKETS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-9)


def _apply_temperature_top_k(logits: np.ndarray, *, temperature: float, top_k: int) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    if top_k > 0:
        top_indices = np.argpartition(-scaled, top_k - 1, axis=-1)[..., :top_k]
        masked = np.full_like(scaled, -np.inf)
        np.put_along_axis(masked, top_indices, np.take_along_axis(scaled, top_indices, axis=-1), axis=-1)
        scaled = masked
    return scaled


def sample_from_logits(
    logits: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    if temperature <= 0:
        return np.argmax(logits, axis=-1)
    scaled = _apply_temperature_top_k(logits, temperature=temperature, top_k=top_k)
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
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[int, np.ndarray]:
    verify_ids = np.asarray(verify_ids)
    verify_logits = _apply_temperature_top_k(verify_logits, temperature=temperature, top_k=top_k)
    k = verify_ids.shape[0]
    p = _softmax(verify_logits)

    if draft_logits is None:
        q = p
    else:
        q = _softmax(_apply_temperature_top_k(draft_logits, temperature=temperature, top_k=top_k))

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


def build_tidar_prefill_bias(
    *,
    prompt_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
) -> jnp.ndarray:
    total = draft_len + prompt_len
    idx = jnp.arange(total)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    is_mask_q = q_idx < draft_len
    is_mask_k = k_idx < draft_len
    is_prompt_q = q_idx >= draft_len
    is_prompt_k = k_idx >= draft_len

    allow_mask_to_mask = is_mask_q & is_mask_k
    allow_mask_to_prompt = is_mask_q & is_prompt_k

    prompt_q_pos = q_idx - draft_len
    prompt_k_pos = k_idx - draft_len
    allow_prompt_to_prompt = is_prompt_q & is_prompt_k & (prompt_k_pos <= prompt_q_pos)

    allow = allow_mask_to_mask | allow_mask_to_prompt | allow_prompt_to_prompt
    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


def build_tidar_decode_bias(
    *,
    prefix_len: int,
    draft_len: int,
    bias_value: float = -1.0e10,
) -> jnp.ndarray:
    total = prefix_len + draft_len + (draft_len * draft_len)
    idx = jnp.arange(total)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    prefix_end = prefix_len
    verify_start = prefix_end
    verify_end = verify_start + draft_len
    cand_start = verify_end

    is_prefix_q = q_idx < prefix_end
    is_prefix_k = k_idx < prefix_end
    is_verify_q = (q_idx >= verify_start) & (q_idx < verify_end)
    is_verify_k = (k_idx >= verify_start) & (k_idx < verify_end)
    is_cand_q = q_idx >= cand_start
    is_cand_k = k_idx >= cand_start

    prefix_q_pos = q_idx
    prefix_k_pos = k_idx
    allow_prefix = is_prefix_q & is_prefix_k & (prefix_k_pos <= prefix_q_pos)

    verify_q_pos = q_idx - verify_start
    verify_k_pos = k_idx - verify_start
    allow_verify_to_prefix = is_verify_q & is_prefix_k
    allow_verify_to_verify = is_verify_q & is_verify_k & (verify_k_pos <= verify_q_pos)

    cand_q_offset = q_idx - cand_start
    cand_k_offset = k_idx - cand_start
    cand_q_block = cand_q_offset // draft_len
    cand_k_block = cand_k_offset // draft_len

    cand_r = cand_q_block + 1
    allow_cand_to_prefix = is_cand_q & is_prefix_k
    allow_cand_to_verify = is_cand_q & is_verify_k & (verify_k_pos < cand_r)
    allow_cand_to_cand = is_cand_q & is_cand_k & (cand_q_block == cand_k_block)

    allow = (
        allow_prefix
        | allow_verify_to_prefix
        | allow_verify_to_verify
        | allow_cand_to_prefix
        | allow_cand_to_verify
        | allow_cand_to_cand
    )

    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :]


@lru_cache(maxsize=None)
def build_prefill_bias_template(max_seq_len: int, draft_len: int, bias_value: float) -> jnp.ndarray:
    # --- TiDAR-specific: Fig.7 prefill mask, cached and sliced per request.
    return build_tidar_prefill_bias(prompt_len=max_seq_len, draft_len=draft_len, bias_value=bias_value)


@lru_cache(maxsize=None)
def build_decode_bias_template(cache_len: int, draft_len: int, bias_value: float) -> jnp.ndarray:
    # --- TiDAR-specific: constant draft->draft + draft->prefix template.
    step_len = draft_len + (draft_len * draft_len)
    key_len = cache_len + step_len

    q_idx = jnp.arange(step_len)[:, None]
    k_idx = jnp.arange(key_len)[None, :]

    is_verify_q = q_idx < draft_len
    is_cand_q = q_idx >= draft_len
    is_prefix_k = k_idx < cache_len
    is_step_k = k_idx >= cache_len

    step_k_idx = k_idx - cache_len
    is_verify_k = is_step_k & (step_k_idx < draft_len)
    is_cand_k = is_step_k & (step_k_idx >= draft_len)

    allow_verify_to_prefix = is_verify_q & is_prefix_k
    allow_verify_to_verify = is_verify_q & is_verify_k & (step_k_idx <= q_idx)

    cand_q_offset = q_idx - draft_len
    cand_k_offset = step_k_idx - draft_len
    cand_q_block = cand_q_offset // draft_len
    cand_k_block = cand_k_offset // draft_len
    cand_r = cand_q_block + 1

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
    return bias[None, None, :, :]


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


def make_prefill_draft_fn(
    model: TiDAR,
    *,
    cache_len: int,
    prefill_bias: jnp.ndarray,
):
    prefill_bias = jax.device_put(prefill_bias)

    @jax.jit
    def prefill_draft(params, cache_vars, tokens, position_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=prefill_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )
        return logits

    return prefill_draft


def make_decode_step_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    bias_value: float,
):
    decode_bias = build_decode_bias_template(cache_len, draft_len, bias_value)
    decode_bias = jax.device_put(decode_bias)

    @jax.jit
    def decode_step(params, cache_vars, tokens, position_ids, prefix_len):
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=position_ids,
            kv_cache_len=cache_len,
        )
        return logits

    return decode_step


def commit_tokens_to_cache(
    model: TiDAR,
    params,
    cache_vars,
    *,
    committed: np.ndarray,
    prefix_len: int,
    kv_cache_len: int,
) -> Tuple[object, int]:
    committed = np.asarray(committed, dtype=np.int32)
    if committed.size == 0:
        return cache_vars, prefix_len

    tokens = committed[None, :]
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
        kv_cache_len=kv_cache_len,
        mutable=["cache"],
    )
    return mutated["cache"], prefix_len + step_len


def build_prefill_position_ids(prefix_len: int, draft_len: int) -> np.ndarray:
    """Positions for the prefill K mask tokens."""
    return np.arange(prefix_len, prefix_len + draft_len, dtype=np.int32)


def build_decode_position_ids(prefix_len: int, draft_len: int) -> np.ndarray:
    """Positions for TiDAR decode layout [VERIFY | PREDRAFT]."""
    pos_verify = np.arange(prefix_len, prefix_len + draft_len, dtype=np.int32)
    offsets = np.arange(1, draft_len + 1, dtype=np.int32)[:, None] + np.arange(
        draft_len, dtype=np.int32
    )[None, :]
    pos_predraft = prefix_len + offsets.reshape(-1)
    return np.concatenate([pos_verify, pos_predraft], axis=0)


def sanity_check_cached_decode(
    *,
    model: TiDAR,
    params,
    cache_vars,
    cache_len: int,
    prefix_ids: np.ndarray,
    draft_tokens: np.ndarray,
    mask_id: int,
    draft_len: int,
    bias_value: float,
) -> None:
    prefix_len = int(prefix_ids.shape[0])
    predraft_masks = np.full((draft_len * draft_len,), mask_id, dtype=np.int32)
    step_tokens = np.concatenate([draft_tokens, predraft_masks], axis=0)
    step_len = step_tokens.shape[0]
    position_ids = build_decode_position_ids(prefix_len, draft_len)
    if position_ids.shape[0] != step_len:
        raise ValueError(
            f"Decode position ids length {position_ids.shape[0]} != step_len {step_len}"
        )

    decode_fn = make_decode_step_fn(
        model,
        cache_len=cache_len,
        draft_len=draft_len,
        bias_value=bias_value,
    )
    cached_logits = decode_fn(
        params,
        cache_vars,
        jnp.asarray(step_tokens[None, :], dtype=jnp.int32),
        jnp.asarray(position_ids[None, :], dtype=jnp.int32),
        jnp.asarray(prefix_len, dtype=jnp.int32),
    )
    cached_logits = np.asarray(cached_logits[0])

    full_tokens = np.concatenate([prefix_ids, step_tokens], axis=0)
    full_position_ids = np.concatenate(
        [np.arange(prefix_len, dtype=np.int32), position_ids], axis=0
    )
    full_bias = build_tidar_decode_bias(
        prefix_len=prefix_len,
        draft_len=draft_len,
        bias_value=bias_value,
    )
    full_logits = model.apply(
        {"params": params},
        jnp.asarray(full_tokens[None, :], dtype=jnp.int32),
        deterministic=True,
        attn_bias=full_bias,
        position_ids=jnp.asarray(full_position_ids[None, :], dtype=jnp.int32),
    )
    full_logits = np.asarray(full_logits[0, prefix_len:])

    max_diff = float(np.max(np.abs(full_logits - cached_logits)))
    print(f"[sanity] cached vs full max diff: {max_diff:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TiDAR inference (KV cache)")
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    parser.add_argument("--always_accept", action="store_true")
    parser.add_argument("--cache_buckets", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)

    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if draft_len <= 0:
        raise ValueError("draft_len must be > 0")

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    context_length = int(cfg.model.context_length)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens. Provide non-empty text.")

    prompt_len = int(prompt_ids.shape[0])
    cache_buckets = parse_cache_buckets(args.cache_buckets, context_length=context_length)
    if not cache_buckets:
        raise ValueError("No valid cache buckets available within context length.")
    required_cache_len = prompt_len + max_steps
    if required_cache_len > context_length:
        raise ValueError(
            f"prompt_len + max_steps ({required_cache_len}) exceeds context_length {context_length}."
        )
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

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)

    # --- TiDAR-specific: prefill cache with committed prompt tokens only.
    cache_vars, cache_index = prefill_prompt_cache(
        model,
        params,
        cache_vars,
        prompt_ids,
        kv_cache_len=cache_len,
    )

    np_rng = np.random.default_rng(args.seed)

    # --- TiDAR-specific: prefill draft using cached prefix + precomputed Fig.7 mask.
    prefill_template = build_prefill_bias_template(context_length, draft_len, bias_value)
    prefill_slice = prefill_template[:, :, :draft_len, : (draft_len + prompt_len)]
    if cache_len > prompt_len:
        pad_width = cache_len - prompt_len
        pad = jnp.zeros((1, 1, draft_len, pad_width), dtype=prefill_slice.dtype)
        prefill_bias = jnp.concatenate([prefill_slice, pad], axis=-1)
    else:
        prefill_bias = prefill_slice

    prefill_draft_fn = make_prefill_draft_fn(
        model,
        cache_len=cache_len,
        prefill_bias=prefill_bias,
    )

    prefill_masks = jnp.full((1, draft_len), mask_id, dtype=jnp.int32)
    prefill_position_ids = build_prefill_position_ids(cache_index, draft_len)

    prefill_start = time.perf_counter()
    prefill_logits = prefill_draft_fn(
        params,
        cache_vars,
        prefill_masks,
        jnp.asarray(prefill_position_ids[None, :], dtype=jnp.int32),
        jnp.asarray(cache_index, dtype=jnp.int32),
    )
    prefill_logits.block_until_ready()
    prefill_logits = np.asarray(prefill_logits[0])
    draft_tokens = sample_from_logits(
        prefill_logits,
        rng=np_rng,
        temperature=temperature,
        top_k=top_k,
    )
    draft_logits = prefill_logits
    prefill_time = time.perf_counter() - prefill_start

    prefix_ids = prompt_ids.astype(np.int32)
    generated = 0
    stop_on_eos = bool(cfg.inference.stop_on_eos)
    eos_id = tokenizer.eos_token_id

    if args.sanity_check:
        sanity_check_cached_decode(
            model=model,
            params=params,
            cache_vars=cache_vars,
            cache_len=cache_len,
            prefix_ids=prefix_ids,
            draft_tokens=draft_tokens,
            mask_id=mask_id,
            draft_len=draft_len,
            bias_value=bias_value,
        )

    decode_fn = make_decode_step_fn(
        model,
        cache_len=cache_len,
        draft_len=draft_len,
        bias_value=bias_value,
    )

    decode_start = time.perf_counter()
    while generated < max_steps:
        step_len = draft_len + (draft_len * draft_len)

        # --- TiDAR-specific: decode layout = verify | predraft masks (prefix in KV cache).
        predraft_masks = np.full((draft_len * draft_len,), mask_id, dtype=np.int32)
        step_tokens = np.concatenate([draft_tokens, predraft_masks], axis=0)
        step_position_ids = build_decode_position_ids(cache_index, draft_len)
        if step_position_ids.shape[0] != step_len:
            raise ValueError(
                f"Decode position ids length {step_position_ids.shape[0]} != step_len {step_len}"
            )

        step_logits = decode_fn(
            params,
            cache_vars,
            jnp.asarray(step_tokens[None, :], dtype=jnp.int32),
            jnp.asarray(step_position_ids[None, :], dtype=jnp.int32),
            jnp.asarray(cache_index, dtype=jnp.int32),
        )
        step_logits = np.asarray(step_logits[0])

        verify_logits = step_logits[:draft_len]
        cand_logits = step_logits[draft_len:].reshape(draft_len, draft_len, -1)

        candidate_tokens = np.zeros((draft_len, draft_len), dtype=np.int32)
        for block_idx in range(draft_len):
            candidate_tokens[block_idx] = sample_from_logits(
                cand_logits[block_idx],
                rng=np_rng,
                temperature=temperature,
                top_k=top_k,
            )

        # --- TiDAR-specific: rejection sampling chooses commit length r + next candidate block.
        if args.always_accept:
            r = draft_len
            committed = draft_tokens
        else:
            r, committed = rejection_sample(
                draft_tokens,
                verify_logits,
                rng=np_rng,
                draft_logits=draft_logits,
                temperature=temperature,
                top_k=top_k,
            )

        remaining = max_steps - generated
        if committed.shape[0] > remaining:
            committed = committed[:remaining]

        cache_vars, cache_index = commit_tokens_to_cache(
            model,
            params,
            cache_vars,
            committed=committed,
            prefix_len=cache_index,
            kv_cache_len=cache_len,
        )

        prefix_ids = np.concatenate([prefix_ids, committed], axis=0)
        generated += int(committed.shape[0])
        if generated >= max_steps:
            break

        if stop_on_eos and eos_id is not None:
            eos_hits = np.where(committed == eos_id)[0]
            if eos_hits.size > 0:
                cut = int(eos_hits[0]) + 1
                prefix_ids = prefix_ids[: -(committed.shape[0] - cut)]
                break

        block_idx = min(max(r - 1, 0), draft_len - 1)
        draft_tokens = candidate_tokens[block_idx]
        draft_logits = cand_logits[block_idx]

    decode_time = time.perf_counter() - decode_start

    text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    if stop_on_eos and eos_id is not None:
        eos_hits = np.where(prefix_ids == eos_id)[0]
        if eos_hits.size > 0:
            cut = int(eos_hits[0])
            text = tokenizer.decode(prefix_ids[:cut], skip_special_tokens=True) + "<EOS>"

    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")

    if args.verbose:
        toks_per_s = (generated / decode_time) if decode_time > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {prompt_ids.shape[0]}")
        print(f"generated_tokens: {generated}")
        print(f"prefill_time_s: {prefill_time:.6f}")
        print(f"decode_time_s:  {decode_time:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")


if __name__ == "__main__":
    main()
