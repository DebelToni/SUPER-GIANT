from __future__ import annotations

import sys

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.GiantGPT import GiantGPT
from model.kv_cache_buckets import select_tidar_kv_bucket
from model.checkpoint_manager import latest as latest_ckpt
from model.checkpoint_manager import load as load_ckpt
from model.tidar_inference import (
    init_kv_cache,
    prefill_prompt_cache,
    tidar_prefill_draft_cached,
    tidar_decode_step_cached,
)
from model.tidar_masks import build_tidar_decode_bias_cached
from model.tidar_utils import sample_from_logits, jax_sample
from model.tokenizer_utils import ensure_tidar_mask_token, resize_embedding_params


def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(model_dir / "Config.yml"),
    )

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str(base_prefix / path)

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


def load_tokenizer(cfg: OmegaConf):
    from transformers import AutoTokenizer

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


def ensure_mask_id(tokenizer, cfg: OmegaConf) -> int:
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added <= 0:
        print(f"[mask] using existing token '{mask_token}' (id={mask_id})")
    else:
        print(f"[mask] added token '{mask_token}' (id={mask_id})")
    return mask_id


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("TiDAR inference")
    cli.add_argument("--prompt", type=str, default="Once upon a time")
    cli.add_argument("--checkpoint", type=str, default=None)
    cli.add_argument("--max_steps", type=int, default=None)
    cli.add_argument("--num_new_tokens", type=int, default=None)
    cli.add_argument("--temperature", type=float, default=None)
    cli.add_argument("--top_k", type=int, default=None)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--draft_len", type=int, default=None)
    cli.add_argument("--always_accept", action="store_true")
    cli.add_argument("--benchmark", action="store_true")
    cli.add_argument("--quiet", action="store_true")
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    tokenizer = load_tokenizer(cfg)

    mask_id = ensure_mask_id(tokenizer, cfg)
    draft_len = int(cfg.tidar.draft_length) if args.draft_len is None else int(args.draft_len)
    context_len = int(cfg.model.context_length)
    max_steps = args.max_steps or int(cfg.inference.max_decode_steps)
    num_new_tokens = args.num_new_tokens
    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)

    prompt_ids = tokenizer(args.prompt, return_tensors="np").input_ids[0].astype(np.int32)
    prompt_len = int(prompt_ids.shape[0])

    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=cfg.model.context_length,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
    )

    checkpoint_path: Optional[str] = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = latest_ckpt(cfg.paths.checkpoints_root)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(cfg.paths.data_root) / ckpt_path).resolve()
    params, _ = load_ckpt(str(ckpt_path))

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, resize_key = jax.random.split(rng_key)
    params, added = resize_embedding_params(params, len(tokenizer), key=resize_key)
    if added:
        print(f"[inference] expanded embeddings by {added} rows for TiDAR mask token")
    rng = np.random.default_rng(args.seed)

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=tokenizer.pad_token_id or 0)
    
    # Use bucketed KV cache allocation
    if num_new_tokens is not None:
        kv_cache_len, effective_window = select_tidar_kv_bucket(
            prompt_len, num_new_tokens, draft_len, context_len
        )
        if not args.quiet:
            print(f"[cache] selected bucket={kv_cache_len}, effective_window={effective_window}")
    else:
        # Fallback for unbounded generation
        decode_window = max_steps
        kv_cache_len = min(context_len, prompt_len + decode_window)
    
    prefill_kv_len = min(kv_cache_len, prompt_len)
    cache_vars, prefix_len = prefill_prompt_cache(
        model,
        params,
        cache_vars,
        prompt_ids,
        kv_cache_len=prefill_kv_len,
    )

    prefix = prompt_ids
    generated = 0

    if draft_len <= 0:
        use_scan = args.benchmark and args.quiet and num_new_tokens is not None
        use_device_sampling = args.benchmark and top_k == 0
        last_token = int(prefix[-1]) if prefix_len > 0 else int(mask_id)
        last_token_dev = jnp.array([[last_token]], dtype=jnp.int32)
        rng_key = jax.random.PRNGKey(args.seed)
        if use_scan:
            last_token_scan = jnp.array(last_token, dtype=jnp.int32)
            prefix_len_scan = jnp.array(prefix_len, dtype=jnp.int32)

            @partial(jax.jit, static_argnames=("model", "num_steps"))
            def scan_ar(model, params, cache_vars, prefix_len, last_token, key, num_steps):
                def step(carry, _):
                    cache_vars, prefix_len, last_token, key = carry
                    tokens = last_token[None, None]
                    pos_ids = prefix_len[None, None]
                    logits, mutated = model.apply(
                        {"params": params, "cache": cache_vars},
                        tokens,
                        deterministic=True,
                        use_kv_cache=True,
                        write_to_cache=True,
                        cur_index=prefix_len,
                        position_ids=pos_ids,
                        kv_cache_len=kv_cache_len,
                        mutable=["cache"],
                    )
                    cache_vars = mutated["cache"]
                    key, sub = jax.random.split(key)
                    next_token = jax_sample(
                        logits[0, -1],
                        key=sub,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    prefix_len = prefix_len + jnp.array(1, jnp.int32)
                    return (cache_vars, prefix_len, next_token, key), None

                (cache_vars, prefix_len, last_token, key), _ = jax.lax.scan(
                    step, (cache_vars, prefix_len, last_token, key), xs=None, length=num_steps
                )
                return cache_vars, prefix_len, last_token, key

            start_time = time.perf_counter()
            cache_vars, prefix_len_scan, last_token_scan, rng_key = scan_ar(
                model,
                params,
                cache_vars,
                prefix_len_scan,
                last_token_scan,
                rng_key,
                num_new_tokens,
            )
            jax.block_until_ready(prefix_len_scan)
            prefix_len = int(jax.device_get(prefix_len_scan))
            generated = num_new_tokens
        else:
            start_time = time.perf_counter()
            while num_new_tokens is None or generated < num_new_tokens:
                if num_new_tokens is not None and generated >= num_new_tokens:
                    break
                tokens = last_token_dev if use_device_sampling else np.array([[last_token]], dtype=np.int32)
                pos_ids = np.array([[prefix_len]], dtype=np.int32)
                logits, mutated = model.apply(
                    {"params": params, "cache": cache_vars},
                    jnp.asarray(tokens),
                    deterministic=True,
                    use_kv_cache=True,
                    write_to_cache=True,
                    cur_index=prefix_len,
                    position_ids=jnp.asarray(pos_ids),
                    kv_cache_len=kv_cache_len,
                    mutable=["cache"],
                )
                cache_vars = mutated["cache"]
                if use_device_sampling:
                    rng_key, sub = jax.random.split(rng_key)
                    step_logits = logits[:, -1, :] / jnp.maximum(temperature, 1e-6)
                    next_token_dev = jax.random.categorical(sub, step_logits, axis=-1)
                    last_token_dev = next_token_dev[:, None]
                    if not args.quiet:
                        next_token = int(jax.device_get(next_token_dev)[0])
                    else:
                        next_token = None
                else:
                    step_logits = np.asarray(logits)[0, -1, :]
                    next_token = sample_from_logits(
                        step_logits,
                        rng=rng,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    next_token = int(next_token)
                    last_token_dev = jnp.array([[next_token]], dtype=jnp.int32)
                if not args.quiet:
                    prefix = np.concatenate([prefix, np.array([next_token], dtype=np.int32)], axis=0)
                if next_token is not None:
                    last_token = next_token
                prefix_len += 1
                generated += 1
                if num_new_tokens is None and generated >= max_steps:
                    break
                if not args.benchmark and cfg.inference.stop_on_eos and tokenizer.eos_token_id is not None:
                    if next_token == tokenizer.eos_token_id:
                        break
    else:
        draft_ids, draft_logits, rng_key = tidar_prefill_draft_cached(
            model,
            params,
            cache_vars,
            prefix_len=prefix_len,
            context_len=context_len,
            batch_size=1,
            mask_id=mask_id,
            draft_len=draft_len,
            key=rng_key,
            temperature=temperature,
            top_k=top_k,
            kv_cache_len=kv_cache_len,
        )
        verify = draft_ids
        verify_logits = draft_logits
        use_scan = (
            args.benchmark
            and args.always_accept
            and args.quiet
            and num_new_tokens is not None
            and num_new_tokens % draft_len == 0
        )
        if use_scan:
            num_steps = num_new_tokens // draft_len
            verify_batched = verify[None, :] if verify.ndim == 1 else verify
            embed_matrix = jnp.asarray(params["Embed_0"]["embedding"])
            attn_bias = build_tidar_decode_bias_cached(
                context_len=kv_cache_len, draft_len=draft_len
            )
            verify_offsets = jnp.arange(draft_len, dtype=jnp.int32)
            predraft_offsets = []
            for r in range(1, draft_len + 1):
                predraft_offsets.extend(np.arange(r, r + draft_len, dtype=np.int32))
            predraft_offsets = jnp.asarray(predraft_offsets, dtype=jnp.int32)
            predraft_tokens = jnp.full((1, draft_len * draft_len), mask_id, dtype=jnp.int32)
            cand_start = draft_len + (draft_len - 1) * draft_len

            def _logits_from_hidden(hidden):
                return jnp.einsum("bld,vd->blv", hidden.astype(jnp.float32), embed_matrix)

            @partial(jax.jit, static_argnames=("model",))
            def scan_steps(model, params, cache_vars, prefix_len, verify_ids, key):
                def step(carry, _):
                    cache_vars, prefix_len, verify_ids, key = carry
                    step_tokens = jnp.concatenate([verify_ids, predraft_tokens], axis=1)
                    pos_verify = prefix_len + verify_offsets
                    pos_predraft = prefix_len + predraft_offsets
                    position_ids = jnp.concatenate([pos_verify, pos_predraft])[None, :]

                    hidden, mutated = model.apply(
                        {"params": params, "cache": cache_vars},
                        step_tokens,
                        deterministic=True,
                        use_kv_cache=True,
                        write_to_cache=False,
                        cache_write_len=draft_len,
                        prefix_len=prefix_len,
                        cur_index=prefix_len,
                        attn_bias=attn_bias,
                        position_ids=position_ids,
                        kv_cache_len=kv_cache_len,
                        return_hidden=True,
                        mutable=["cache"],
                    )
                    cache_vars = mutated["cache"]
                    hidden_cand = jax.lax.dynamic_slice(
                        hidden, (0, cand_start, 0), (1, draft_len, hidden.shape[-1])
                    )
                    logits_cand = _logits_from_hidden(hidden_cand)
                    key, sub = jax.random.split(key)
                    next_verify = jax_sample(
                        logits_cand[0], key=sub, temperature=temperature, top_k=top_k
                    )

                    prefix_len = prefix_len + draft_len
                    verify_ids = next_verify[None, :]
                    return (cache_vars, prefix_len, verify_ids, key), None

                (cache_vars, prefix_len, verify_ids, key), _ = jax.lax.scan(
                    step, (cache_vars, prefix_len, verify_ids, key), xs=None, length=num_steps
                )
                return cache_vars, prefix_len, verify_ids, key

            start_time = time.perf_counter()
            cache_vars, prefix_len, verify, rng_key = scan_steps(
                model, params, cache_vars, jnp.array(prefix_len, jnp.int32), verify_batched, rng_key
            )
            jax.block_until_ready(prefix_len)
            generated = num_new_tokens
        else:
            start_time = time.perf_counter()
            while num_new_tokens is None or generated < num_new_tokens:
                if num_new_tokens is not None:
                    remaining = num_new_tokens - generated
                    if remaining <= 0:
                        break
                    max_commit = remaining if args.always_accept else None
                else:
                    max_commit = None

                prefix, verify, verify_logits, r, cache_vars, prefix_len, rng_key = tidar_decode_step_cached(
                    model,
                    params,
                    cache_vars,
                    prefix_len=prefix_len,
                    context_len=context_len,
                    prefix_ids=prefix,
                    verify_ids=verify,
                    mask_id=mask_id,
                    draft_len=draft_len,
                    key=rng_key,
                    temperature=temperature,
                    top_k=top_k,
                    draft_logits=verify_logits,
                    always_accept=args.always_accept,
                    max_commit=max_commit,
                    return_prefix=not args.quiet,
                    kv_cache_len=kv_cache_len,
                )
                generated += r
                if num_new_tokens is None and generated >= max_steps:
                    break
                if not args.benchmark and cfg.inference.stop_on_eos and tokenizer.eos_token_id is not None:
                    committed = prefix[-r:] if prefix is not None else []
                    if tokenizer.eos_token_id in committed:
                        break

    elapsed = time.perf_counter() - start_time
    if args.benchmark:
        total_tokens = generated if num_new_tokens is not None else generated
        tok_s = total_tokens / max(elapsed, 1e-9)
        print(f"[benchmark] generated_tokens={total_tokens} time_s={elapsed:.4f} tok_s={tok_s:.2f}")

    if not args.quiet:
        output = tokenizer.decode(prefix, skip_special_tokens=True)
        print(output)


if __name__ == "__main__":
    main()
