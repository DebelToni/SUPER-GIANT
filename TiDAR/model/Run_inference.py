from __future__ import annotations

import sys

import argparse
import time
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
from model.checkpoint_manager import latest as latest_ckpt
from model.checkpoint_manager import load as load_ckpt
from model.tidar_inference import (
    init_kv_cache,
    prefill_prompt_cache,
    tidar_prefill_draft_cached,
    tidar_decode_step_cached,
)
from model.tidar_utils import sample_from_logits


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


def resolve_mask_id(tokenizer, cfg: OmegaConf) -> int:
    mask_token = getattr(cfg.tokenizer, "mask_token_override", None)
    if mask_token:
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_id is None or mask_id == tokenizer.unk_token_id:
            print(f"[mask] override '{mask_token}' not found; falling back to pad token")
            mask_token = None
    if mask_token is None:
        if tokenizer.mask_token_id is not None:
            return int(tokenizer.mask_token_id)
        if tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
    return 0


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

    draft_len = int(cfg.tidar.draft_length) if args.draft_len is None else int(args.draft_len)
    max_steps = args.max_steps or int(cfg.inference.max_decode_steps)
    num_new_tokens = args.num_new_tokens
    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)

    prompt_ids = tokenizer(args.prompt, return_tensors="np").input_ids[0].astype(np.int32)

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

    mask_id = resolve_mask_id(tokenizer, cfg)
    rng = np.random.default_rng(args.seed)

    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=tokenizer.pad_token_id or 0)
    cache_vars, prefix_len = prefill_prompt_cache(model, params, cache_vars, prompt_ids)

    prefix = prompt_ids
    generated = 0
    start_time = time.perf_counter()

    if draft_len <= 0:
        use_device_sampling = args.benchmark and top_k == 0
        last_token = int(prefix[-1]) if prefix_len > 0 else int(mask_id)
        last_token_dev = jnp.array([[last_token]], dtype=jnp.int32)
        rng_key = jax.random.PRNGKey(args.seed)
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
        draft_ids, draft_logits = tidar_prefill_draft_cached(
            model,
            params,
            cache_vars,
            prefix_len=prefix_len,
            batch_size=1,
            mask_id=mask_id,
            draft_len=draft_len,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
        )
        verify = draft_ids
        verify_logits = draft_logits

        while num_new_tokens is None or generated < num_new_tokens:
            if num_new_tokens is not None:
                remaining = num_new_tokens - generated
                if remaining <= 0:
                    break
                max_commit = remaining if args.always_accept else None
            else:
                max_commit = None

            prefix, verify, verify_logits, r, cache_vars, prefix_len = tidar_decode_step_cached(
                model,
                params,
                cache_vars,
                prefix_len=prefix_len,
                prefix_ids=prefix,
                verify_ids=verify,
                mask_id=mask_id,
                draft_len=draft_len,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
                draft_logits=verify_logits,
                always_accept=args.always_accept,
                max_commit=max_commit,
            )
            generated += r
            if num_new_tokens is None and generated >= max_steps:
                break
            if not args.benchmark and cfg.inference.stop_on_eos and tokenizer.eos_token_id is not None:
                committed = prefix[-r:]
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
