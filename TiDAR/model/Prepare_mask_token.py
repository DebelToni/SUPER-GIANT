from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import core as flax_core
from omegaconf import OmegaConf

from GIANT.v2.model.checkpoint_manager import latest as latest_ckpt
from GIANT.v2.model.checkpoint_manager import load_npz, save_npz


def _resolve_config_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def load_configs(
    model_config_path: str | None = None,
    global_config_path: str | None = None,
) -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    resolved_model_cfg = _resolve_config_path(model_config_path, model_dir / "Config.yml")
    resolved_global_cfg = _resolve_config_path(global_config_path, project_root / "Global_Config.yml")
    cfg = OmegaConf.merge(
        OmegaConf.load(resolved_global_cfg),
        OmegaConf.load(resolved_model_cfg),
    )

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: str | None) -> str | None:
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

    for key in ("checkpoints_root", "hf_cache_root"):
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
    vocab = tokenizer.get_vocab()
    if tokenizer.mask_token and tokenizer.mask_token in vocab:
        token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        return tokenizer.mask_token, int(token_id), 0

    if base_token in vocab:
        try:
            tokenizer.add_special_tokens({"mask_token": base_token})
        except Exception:
            pass
        token_id = tokenizer.convert_tokens_to_ids(base_token)
        return base_token, int(token_id), 0

    alt_candidates = sorted(token for token in vocab if token.startswith(alt_prefix))
    if alt_candidates:
        candidate = alt_candidates[0]
        try:
            tokenizer.add_special_tokens({"mask_token": candidate})
        except Exception:
            pass
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        return candidate, int(token_id), 0

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


def resolve_params_dir(root: Path) -> Path:
    return root if root.name == "params" else root / "params"


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
            params_dir = resolve_params_dir(path)
            latest = latest_ckpt(str(params_dir))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {params_dir}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

    params_dir = resolve_params_dir(ckpt_dir)
    latest = latest_ckpt(str(params_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {params_dir}")
    return Path(latest)


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("Prepare TiDAR tokenizer + checkpoint")
    cli.add_argument("--config", type=str, default=None)
    cli.add_argument("--global_config", type=str, default=None)
    cli.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint (.npz) path or 'latest' (default).",
    )
    cli.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Override checkpoint directory for --checkpoint latest.",
    )
    cli.add_argument(
        "--output_checkpoint",
        default=None,
        help="Path to save updated checkpoint (defaults to in-place).",
    )
    cli.add_argument(
        "--tokenizer_out",
        default=None,
        help="Directory to save tokenizer if new tokens were added.",
    )
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--dry_run", action="store_true")
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    tokenizer = load_tokenizer(cfg)

    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"[mask] added token '{mask_token}' (id={mask_id})")
    else:
        print(f"[mask] using existing token '{mask_token}' (id={mask_id})")

    tokenizer_out = args.tokenizer_out
    if tokenizer_out is None and added_tokens > 0:
        tokenizer_out = getattr(cfg.tokenizer, "custom_path", None)
    if tokenizer_out:
        tok_path = Path(tokenizer_out)
        if not tok_path.is_absolute():
            tok_path = (Path(cfg.paths.data_root) / tok_path).resolve()
        tok_path.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            tokenizer.save_pretrained(tok_path)
        print(f"[tokenizer] saved to {tok_path}")
        if not cfg.tokenizer.use_custom:
            print("[tokenizer] set tokenizer.use_custom=true to use the saved tokenizer")

    ckpt_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    params = load_npz(ckpt_path)

    embedding = params["Embed_0"]["embedding"]
    old_vocab = int(embedding.shape[0])
    new_vocab = len(tokenizer)
    if new_vocab < old_vocab:
        raise ValueError(
            f"Tokenizer vocab ({new_vocab}) smaller than checkpoint ({old_vocab}); "
            "refusing to shrink embeddings."
        )

    rng = jax.random.PRNGKey(args.seed)
    params, added_rows = resize_embedding_params(params, new_vocab, key=rng)

    if added_rows == 0:
        print("[checkpoint] embeddings already match tokenizer; nothing to do.")
        return

    output_path = Path(args.output_checkpoint) if args.output_checkpoint else ckpt_path
    if not output_path.is_absolute():
        output_path = (Path(cfg.paths.data_root) / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[checkpoint] would save updated checkpoint to {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_npz(params, str(tmp_path))
    os.replace(tmp_path, output_path)
    print(f"[checkpoint] expanded embeddings by {added_rows} rows -> {output_path}")


if __name__ == "__main__":
    main()
