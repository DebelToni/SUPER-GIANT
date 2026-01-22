from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from TiDAR.model.GiantTiDAR import TiDAR
from GIANT.v2.model.checkpoint_manager import save_npz


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


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.name,
        cache_dir=cfg.tokenizer.cache_dir,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Save TiDAR init params")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    tokenizer = load_tokenizer(cfg)

    model_cfg = cfg.model
    model = TiDAR(
        vocab_size=len(tokenizer),
        context_length=model_cfg.context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
    )

    dummy = jnp.full((1, 1), 0, dtype=jnp.int32)
    variables = model.init({"params": jax.random.PRNGKey(0)}, dummy, deterministic=True)
    params = variables["params"]

    data_root = Path(cfg.paths.data_root)
    checkpoints_dir = data_root / cfg.paths.checkpoints_root
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoints_dir / "init_params.npz"
    save_npz(params, str(save_path))
    print(f"Saved init params to {save_path}")


if __name__ == "__main__":
    main()
