from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


PROJECT_DIR = Path(__file__).resolve().parent
TRM_ROOT = PROJECT_DIR.parent
V2_ROOT = TRM_ROOT.parent / "v2"

GLOBAL_CFG_PATH = TRM_ROOT / "Global_Config.yml"
SMOL_CFG_PATH = V2_ROOT / "smol" / "Config.yml"
LOCAL_CFG_PATH = PROJECT_DIR / "Config.yml"


def _resolve_with_base(base: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_config() -> OmegaConf:
    configs = []
    if GLOBAL_CFG_PATH.exists():
        configs.append(OmegaConf.load(GLOBAL_CFG_PATH))
    if SMOL_CFG_PATH.exists():
        configs.append(OmegaConf.load(SMOL_CFG_PATH))
    if LOCAL_CFG_PATH.exists():
        configs.append(OmegaConf.load(LOCAL_CFG_PATH))
    if not configs:
        raise FileNotFoundError("No configuration files found for TRM token tool.")

    cfg = OmegaConf.merge(*configs)

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else TRM_ROOT
    if "paths" not in cfg:
        cfg.paths = {}
    cfg.paths.data_root = str(base_prefix)

    for key in (
        "processed_data_root",
        "dataloader_state_root",
        "logs_root",
        "raw_data_root",
        "tokenizer_root",
        "checkpoint_root",
    ):
        if "paths" in cfg and key in cfg.paths and cfg.paths[key] is not None:
            cfg.paths[key] = _resolve_with_base(base_prefix, cfg.paths[key])

    if "tokenizer" in cfg:
        if "cache_dir" in cfg.tokenizer and cfg.tokenizer.cache_dir is not None:
            cfg.tokenizer.cache_dir = _resolve_with_base(base_prefix, cfg.tokenizer.cache_dir)
        if "custom_path" in cfg.tokenizer and cfg.tokenizer.custom_path is not None:
            cfg.tokenizer.custom_path = _resolve_with_base(base_prefix, cfg.tokenizer.custom_path)

    if "data" in cfg:
        for key in ("prompt_template_path", "samples_path", "hf_cache_dir"):
            if key in cfg.data and cfg.data[key] is not None:
                cfg.data[key] = _resolve_with_base(base_prefix, cfg.data[key])

    if "training" in cfg and "init_checkpoint" in cfg.training and cfg.training.init_checkpoint is not None:
        cfg.training.init_checkpoint = _resolve_with_base(base_prefix, cfg.training.init_checkpoint)

    return cfg


def resolve_data_path(cfg: OmegaConf, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(cfg.paths.data_root) / path
