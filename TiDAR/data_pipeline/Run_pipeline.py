from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
from GIANT.v2.data_pipeline.build_corpus import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TiDAR data pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Override outputs.processed_root (relative to paths.data_root unless absolute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(Path(__file__).resolve().parent / "Config.yml")
    global_config_path = args.global_config
    if global_config_path is None:
        global_config_path = str(Path(__file__).resolve().parents[1] / "Global_Config.yml")
    
    # Set HF cache environment variables from global config
    if global_config_path and Path(global_config_path).exists():
        global_cfg = OmegaConf.load(global_config_path)
        hf_cache_root = global_cfg.get("paths", {}).get("hf_cache_root")
        if hf_cache_root:
            os.environ["HF_HOME"] = str(hf_cache_root)
            os.environ["HF_DATASETS_CACHE"] = str(Path(hf_cache_root) / "datasets")
            os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_cache_root) / "transformers")
            print(f"[HF Cache] Set HF_HOME={hf_cache_root}")
    
    run_pipeline(
        config_path=config_path,
        global_config_path=global_config_path,
        stage=args.stage,
        dry_run=args.dry_run,
        dataset_dir=args.dataset_dir,
    )
    
    # Force exit to prevent hanging from HuggingFace datasets background threads
    # When streaming datasets are stopped early, background threads may not cleanup properly
    os._exit(0)


if __name__ == "__main__":
    main()
