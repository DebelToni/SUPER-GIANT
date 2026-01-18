from __future__ import annotations

import argparse
from pathlib import Path

from GIANT.v2.data_pipeline.build_corpus import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TiDAR data pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(Path(__file__).resolve().parent / "Config.yml")
    run_pipeline(
        config_path=config_path,
        global_config_path=args.global_config,
        stage=args.stage,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
