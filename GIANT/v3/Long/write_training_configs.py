from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _cfg(level: int, ctx: int, tokenizer_path: str, dataset_root: str, checkpoint_tag: str) -> dict:
    name = f"longrecords_l{level}_ctx{ctx}"
    return {
        "global_seed": 0,
        "model": {
            "embedding_size": 448,
            "num_heads": 7,
            "num_kv_heads": 7,
            "num_layers": 12,
            "feed_forward_size": 1792,
            "rope_dim": 64,
            "context_length": ctx,
            "dropout_rate": 0.0,
            "activation": "silu",
            "use_remat": False,
            "enable_xsa": True,
            "causal": True,
            "param_dtype": "float32",
            "compute_dtype": "bfloat16",
        },
        "tokenizer": {
            "name": "unused",
            "cache_dir": "/proj/giant-data/hf_cache/tokenizer_cache",
            "use_custom": True,
            "custom_path": tokenizer_path,
            "pad_token_override": None,
        },
        "optimizer": {
            "base_learning_rate": 2.0e-4,
            "min_learning_rate": 2.0e-5,
            "warmup_tokens": 12_000,
            "weight_decay": 0.10,
            "gradient_clip_norm": 1.0,
            "weight_decay_exclusions": ["bias", "scale", "embedding"],
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation": 1,
            "seed": 0,
            "log_every": 10,
            "checkpoint_every": 250,
            "mini_checkpoint_every": 0,
            "mini_max_to_keep": 1,
            "scan_chunk": 1,
            "prefetch_size": 1,
            "nan_check": False,
            "autotune_scan_chunk": False,
            "autotune_prefetch": False,
        },
        "stages": [
            {
                "name": f"{name}_lm",
                "dataset": f"{name}_lm",
                "seq_len": ctx,
                "epochs": 1,
                "fraction": 1.0,
                "shuffle": True,
            },
            {
                "name": f"{name}_ans",
                "dataset": f"{name}_ans",
                "seq_len": ctx,
                "epochs": 6 if level == 1 else 8,
                "fraction": 1.0,
                "shuffle": True,
            }
        ],
        "qa_finetune": {},
        "paths": {
            "data_root": "/proj/giant-data/GIANT",
            "processed_data_root": dataset_root,
            "dataloader_state_root": f"single-gpu/checkpoints/long/{checkpoint_tag}/dataloader_state",
            "logs_root": f"single-gpu/logs/long/{checkpoint_tag}",
            "checkpoints_root": f"single-gpu/checkpoints/long/{checkpoint_tag}",
            "hf_cache_root": "/proj/giant-data/hf_cache",
        },
        "inference": {
            "stop_on_eos": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Long hidden-world training configs")
    parser.add_argument("--out_dir", default="GIANT/v3/Configs/Training/Long")
    parser.add_argument("--levels", default="1,2")
    parser.add_argument("--contexts", default="128,256,512")
    parser.add_argument("--tokenizer_path", default="/proj/giant-data/GIANT/Long/tokenizers/long_wordlevel")
    parser.add_argument("--dataset_root", default="dataset_artifacts/long_records")
    parser.add_argument("--run_tag", default="records_bootstrap")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in _parse_int_list(args.levels):
        for ctx in _parse_int_list(args.contexts):
            checkpoint_tag = f"{args.run_tag}/40m/l{level}_ctx{ctx}"
            payload = _cfg(level, ctx, args.tokenizer_path, args.dataset_root, checkpoint_tag)
            out_path = out_dir / f"long_40m_l{level}_ctx{ctx}.yml"
            OmegaConf.save(config=OmegaConf.create(payload), f=str(out_path))
            print(f"[long] wrote {out_path}")


if __name__ == "__main__":
    main()
