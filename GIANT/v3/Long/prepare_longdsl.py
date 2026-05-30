from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from GIANT.v3.Long.longdsl import GeneratorConfig, TokenizerSpec, generate_example, save_tokenizer, surface_tokens, write_jsonl
from GIANT.v3.data_pipeline.build_corpus import ShardWriter, _arrow_schema, _pack_loss_mask


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _dataset_name(level: int, ctx: int) -> str:
    return f"longrecords_l{level}_ctx{ctx}"


def _num_train_examples(train_tokens_per_context: int, ctx: int) -> int:
    return max(64, math.ceil(train_tokens_per_context / max(1, ctx)))


def _resolve_train_examples(args, ctx: int) -> int:
    if args.train_examples is not None:
        return max(1, int(args.train_examples))
    return _num_train_examples(args.train_tokens_per_context, ctx)


def _generate_split(cfg: GeneratorConfig, count: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    return [generate_example(cfg, rng) for _ in range(count)]


def _write_arrow_stage(
    stage_name: str,
    ctx: int,
    raw_path: Path,
    output_root: Path,
    tokenizer,
    *,
    mask_mode: str,
) -> dict[str, object]:
    stage_output_dir = output_root / stage_name
    if stage_output_dir.exists():
        shutil.rmtree(stage_output_dir)
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_shard = 8192
    writer = ShardWriter(stage_output_dir, stage_name, _arrow_schema(ctx), rows_per_shard)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buffer: list[dict[str, object]] = []
    total_tokens = 0
    documents = 0
    discarded = 0

    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = surface_tokens(str(row["text"]))
            ids = tokenizer.convert_tokens_to_ids(tokens)
            if any(int(x) == tokenizer.unk_token_id for x in ids):
                raise ValueError(f"Unknown token found while encoding {stage_name}")
            if len(ids) > ctx:
                discarded += 1
                continue
            length = len(ids)
            answer_idx = int(row["answer_token_index"])
            pred_idx = answer_idx - 1
            if pred_idx < 0 or pred_idx >= length:
                discarded += 1
                continue
            if mask_mode == "answer_only":
                mask = [0.0] * length
                mask[pred_idx] = 1.0
            elif mask_mode == "lm":
                mask = [1.0] * max(length - 1, 0) + [0.0]
            else:
                raise ValueError(f"Unknown mask_mode={mask_mode}")
            padded = ids + [pad_id] * (ctx - length)
            packed_mask = _pack_loss_mask(mask + [0.0] * (ctx - length), ctx)
            buffer.append({"input_ids": padded, "length": length, "loss_mask": packed_mask})
            documents += 1
            total_tokens += length
            if len(buffer) >= 256:
                writer.append(buffer)
                buffer = []

    if buffer:
        writer.append(buffer)
    writer.close()
    manifest = writer.manifest
    with (stage_output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage": stage_name,
                "description": "Long hidden-world custom arrow stage",
                "total_records": documents,
                "total_tokens": total_tokens,
                "documents": documents,
                "shards": manifest,
            },
            handle,
            indent=2,
        )
    return {
        "documents": documents,
        "sequences": documents,
        "tokens": total_tokens,
        "discarded": discarded,
        "output_dir": str(stage_output_dir),
        "shards": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongGIANT natural-language hidden-world tokenizer and datasets")
    parser.add_argument("--artifact_root", default="/proj/giant-data/GIANT/Long")
    parser.add_argument("--contexts", default="128,256,512")
    parser.add_argument("--levels", default="1,2")
    parser.add_argument("--train_tokens_per_context", type=int, default=196_608)
    parser.add_argument("--train_examples", type=int, default=None)
    parser.add_argument("--val_examples", type=int, default=128)
    parser.add_argument("--test_examples", type=int, default=128)
    parser.add_argument("--fill_ratio", type=float, default=0.88)
    parser.add_argument("--min_alias_chain", type=int, default=1)
    parser.add_argument("--max_alias_chain", type=int, default=3)
    parser.add_argument("--min_fill_events", type=int, default=4)
    parser.add_argument("--max_fill_events", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_root", default="/proj/giant-data/GIANT/dataset_artifacts/long_records")
    parser.add_argument("--build_corpus", action="store_true")
    parser.add_argument("--global_config", default=str(Path(__file__).resolve().parents[1] / "Global_Config.yml"))
    parser.add_argument("--config_out", default=None)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    tokenizer_dir = artifact_root / "tokenizers" / "long_wordlevel"
    raw_root = artifact_root / "records" / "raw"
    config_root = artifact_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    spec = TokenizerSpec()
    save_tokenizer(tokenizer_dir, spec)

    contexts = _parse_int_list(args.contexts)
    levels = _parse_int_list(args.levels)
    corpus_cfg = {
        "tokenizer": {
            "use_custom": True,
            "custom_path": str(tokenizer_dir),
        },
        "paths": {
            "data_root": "/proj/giant-data/GIANT",
            "processed_data_root": str(Path(args.dataset_root).relative_to("/proj/giant-data/GIANT")),
        },
        "outputs": {
            "processed_root": str(Path(args.dataset_root).relative_to("/proj/giant-data/GIANT")),
            "rows_per_shard": 8192,
        },
        "scheduling": {
            "seed": args.seed,
            "write_batch_size": 256,
        },
        "stages": {},
    }

    for level in levels:
        for ctx in contexts:
            cfg = GeneratorConfig(
                level=level,
                context_length=ctx,
                fill_ratio=args.fill_ratio,
                min_alias_chain=args.min_alias_chain,
                max_alias_chain=args.max_alias_chain,
                min_fill_events=args.min_fill_events,
                max_fill_events=args.max_fill_events,
            )
            dataset_name = _dataset_name(level, ctx)
            stage_dir = raw_root / f"level{level}_ctx{ctx}"
            train_examples = _resolve_train_examples(args, ctx)
            train_rows = _generate_split(cfg, train_examples, args.seed + level * 1000 + ctx)
            val_rows = _generate_split(cfg, args.val_examples, args.seed + level * 1000 + ctx + 1)
            test_rows = _generate_split(cfg, args.test_examples, args.seed + level * 1000 + ctx + 2)
            write_jsonl(stage_dir / "train.jsonl", train_rows)
            write_jsonl(stage_dir / "val.jsonl", val_rows)
            write_jsonl(stage_dir / "test.jsonl", test_rows)
            print(
                f"[long] level={level} ctx={ctx} train_examples={train_examples} "
                f"(train_tokens_per_context={args.train_tokens_per_context})"
            )

            corpus_cfg["stages"][dataset_name] = {
                "output_dir": dataset_name,
                "sequence_length": ctx,
                "min_tokens": 1,
                "pack_sequences": False,
                "add_eos": False,
                "emit_final_partial": False,
                "long_document_strategy": "sequential",
                "max_windows_per_document": 1,
                "sources": [
                    {
                        "type": "json",
                        "json_root": str(stage_dir.relative_to("/proj/giant-data/GIANT")),
                        "file_glob": "train.jsonl",
                        "text_field": "text",
                    }
                ],
            }

    config_out = Path(args.config_out) if args.config_out else config_root / "long_records_datasets.yml"
    OmegaConf.save(config=OmegaConf.create(corpus_cfg), f=str(config_out))
    print(f"[long] tokenizer -> {tokenizer_dir}")
    print(f"[long] dataset config -> {config_out}")

    if args.build_corpus:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
        output_root = Path(args.dataset_root)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = {}
        stats = {}
        for level in levels:
            for ctx in contexts:
                dataset_name = _dataset_name(level, ctx)
                train_path = raw_root / f"level{level}_ctx{ctx}" / "train.jsonl"
                for suffix, mask_mode in (("lm", "lm"), ("ans", "answer_only")):
                    stage_name = f"{dataset_name}_{suffix}"
                    stage_stats = _write_arrow_stage(
                        stage_name,
                        ctx,
                        train_path,
                        output_root,
                        tokenizer,
                        mask_mode=mask_mode,
                    )
                    manifest[stage_name] = {
                        "path": str(output_root / stage_name),
                        "sequence_length": ctx,
                    }
                    stats[stage_name] = stage_stats
                    print(
                        f"[long] stage={stage_name} docs={stage_stats['documents']} tokens={stage_stats['tokens']} discarded={stage_stats['discarded']}"
                    )
        with (output_root / "datasets_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        with (output_root / "dataset_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)


if __name__ == "__main__":
    main()
