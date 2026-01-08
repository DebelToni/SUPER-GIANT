from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer


@dataclass
class StageCfg:
    name: str
    output_dir: str
    sequence_length: int
    target_tokens: Optional[int]
    min_tokens: int
    pack_sequences: bool
    add_eos: bool
    max_documents: Optional[int]
    sources: List[Dict]


def load_configs() -> OmegaConf:
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(pipeline_dir / "Config.yml"),
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

    for key in ("processed_data_root", "hf_cache_root"):
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


def parse_stage(stage_cfg: Dict) -> StageCfg:
    return StageCfg(
        name=stage_cfg["name"],
        output_dir=stage_cfg.get("output_dir", stage_cfg["name"]),
        sequence_length=int(stage_cfg["sequence_length"]),
        target_tokens=int(stage_cfg.get("target_tokens")) if stage_cfg.get("target_tokens") else None,
        min_tokens=int(stage_cfg.get("min_tokens", 1)),
        pack_sequences=bool(stage_cfg.get("pack_sequences", True)),
        add_eos=bool(stage_cfg.get("add_eos", True)),
        max_documents=int(stage_cfg.get("max_documents")) if stage_cfg.get("max_documents") else None,
        sources=list(stage_cfg.get("sources", [])),
    )


def iter_documents(stage: StageCfg, *, cache_dir: Optional[str]) -> Iterable[str]:
    for source in stage.sources:
        if source.get("type") != "huggingface":
            raise ValueError(f"Unsupported source type: {source.get('type')}")
        dataset_name = source.get("dataset_name")
        if not dataset_name:
            raise ValueError("huggingface source requires dataset_name")
        split = source.get("split", "train")
        streaming = bool(source.get("streaming", False))
        text_field = source.get("text_field")
        ds = load_dataset(dataset_name, split=split, streaming=streaming, cache_dir=cache_dir)
        count = 0
        for row in ds:
            if stage.max_documents and count >= stage.max_documents:
                break
            text = row.get(text_field) if text_field else None
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            count += 1
            yield text


def write_shard(sequences: List[List[int]], lengths: List[int], out_path: Path) -> int:
    if not sequences:
        return 0
    array = pa.array(sequences, type=pa.list_(pa.int32()))
    lengths_arr = pa.array(lengths, type=pa.int32())
    table = pa.Table.from_arrays([array, lengths_arr], names=["input_ids", "length"])
    with pa_ipc.new_file(out_path, table.schema) as writer:
        writer.write_table(table)
    return len(sequences)


def build_stage(stage: StageCfg, *, cfg: OmegaConf, tokenizer) -> Dict[str, int]:
    data_root = Path(cfg.paths.data_root)
    processed_root = Path(cfg.paths.processed_data_root)
    if not processed_root.is_absolute():
        processed_root = (data_root / processed_root).resolve()
    output_dir = processed_root / stage.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_shard = int(cfg.outputs.rows_per_shard)
    sequences: List[List[int]] = []
    lengths: List[int] = []
    shard_idx = 0
    total_tokens = 0
    total_sequences = 0

    eos_id = tokenizer.eos_token_id
    cache_dir = cfg.paths.get("hf_cache_root")

    buffer: List[int] = []
    for doc in iter_documents(stage, cache_dir=cache_dir):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if stage.add_eos and eos_id is not None:
            ids.append(eos_id)
        if stage.pack_sequences:
            buffer.extend(ids)
            while len(buffer) >= stage.sequence_length:
                seq = buffer[: stage.sequence_length]
                buffer = buffer[stage.sequence_length :]
                if len(seq) < stage.min_tokens:
                    continue
                sequences.append(seq)
                lengths.append(len(seq))
                total_sequences += 1
                total_tokens += len(seq)
                if len(sequences) >= rows_per_shard:
                    shard_path = output_dir / f"shard_{shard_idx:05d}.arrow"
                    write_shard(sequences, lengths, shard_path)
                    shard_idx += 1
                    sequences, lengths = [], []
                if stage.target_tokens and total_tokens >= stage.target_tokens:
                    break
        else:
            for start in range(0, len(ids), stage.sequence_length):
                seq = ids[start : start + stage.sequence_length]
                if len(seq) < stage.min_tokens:
                    continue
                sequences.append(seq)
                lengths.append(len(seq))
                total_sequences += 1
                total_tokens += len(seq)
                if len(sequences) >= rows_per_shard:
                    shard_path = output_dir / f"shard_{shard_idx:05d}.arrow"
                    write_shard(sequences, lengths, shard_path)
                    shard_idx += 1
                    sequences, lengths = [], []
                if stage.target_tokens and total_tokens >= stage.target_tokens:
                    break
        if stage.target_tokens and total_tokens >= stage.target_tokens:
            break

    if sequences:
        shard_path = output_dir / f"shard_{shard_idx:05d}.arrow"
        write_shard(sequences, lengths, shard_path)

    manifest = {
        "stage": stage.name,
        "shards": [],
    }
    for path in sorted(output_dir.glob("shard_*.arrow")):
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_file(source)
            table = reader.read_all()
        manifest["shards"].append({"filename": path.name, "rows": table.num_rows})

    manifest_path = output_dir / cfg.outputs.manifest_filename
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    stats = {
        "stage": stage.name,
        "sequences": total_sequences,
        "tokens": total_tokens,
    }
    stats_path = output_dir / cfg.outputs.stats_filename
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("Build TiDAR dataset")
    cli.add_argument("--stage", default=None, help="Only build a single stage by name.")
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    tokenizer = load_tokenizer(cfg)

    stages_raw = OmegaConf.to_container(cfg.stages, resolve=True)
    stages = [parse_stage(stage) for stage in stages_raw]
    if args.stage:
        stages = [stage for stage in stages if stage.name == args.stage]
        if not stages:
            raise ValueError(f"Stage '{args.stage}' not found in config")

    for stage in stages:
        stats = build_stage(stage, cfg=cfg, tokenizer=tokenizer)
        print(f"[data] {stage.name}: {stats['sequences']} sequences, {stats['tokens']} tokens")


if __name__ == "__main__":
    main()
