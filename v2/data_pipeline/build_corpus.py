from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import load_dataset, IterableDataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Allow running as a script without -m.
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from .cleaning import normalise_text
    from .dedup import Deduplicator
except ImportError:
    from cleaning import normalise_text
    from dedup import Deduplicator

LOGGER = logging.getLogger("data_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class StageStats:
    records: int = 0
    tokens: int = 0
    duplicates: int = 0
    discarded: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "records": self.records,
            "tokens": self.tokens,
            "duplicates": self.duplicates,
            "discarded": self.discarded,
        }


class ArrowBatchWriter:
    def __init__(self, output_path: Path, schema: pa.Schema, batch_size: int) -> None:
        self._output_path = output_path
        self._schema = schema
        self._batch_size = batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pa_ipc.RecordBatchFileWriter] = None
        self._sink: Optional[pa.OSFile] = None

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = pa.OSFile(str(self._output_path), "wb")
        self._writer = pa_ipc.new_file(self._sink, self._schema)

    def write(self, record: Dict[str, Any]) -> None:
        self._ensure_writer()
        self._buffer.append(record)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def close(self) -> None:
        if self._writer is None:
            return
        self._flush(force=True)
        self._writer.close()
        assert self._sink is not None
        self._sink.close()
        self._writer = None
        self._sink = None

    def _flush(self, force: bool = False) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        assert self._writer is not None
        self._writer.write_table(table)
        self._buffer.clear()


def stage_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source", pa.string()),
            ("document_id", pa.string()),
            ("window_index", pa.int32()),
            ("seq_length", pa.int32()),
            ("context_length", pa.int32()),
            ("pad_id", pa.int32()),
            ("hash_key", pa.uint64()),
            ("input_ids", pa.list_(pa.int32())),
        ]
    )


def _writer_batch_size(cfg: OmegaConf) -> int:
    scheduling = getattr(cfg, "scheduling", None)
    if scheduling is not None and "write_batch_size" in scheduling:
        return int(scheduling.write_batch_size)
    return 512


def load_configs() -> tuple[OmegaConf, OmegaConf, OmegaConf]:
    root = Path(__file__).resolve().parent.parent
    global_cfg = OmegaConf.load(root / "Global_Config.yml")
    local_cfg = OmegaConf.load(Path(__file__).resolve().parent / "Config.yml")
    merged = OmegaConf.merge(global_cfg, local_cfg)
    return global_cfg, local_cfg, merged


def load_tokenizer(global_cfg: OmegaConf) -> PreTrainedTokenizerBase:
    tok_cfg = global_cfg.tokenizer
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


def _trim_padding(tokens: List[int], pad_id: int) -> List[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] == pad_id:
        end -= 1
    return tokens[:end]


def _yield_windows(
    tokens: List[int],
    *,
    context_length: int,
    min_tokens: int,
    pad_id: int,
    pad_to_context: bool,
) -> Iterator[List[int]]:
    position = 0
    total = len(tokens)
    while position < total:
        window = tokens[position : position + context_length]
        if len(window) < min_tokens:
            break
        if pad_to_context and len(window) < context_length:
            window = window + [pad_id] * (context_length - len(window))
        yield window[:context_length]
        position += context_length


def _get_doc_id(record: Dict[str, Any], fallback: str) -> str:
    for key in ("id", "doc_id", "article_id", "story_id"):
        if key in record and record[key] is not None:
            return str(record[key])
    meta = record.get("meta")
    if isinstance(meta, dict):
        for key in ("id", "doc_id", "article_id"):
            if key in meta:
                return str(meta[key])
    return fallback


def _resolve_text(record: Dict[str, Any]) -> Optional[str]:
    for key in ("text", "story", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return None


def build_tinystories(
    cfg: OmegaConf,
    global_cfg: OmegaConf,
    tokenizer: PreTrainedTokenizerBase,
    output_path: Path,
) -> StageStats:
    stats = StageStats()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    min_tokens = int(getattr(cfg, "min_tokens", global_cfg.io.min_sequence_tokens))
    batch_size = _writer_batch_size(global_cfg)

    writer = ArrowBatchWriter(output_path, stage_schema(), batch_size)
    dedup = Deduplicator()

    options = SimpleNamespace(nfkc=True, collapse_whitespace=True, strip_markup=False)

    LOGGER.info("Building TinyStories shard → %s", output_path)
    dataset = load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
        streaming=bool(cfg.streaming),
    )

    if isinstance(dataset, IterableDataset):
        iterator = dataset
    else:
        iterator = iter(dataset)

    max_records = cfg.max_records if cfg.max_records else None

    for idx, record in enumerate(iterator):
        if max_records is not None and idx >= max_records:
            break
        text = _resolve_text(record)
        if not text:
            stats.discarded += 1
            continue
        cleaned = normalise_text(text, options)
        if not cleaned:
            stats.discarded += 1
            continue
        tokens = tokenizer.encode(cleaned, add_special_tokens=False)
        if len(tokens) < min_tokens:
            stats.discarded += 1
            continue
        windows = list(
            _yield_windows(
                tokens,
                context_length=cfg.context_length,
                min_tokens=min_tokens,
                pad_id=pad_id,
                pad_to_context=True,
            )
        )
        if not windows:
            stats.discarded += 1
            continue
        doc_id = _get_doc_id(record, f"tinystory-{idx}")
        for w_idx, window in enumerate(windows):
            trimmed = _trim_padding(window, pad_id)
            is_dup, key = dedup.check_and_add(trimmed)
            if is_dup:
                stats.duplicates += 1
                continue
            key_value = int(key)
            writer.write(
                {
                    "source": "tinystories",
                    "document_id": doc_id,
                    "window_index": w_idx,
                    "seq_length": len(trimmed),
                    "context_length": cfg.context_length,
                    "pad_id": pad_id,
                    "hash_key": key_value,
                    "input_ids": window,
                }
            )
            stats.records += 1
            stats.tokens += len(trimmed)

    writer.close()
    LOGGER.info(
        "TinyStories done → %d windows | %d duplicates filtered | %d discarded",
        stats.records,
        stats.duplicates,
        stats.discarded,
    )
    return stats


def _iter_wikipedia_files(cfg: OmegaConf) -> Iterator[Path]:
    root = Path(cfg.raw_root)
    glob_pattern = cfg.file_glob or "**/*.json"
    files = list(root.glob(glob_pattern))
    files.sort()
    if cfg.max_files is not None:
        files = files[: int(cfg.max_files)]
    for path in files:
        if path.is_file():
            yield path


def _load_wikipedia_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as err:
            LOGGER.warning("Skipping %s (%s)", path.name, err)
            return
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield entry
    elif isinstance(data, dict):
        yield data


def build_wikipedia(
    cfg: OmegaConf,
    global_cfg: OmegaConf,
    tokenizer: PreTrainedTokenizerBase,
    output_path: Path,
) -> StageStats:
    stats = StageStats()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    min_tokens = int(getattr(cfg, "min_tokens", global_cfg.io.min_sequence_tokens))
    batch_size = _writer_batch_size(global_cfg)

    writer = ArrowBatchWriter(output_path, stage_schema(), batch_size)
    dedup = Deduplicator(hash_bits=cfg.dedup_hash_bits if cfg.deduplicate else 64)

    options = cfg.normalization

    LOGGER.info("Building Wikipedia shard → %s", output_path)
    record_limit = int(cfg.max_records) if cfg.max_records else None

    record_counter = 0
    for file_index, path in enumerate(_iter_wikipedia_files(cfg)):
        for rec_idx, record in enumerate(_load_wikipedia_records(path)):
            if record_limit is not None and record_counter >= record_limit:
                break
            record_counter += 1
            text = record.get(cfg.text_field) if cfg.text_field else _resolve_text(record)
            if not isinstance(text, str) or not text.strip():
                stats.discarded += 1
                continue
            cleaned = normalise_text(text, options)
            if not cleaned:
                stats.discarded += 1
                continue
            tokens = tokenizer.encode(cleaned, add_special_tokens=False)
            if len(tokens) < min_tokens:
                stats.discarded += 1
                continue
            windows = list(
                _yield_windows(
                    tokens,
                    context_length=cfg.context_length,
                    min_tokens=min_tokens,
                    pad_id=pad_id,
                    pad_to_context=bool(cfg.pads_to_context),
                )
            )
            if not windows:
                stats.discarded += 1
                continue
            doc_id = _get_doc_id(record, f"wiki-{file_index}-{rec_idx}")
            for w_idx, window in enumerate(windows):
                trimmed = _trim_padding(window, pad_id)
                if len(trimmed) < min_tokens:
                    stats.discarded += 1
                    continue
                if cfg.deduplicate:
                    is_dup, key = dedup.check_and_add(trimmed)
                else:
                    is_dup, key = False, dedup.check_and_add(trimmed)[1]
                if is_dup:
                    stats.duplicates += 1
                    continue
                key_value = int(key)
                writer.write(
                    {
                        "source": "wikipedia",
                        "document_id": doc_id,
                        "window_index": w_idx,
                        "seq_length": len(trimmed),
                        "context_length": cfg.context_length,
                        "pad_id": pad_id,
                        "hash_key": key_value,
                        "input_ids": window,
                    }
                )
                stats.records += 1
                stats.tokens += len(trimmed)
        if record_limit is not None and record_counter >= record_limit:
            break

    writer.close()
    LOGGER.info(
        "Wikipedia done → %d windows | %d duplicates filtered | %d discarded",
        stats.records,
        stats.duplicates,
        stats.discarded,
    )
    return stats


def merge_datasets(
    merged_cfg: OmegaConf,
    tokenizer: PreTrainedTokenizerBase,
    input_files: List[Path],
    output_path: Path,
) -> StageStats:
    stats = StageStats()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    target_context = max(
        int(merged_cfg.tiny_stories.context_length),
        int(merged_cfg.wikipedia.context_length),
    )
    batch_size = _writer_batch_size(merged_cfg)

    writer = ArrowBatchWriter(output_path, stage_schema(), batch_size)
    dedup = Deduplicator()

    LOGGER.info("Merging %d shards → %s", len(input_files), output_path)

    for path in input_files:
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_file(source)
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index).to_pylist()
                for row in batch:
                    tokens = list(row["input_ids"])
                    row_pad = int(row.get("pad_id", pad_id))
                    trimmed = _trim_padding(tokens, row_pad)
                    is_dup, key = dedup.check_and_add(trimmed)
                    if is_dup:
                        stats.duplicates += 1
                        continue
                    if len(tokens) > target_context:
                        tokens = tokens[:target_context]
                        trimmed = _trim_padding(tokens, row_pad)
                    elif len(tokens) < target_context:
                        tokens = tokens + [pad_id] * (target_context - len(tokens))
                    writer.write(
                        {
                            "source": row["source"],
                            "document_id": row["document_id"],
                            "window_index": int(row["window_index"]),
                            "seq_length": len(trimmed),
                            "context_length": target_context,
                            "pad_id": pad_id,
                            "hash_key": key,
                            "input_ids": tokens,
                        }
                    )
                    stats.records += 1
                    stats.tokens += len(trimmed)

    writer.close()
    LOGGER.info(
        "Merged dataset → %d windows | %d duplicates filtered",
        stats.records,
        stats.duplicates,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SUPER-GIANT dataset builder")
    parser.add_argument(
        "--stage",
        choices=["tinystories", "wikipedia", "merge", "all"],
        default="all",
        help="Stage to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and preprocess but do not write output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg, local_cfg, merged_cfg = load_configs()
    tokenizer = load_tokenizer(global_cfg)
    output_root = Path(merged_cfg.outputs.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}

    if args.stage in ("tinystories", "all"):
        path = output_root / merged_cfg.outputs.tinystories_filename
        if not args.dry_run:
            stats["tinystories"] = build_tinystories(
                merged_cfg.tiny_stories,
                merged_cfg,
                tokenizer,
                path,
            ).to_dict()
        else:
            LOGGER.info("Dry run: skipping TinyStories write")

    if args.stage in ("wikipedia", "all"):
        path = output_root / merged_cfg.outputs.wikipedia_filename
        if not args.dry_run:
            stats["wikipedia"] = build_wikipedia(
                merged_cfg.wikipedia,
                merged_cfg,
                tokenizer,
                path,
            ).to_dict()
        else:
            LOGGER.info("Dry run: skipping Wikipedia write")

    if args.stage in ("merge", "all"):
        tinypath = output_root / merged_cfg.outputs.tinystories_filename
        wikipath = output_root / merged_cfg.outputs.wikipedia_filename
        inputs = [p for p in [tinypath, wikipath] if p.exists()]
        if not inputs:
            LOGGER.warning("No input shards found to merge")
        elif args.dry_run:
            LOGGER.info("Dry run: skipping merge")
        else:
            merged_path = output_root / merged_cfg.outputs.merged_filename
            stats["merged"] = merge_datasets(
                merged_cfg,
                tokenizer,
                inputs,
                merged_path,
            ).to_dict()

    stats_path = output_root / merged_cfg.outputs.stats_filename
    if stats:
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        LOGGER.info("Wrote stats → %s", stats_path)
    else:
        LOGGER.info("No stats to write (likely dry run or empty stages)")


if __name__ == "__main__":
    main()
