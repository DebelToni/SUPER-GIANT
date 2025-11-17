from __future__ import annotations

import argparse
import bz2
import gzip
import json
import logging
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cleaning import normalise_text

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None


LOGGER = logging.getLogger("build_corpus")
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOGGER.addHandler(_handler)


DEFAULT_TEXT_FIELDS = ("text", "content", "body", "article", "story", "completion")


@dataclass
class TokenizerCfg:
    name: str = "bert-base-uncased"
    cache_dir: Optional[str] = ".cache/hf"
    use_custom: bool = False
    custom_path: str = ""
    hf_fallback: Optional[str] = None
    pad_token_override: Optional[str] = None


@dataclass
class PathsCfg:
    data_root: str = ""
    processed_data_root: str = "dataset_artifacts"
    dataloader_state_root: str = "checkpoints/dataloader_state"
    logs_root: str = "logs"


@dataclass
class TrainingDefaultsCfg:
    progressive_sequence_lengths: Optional[List[int]] = None
    lr_milestones: Optional[List[float]] = None
    warmup_steps: int = 0
    grad_accumulation: int = 1
    global_seed: int = 0


@dataclass
class OutputsCfg:
    processed_root: str = "dataset_artifacts"
    rows_per_shard: int = 65536
    manifest_filename: str = "datasets_manifest.json"
    stats_filename: str = "dataset_stats.json"


@dataclass
class SchedulingCfg:
    seed: int = 4212
    write_batch_size: int = 256
    dry_run_preview_rows: int = 200


@dataclass
class StageSourceCfg:
    type: str = "huggingface"
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    data_files: Optional[Any] = None
    text_field: Optional[str] = None
    text_fields: List[str] = field(default_factory=list)
    join_fields: List[str] = field(default_factory=list)
    join_separator: str = " \n"
    text_template: Optional[str] = None
    json_root: Optional[str] = None
    file_glob: str = "**/*.json"
    max_documents: Optional[int] = None


@dataclass
class StageCfg:
    name: str = ""
    description: str = ""
    output_dir: Optional[str] = None
    sequence_length: int = 512
    target_tokens: Optional[int] = None
    target_sequences: Optional[int] = None
    min_tokens: int = 0
    pack_sequences: bool = True
    add_eos: bool = True
    emit_final_partial: bool = True
    long_document_strategy: str = "random_window"
    sequential_window_stride: Optional[int] = None
    max_windows_per_document: Optional[int] = None
    drop_remainder_windows: bool = False
    normalization: Dict[str, Any] = field(default_factory=dict)
    deduplicate: bool = False
    dedup_hash_bits: int = 21
    dedup_max_keys: Optional[int] = None
    rows_per_shard: Optional[int] = None
    max_documents: Optional[int] = None
    seed_offset: Optional[int] = None
    sources: List[StageSourceCfg] = field(default_factory=list)


@dataclass
class TopConfig:
    tokenizer: TokenizerCfg
    paths: PathsCfg
    training_defaults: TrainingDefaultsCfg
    outputs: OutputsCfg
    scheduling: SchedulingCfg
    stages: List[StageCfg]
    dry_run: bool = False


@dataclass
class StageStats:
    documents: int = 0
    unique_documents: int = 0
    duplicates: int = 0
    discarded: int = 0
    sequences: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "documents": self.documents,
            "unique_documents": self.unique_documents,
            "duplicates": self.duplicates,
            "discarded": self.discarded,
            "sequences": self.sequences,
            "tokens": self.tokens,
        }


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _resolve_path(base: Optional[str], value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if base and not path.is_absolute():
        return str(Path(base) / path)
    return str(path)


def _open_text_stream(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _stream_json_records(path: Path, chunk_size: int = 65536) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    with _open_text_stream(path) as handle:
        for chunk in iter(lambda: handle.read(chunk_size), ""):
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] in ",]":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "[":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = buffer[idx:]
    buffer = buffer.lstrip()
    if buffer:
        try:
            obj, _ = decoder.raw_decode(buffer)
            yield obj
        except json.JSONDecodeError:
            pass


def _maybe_hash(text: str) -> int:
    return hash(text)


class ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, schema: pa.Schema, rows_per_shard: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.schema = schema
        self.rows_per_shard = max(1, rows_per_shard)
        self._buffer: List[Dict[str, Any]] = []
        self._shard_index = 0
        self.manifest: List[Dict[str, Any]] = []

    def append(self, rows: List[Dict[str, Any]]) -> None:
        self._buffer.extend(rows)
        while len(self._buffer) >= self.rows_per_shard:
            self._flush_chunk(self.rows_per_shard)

    def close(self) -> None:
        if self._buffer:
            self._flush_chunk(len(self._buffer))

    def _flush_chunk(self, count: int) -> None:
        chunk = self._buffer[:count]
        del self._buffer[:count]
        filename = f"{self.prefix}-{self._shard_index:06d}.arrow"
        path = self.output_dir / filename
        table = pa.Table.from_pylist(chunk, schema=self.schema)
        with pa.OSFile(str(path), "wb") as sink:
            with pa_ipc.new_file(sink, self.schema) as writer:
                writer.write_table(table)
        self.manifest.append({"filename": filename, "rows": count})
        self._shard_index += 1


def _materialize_tokenizer_dir(src: Path, cache_dir: Optional[str]) -> Path:
    dst = Path(cache_dir or ".cache") / "tokenizer.materialized"
    dst.mkdir(parents=True, exist_ok=True)
    needed = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
    ]
    any_copied = False
    for name in needed:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
            any_copied = True
    if not any_copied:
        raise FileNotFoundError(f"No tokenizer files found under {src}")
    return dst


def load_tokenizer(tok_cfg: TokenizerCfg) -> PreTrainedTokenizerBase:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    last_err: Optional[BaseException] = None
    for attempt in range(3):
        try:
            if tok_cfg.use_custom:
                src = Path(tok_cfg.custom_path)
                local_dir = _materialize_tokenizer_dir(src, tok_cfg.cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tok_cfg.name,
                    use_fast=True,
                    cache_dir=tok_cfg.cache_dir,
                )
            if tok_cfg.pad_token_override:
                tokenizer.pad_token = tok_cfg.pad_token_override
            break
        except Exception as exc:  # pragma: no cover - retry path
            last_err = exc
            time_wait = 1.5 * (attempt + 1)
            LOGGER.warning("Tokenizer load failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(time_wait)
    else:
        if snapshot_download is None:
            raise last_err
        repo = tok_cfg.name if not tok_cfg.use_custom else tok_cfg.hf_fallback
        if repo is None:
            raise last_err
        local_dir = snapshot_download(repo_id=repo, local_dir=tok_cfg.cache_dir, local_dir_use_symlinks=False)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
        if tok_cfg.pad_token_override:
            tokenizer.pad_token = tok_cfg.pad_token_override

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


class SequenceEmitter:
    def __init__(
        self,
        stage: StageCfg,
        tokenizer: PreTrainedTokenizerBase,
        rng: np.random.Generator,
        stats: StageStats,
    ) -> None:
        self.stage = stage
        self.tokenizer = tokenizer
        self.rng = rng
        self.stats = stats
        self.seq_len = int(stage.sequence_length)
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.sep_token_id
        self.eos_id = eos if eos is not None else 0
        self.pack_buffer: List[int] = []
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def _check_targets(self) -> None:
        target_tokens = self.stage.target_tokens
        target_sequences = self.stage.target_sequences
        if target_tokens is not None and self.stats.tokens >= target_tokens:
            self._done = True
        if target_sequences is not None and self.stats.sequences >= target_sequences:
            self._done = True

    def encode_batch(self, texts: List[str]) -> Iterator[Dict[str, Any]]:
        if not texts:
            return
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        for tokens in encoded:
            if self.done:
                return
            yield from self.consume_tokens(tokens)
            if self.done:
                return

    def consume_tokens(self, tokens: Iterable[int]) -> Iterator[Dict[str, Any]]:
        ids = [int(t) for t in tokens]
        if len(ids) < self.stage.min_tokens:
            self.stats.discarded += 1
            return
        if len(ids) >= self.seq_len:
            yield from self._emit_long(ids)
        else:
            yield from self._emit_short(ids)

    def _emit_long(self, tokens: List[int]) -> Iterator[Dict[str, Any]]:
        strategy = (self.stage.long_document_strategy or "random_window").lower()
        seq_len = self.seq_len
        length = len(tokens)
        if strategy == "sequential":
            stride = int(self.stage.sequential_window_stride or seq_len)
            max_windows = int(self.stage.max_windows_per_document or 0)
            emitted = 0
            offset = 0
            while offset < length:
                window = tokens[offset : offset + seq_len]
                if not window:
                    break
                if len(window) < self.stage.min_tokens:
                    break
                row = self._emit_sequence(window)
                if row:
                    yield row
                emitted += 1
                if self.done:
                    return
                if max_windows and emitted >= max_windows:
                    return
                if len(window) < seq_len:
                    if self.stage.drop_remainder_windows:
                        return
                    break
                offset += stride
            return

        max_start = max(0, length - seq_len)
        start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        window = tokens[start : start + seq_len]
        row = self._emit_sequence(window)
        if row:
            yield row

    def _emit_short(self, tokens: List[int]) -> Iterator[Dict[str, Any]]:
        if self.stage.pack_sequences:
            seq = list(tokens)
            if self.stage.add_eos:
                seq.append(self.eos_id)
            self.pack_buffer.extend(seq)
            while len(self.pack_buffer) >= self.seq_len:
                chunk = self.pack_buffer[: self.seq_len]
                del self.pack_buffer[: self.seq_len]
                row = self._emit_sequence(chunk)
                if row:
                    yield row
                if self.done:
                    return
        else:
            seq = list(tokens)
            if self.stage.add_eos and len(seq) < self.seq_len:
                seq.append(self.eos_id)
            row = self._emit_sequence(seq)
            if row:
                yield row

    def flush_remainder(self) -> Iterator[Dict[str, Any]]:
        if not self.stage.pack_sequences or not self.stage.emit_final_partial:
            return
        if not self.pack_buffer or self.done:
            return
        chunk = list(self.pack_buffer)
        self.pack_buffer.clear()
        row = self._emit_sequence(chunk)
        if row:
            yield row

    def _emit_sequence(self, seq: List[int]) -> Optional[Dict[str, Any]]:
        if not seq:
            return None
        if len(seq) > self.seq_len:
            seq = seq[: self.seq_len]
        length = len(seq)
        self.stats.sequences += 1
        self.stats.tokens += length
        row = {"input_ids": seq, "length": length}
        self._check_targets()
        return row


def _extract_text(row: Dict[str, Any], source: StageSourceCfg) -> Optional[str]:
    if source.text_template:
        safe_map = defaultdict(str)
        for key, value in row.items():
            if isinstance(value, (str, int, float)):
                safe_map[key] = value
        try:
            text = source.text_template.format_map(safe_map)
        except KeyError:
            text = None
        if text:
            return str(text)

    if source.join_fields:
        parts = []
        for key in source.join_fields:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        if parts:
            return source.join_separator.join(parts)

    candidates: List[str] = []
    if source.text_field:
        candidates.append(source.text_field)
    candidates.extend(source.text_fields)
    keys = candidates or list(DEFAULT_TEXT_FIELDS)
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_hf_source(source: StageSourceCfg) -> Iterator[str]:
    if not source.dataset_name:
        raise ValueError("HuggingFace source requires 'dataset_name'.")
    kwargs: Dict[str, Any] = {}
    if source.dataset_config:
        kwargs["name"] = source.dataset_config
    if source.data_files is not None:
        kwargs["data_files"] = source.data_files
    ds = load_dataset(
        source.dataset_name,
        split=source.split,
        streaming=source.streaming,
        **kwargs,
    )
    count = 0
    for row in ds:
        text = _extract_text(row, source) if isinstance(row, dict) else None
        if text:
            yield text
            count += 1
            if source.max_documents and count >= source.max_documents:
                break


def _iter_json_dir(source: StageSourceCfg) -> Iterator[str]:
    if not source.json_root:
        raise ValueError("json_dir source requires 'json_root'.")
    root = _as_path(source.json_root)
    files = sorted(root.rglob(source.file_glob))
    count = 0
    for path in files:
        for record in _stream_json_records(path):
            if not isinstance(record, dict):
                continue
            text = _extract_text(record, source)
            if text:
                yield text
                count += 1
                if source.max_documents and count >= source.max_documents:
                    return


def iter_stage_text(stage: StageCfg) -> Iterator[str]:
    for source in stage.sources:
        source_type = (source.type or "huggingface").lower()
        if source_type in {"hf", "huggingface"}:
            yield from _iter_hf_source(source)
        elif source_type in {"json", "jsonl", "json_dir"}:
            yield from _iter_json_dir(source)
        else:
            raise ValueError(f"Unknown source.type '{source.type}' for stage {stage.name}")


def iter_stage_rows(
    stage: StageCfg,
    tokenizer: PreTrainedTokenizerBase,
    rng: np.random.Generator,
    stats: StageStats,
    batch_size: int,
) -> Iterator[Dict[str, Any]]:
    emitter = SequenceEmitter(stage, tokenizer, rng, stats)
    norm = SimpleNamespace(**(stage.normalization or {}))
    seen_hashes: set[int] = set()
    dedupe_queue: Optional[deque[int]] = None
    dedup_mask = None
    if stage.deduplicate:
        if stage.dedup_hash_bits:
            dedup_mask = (1 << int(stage.dedup_hash_bits)) - 1
        if stage.dedup_max_keys:
            dedupe_queue = deque()

    batch: List[str] = []
    doc_limit = stage.max_documents
    for raw_text in iter_stage_text(stage):
        if emitter.done:
            break
        if doc_limit and stats.documents >= doc_limit:
            break
        if not isinstance(raw_text, str):
            continue
        stats.documents += 1
        text = raw_text.strip()
        if stage.normalization:
            text = normalise_text(text, norm)
        text = text.strip()
        if not text:
            stats.discarded += 1
            continue
        if stage.deduplicate:
            h = _maybe_hash(text)
            if dedup_mask is not None:
                h &= dedup_mask
            if h in seen_hashes:
                stats.duplicates += 1
                continue
            seen_hashes.add(h)
            if dedupe_queue is not None:
                dedupe_queue.append(h)
                if stage.dedup_max_keys and len(dedupe_queue) > stage.dedup_max_keys:
                    old = dedupe_queue.popleft()
                    if old in seen_hashes:
                        seen_hashes.remove(old)
        stats.unique_documents += 1
        batch.append(text)
        if len(batch) >= batch_size:
            yield from emitter.encode_batch(batch)
            batch.clear()
    if batch and not emitter.done:
        yield from emitter.encode_batch(batch)
    if not emitter.done:
        yield from emitter.flush_remainder()


def _arrow_schema() -> pa.Schema:
    return pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("length", pa.int32()),
    ])


def stage_tokenize(top_cfg: TopConfig, stage: StageCfg, tokenizer: PreTrainedTokenizerBase, index: int) -> Dict[str, Any]:
    rows_per_shard = int(stage.rows_per_shard or top_cfg.outputs.rows_per_shard)
    output_root = _as_path(top_cfg.outputs.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_dir_name = stage.output_dir or stage.name
    stage_output_dir = output_root / stage_dir_name
    if not top_cfg.dry_run:
        if stage_output_dir.exists():
            shutil.rmtree(stage_output_dir)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

    rng_seed = (top_cfg.scheduling.seed or 0) + (stage.seed_offset or 0) + index * 7919
    rng = np.random.default_rng(rng_seed)
    stats = StageStats()
    record_stream = iter_stage_rows(
        stage,
        tokenizer,
        rng,
        stats,
        batch_size=max(1, top_cfg.scheduling.write_batch_size),
    )

    manifest: List[Dict[str, Any]] = []
    shard_writer: Optional[ShardWriter] = None
    if not top_cfg.dry_run:
        shard_writer = ShardWriter(stage_output_dir, stage_dir_name, _arrow_schema(), rows_per_shard)

    preview = top_cfg.scheduling.dry_run_preview_rows
    if top_cfg.dry_run:
        count = 0
        for _ in record_stream:
            count += 1
            if preview and count >= preview:
                break
        LOGGER.info("[dry-run] Stage '%s' previewed %d sequences", stage.name, count)
    else:
        assert shard_writer is not None
        batch_buffer: List[Dict[str, Any]] = []
        for row in record_stream:
            batch_buffer.append(row)
            if len(batch_buffer) >= top_cfg.scheduling.write_batch_size:
                shard_writer.append(batch_buffer)
                batch_buffer = []
        if batch_buffer:
            shard_writer.append(batch_buffer)
        shard_writer.close()
        manifest = shard_writer.manifest
        with (stage_output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": stage.name,
                    "description": getattr(stage, "description", ""),
                    "target_tokens": stage.target_tokens,
                    "total_records": stats.sequences,
                    "total_tokens": stats.tokens,
                    "documents": stats.documents,
                    "shards": manifest,
                },
                handle,
                indent=2,
            )

    LOGGER.info(
        "Stage '%s' done | documents=%d sequences=%d tokens=%d duplicates=%d discarded=%d",
        stage.name,
        stats.documents,
        stats.sequences,
        stats.tokens,
        stats.duplicates,
        stats.discarded,
    )
    stage_stats = stats.to_dict()
    stage_stats["target_tokens"] = stage.target_tokens
    stage_stats["sequence_length"] = stage.sequence_length
    stage_stats["output_dir"] = str(stage_output_dir)
    if not top_cfg.dry_run:
        stage_stats["shards"] = manifest
    return stage_stats


def stage_merge(top_cfg: TopConfig, stages: List[StageCfg]) -> Dict[str, Any]:
    root = _as_path(top_cfg.outputs.processed_root)
    manifest = {}
    for stage in stages:
        stage_dir = root / (stage.output_dir or stage.name)
        entry = {
            "path": str(stage_dir),
            "sequence_length": stage.sequence_length,
            "target_tokens": stage.target_tokens,
        }
        manifest[stage.name] = entry
    manifest_path = root / top_cfg.outputs.manifest_filename
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    LOGGER.info("Wrote dataset manifest → %s", manifest_path)
    return manifest


def _find_default(path_candidates: List[str]) -> Optional[str]:
    for p in path_candidates:
        if Path(p).exists():
            return p
    return None


def _parse_stage_sources(raw_sources: Iterable[Any]) -> List[StageSourceCfg]:
    sources: List[StageSourceCfg] = []
    for src in raw_sources:
        if src is None:
            continue
        src_dict = dict(src)
        cfg = StageSourceCfg(
            type=src_dict.get("type", "huggingface"),
            dataset_name=src_dict.get("dataset_name"),
            dataset_config=src_dict.get("dataset_config"),
            split=src_dict.get("split", "train"),
            streaming=bool(src_dict.get("streaming", True)),
            data_files=src_dict.get("data_files"),
            text_field=src_dict.get("text_field"),
            text_fields=list(src_dict.get("text_fields", []) or []),
            join_fields=list(src_dict.get("join_fields", []) or []),
            join_separator=str(src_dict.get("join_separator", " \n")),
            text_template=src_dict.get("text_template"),
            json_root=src_dict.get("json_root"),
            file_glob=src_dict.get("file_glob", "**/*.json"),
            max_documents=src_dict.get("max_documents"),
        )
        sources.append(cfg)
    return sources


def _parse_stages(corpus_cfg: OmegaConf, outputs: OutputsCfg) -> List[StageCfg]:
    if "stages" not in corpus_cfg or corpus_cfg.get("stages") is None:
        raise ValueError("No 'stages' section found in Config.yml")
    stage_map = OmegaConf.to_container(corpus_cfg["stages"], resolve=True)
    stages: List[StageCfg] = []
    for name, raw_stage in stage_map.items():
        raw_dict = dict(raw_stage or {})
        raw_dict.setdefault("name", name)
        raw_dict.setdefault("output_dir", raw_dict.get("output_dir", name))
        raw_dict["sources"] = _parse_stage_sources(raw_dict.get("sources", []))
        stage_cfg = StageCfg(**raw_dict)
        stage_cfg.sequence_length = int(stage_cfg.sequence_length)
        stage_cfg.min_tokens = int(stage_cfg.min_tokens or 0)
        stage_cfg.rows_per_shard = int(stage_cfg.rows_per_shard or outputs.rows_per_shard)
        if stage_cfg.target_tokens is not None:
            stage_cfg.target_tokens = int(stage_cfg.target_tokens)
        if stage_cfg.target_sequences is not None:
            stage_cfg.target_sequences = int(stage_cfg.target_sequences)
        if stage_cfg.dedup_max_keys is not None:
            stage_cfg.dedup_max_keys = int(stage_cfg.dedup_max_keys)
        if not stage_cfg.sources:
            raise ValueError(f"Stage '{stage_cfg.name}' has no sources defined")
        stages.append(stage_cfg)
    return stages


def load_combined_config(user_cfg_path: Optional[str]) -> TopConfig:
    script_dir = Path(__file__).resolve().parent
    corpus_cfg_path = user_cfg_path or _find_default(
        [
            "Config.yml",
            "config.yml",
            str(script_dir / "Config.yml"),
            str(script_dir / "config.yml"),
        ]
    )
    global_cfg_path = _find_default([
        "Global_Config.yml",
        "global_config.yml",
    ])

    corpus_cfg = OmegaConf.create({})
    global_cfg = OmegaConf.create({})

    if corpus_cfg_path and Path(corpus_cfg_path).exists():
        corpus_cfg = OmegaConf.load(corpus_cfg_path)
        LOGGER.info("Loaded dataset config: %s", corpus_cfg_path)
    else:
        raise FileNotFoundError("Config.yml not found; cannot build datasets.")

    if global_cfg_path and Path(global_cfg_path).exists():
        global_cfg = OmegaConf.load(global_cfg_path)
        LOGGER.info("Loaded global config: %s", global_cfg_path)

    tok = TokenizerCfg(**(global_cfg.get("tokenizer") or {}))
    paths = PathsCfg(**(global_cfg.get("paths") or {}))
    train_defaults = TrainingDefaultsCfg(**(global_cfg.get("training_defaults") or {}))
    outputs = OutputsCfg(**(corpus_cfg.get("outputs") or {}))
    scheduling = SchedulingCfg(**(corpus_cfg.get("scheduling") or {}))
    stages = _parse_stages(corpus_cfg, outputs)

    base_prefix = paths.data_root or ""
    if base_prefix:
        base_prefix = str(Path(base_prefix))
        paths.data_root = base_prefix
    paths.processed_data_root = _resolve_path(base_prefix, paths.processed_data_root) or paths.processed_data_root
    paths.dataloader_state_root = _resolve_path(base_prefix, paths.dataloader_state_root) or paths.dataloader_state_root
    if paths.logs_root:
        paths.logs_root = _resolve_path(base_prefix, paths.logs_root)
    outputs.processed_root = _resolve_path(base_prefix, outputs.processed_root) or outputs.processed_root
    for stage in stages:
        for source in stage.sources:
            if source.json_root:
                source.json_root = _resolve_path(base_prefix, source.json_root)

    return TopConfig(
        tokenizer=tok,
        paths=paths,
        training_defaults=train_defaults,
        outputs=outputs,
        scheduling=scheduling,
        stages=stages,
        dry_run=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to Config.yml (optional)")
    parser.add_argument("--stage", type=str, default="all", help="Stage name or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Tokenize without writing shards")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_cfg = load_combined_config(args.config)
    if args.dry_run:
        top_cfg.dry_run = True

    tokenizer = load_tokenizer(top_cfg.tokenizer)
    np.random.seed(top_cfg.training_defaults.global_seed or top_cfg.scheduling.seed)

    stage_lookup = {stage.name: stage for stage in top_cfg.stages}
    if args.stage != "all":
        key = args.stage.strip()
        if key not in stage_lookup:
            available = ", ".join(stage_lookup.keys())
            raise KeyError(f"Stage '{key}' not found. Available stages: {available}")
        stages_to_run = [stage_lookup[key]]
    else:
        stages_to_run = list(top_cfg.stages)

    stats_bundle: Dict[str, Any] = {}
    for idx, stage in enumerate(stages_to_run):
        stats_bundle[stage.name] = stage_tokenize(top_cfg, stage, tokenizer, idx)

    if not top_cfg.dry_run and stages_to_run:
        manifest = stage_merge(top_cfg, stages_to_run)
        stats_bundle["manifest"] = manifest

    stats_path = _as_path(top_cfg.outputs.processed_root) / top_cfg.outputs.stats_filename
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_bundle, handle, indent=2)
    LOGGER.info("Wrote stats → %s", stats_path)


if __name__ == "__main__":
    import time

    main()
