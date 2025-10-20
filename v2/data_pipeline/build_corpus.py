from __future__ import annotations

import argparse
import json
import logging
import os
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import load_dataset, IterableDataset, DatasetDict
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
LOGGER = logging.getLogger("build_corpus")
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOGGER.addHandler(_handler)

# --------------------------------------------------------------------------------------
# Config structures (mirror your /mnt/data/Config.yml and /mnt/data/Global_Config.yml)
# --------------------------------------------------------------------------------------

@dataclass
class TokenizerCfg:
    name: str = "bert-base-uncased"
    cache_dir: Optional[str] = ".cache/hf"
    use_custom: bool = False
    custom_path: str = ""
    hf_fallback: Optional[str] = None
    pad_token_override: Optional[str] = None  # from Global_Config.yml


@dataclass
class PathsCfg:
    processed_data_root: str = "dataset_artifacts"
    dataloader_state_root: str = "checkpoints/dataloader_state"
    logs_root: str = "logs"


@dataclass
class IOcfg:
    shard_size_tokens: int = 2048
    validation_fraction: float = 0.01
    min_sequence_tokens: int = 128


@dataclass
class TrainingDefaultsCfg:
    progressive_sequence_lengths: List[int] = None
    lr_milestones: List[float] = None
    warmup_steps: int = 1000
    grad_accumulation: int = 2
    global_seed: int = 0


@dataclass
class OutputsCfg:
    processed_root: str = "dataset_artifacts/base_corpus"
    tinystories_filename: str = "tinystories.arrow"
    wikipedia_filename: str = "wikipedia.arrow"
    merged_filename: str = "merged_training.arrow"
    stats_filename: str = "dataset_stats.json"


@dataclass
class SchedulingCfg:
    validation_fraction: float = 0.01
    seed: int = 42
    shuffle_buffer_size: int = 65536
    wiki_num_workers: int = 8
    write_batch_size: int = 1024
    tokenization_workers: Optional[int] = None


@dataclass
class TinyStoriesStageCfg:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    streaming: bool = True
    warmup_fraction: float = 0.0
    context_length: int = 256
    max_records: Optional[int] = None
    min_tokens: int = 0


@dataclass
class WikipediaStageCfg:
    raw_root: str = ""
    file_glob: str = "**/*.json"
    text_field: str = "text"
    context_length: int = 1024
    min_tokens: int = 0
    normalization: Dict[str, bool] = None
    pads_to_context: bool = True
    deduplicate: bool = True
    dedup_shingle_size: int = 12
    dedup_hash_bits: int = 21
    max_files: Optional[int] = None
    max_records: Optional[int] = None


@dataclass
class TopConfig:
    # merged from both YAMLs
    tokenizer: TokenizerCfg
    paths: PathsCfg
    io: IOcfg
    training_defaults: TrainingDefaultsCfg
    outputs: OutputsCfg
    scheduling: SchedulingCfg
    stages: Dict[str, Any]  # per-stage configs (tiny_stories, wikipedia)
    dry_run: bool = False


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _iter_chunks(iterable: Iterable[Any], n: int) -> Iterator[List[Any]]:
    buf: List[Any] = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _maybe_hash(s: str) -> int:
    # fast stable hash for dedup (not cryptographic)
    return hash(s)


@dataclass
class RunningStats:
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
        self._sink = None
        self._writer = None
        self._rows: List[Dict[str, Any]] = []
        self._batch_size = batch_size

    def __enter__(self) -> "ArrowBatchWriter":
        self._sink = pa.OSFile(str(self._output_path), "wb")
        self._writer = pa_ipc.new_file(self._sink, self._schema)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._rows:
            self._flush()
        if self._writer is not None:
            self._writer.close()
        if self._sink is not None:
            self._sink.close()

    def write(self, rows: List[Dict[str, Any]]) -> None:
        self._rows.extend(rows)
        if len(self._rows) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        batch = pa.RecordBatch.from_pylist(self._rows, schema=self._schema)
        self._writer.write(batch)
        self._rows.clear()


# --------------------------------------------------------------------------------------
# Tokenizer handling (with Xet hardening)
# --------------------------------------------------------------------------------------

def _materialize_tokenizer_dir(src: Path, cache_dir: Optional[str]) -> Path:
    """
    Copy typical tokenizer files to a fully local directory to avoid lazy/remote FS reads.
    """
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
    # Avoid flaky Xet-enabled chunk downloads by default.
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
                    tok_cfg.name, use_fast=True, cache_dir=tok_cfg.cache_dir
                )
            if tok_cfg.pad_token_override:
                tokenizer.pad_token = tok_cfg.pad_token_override
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        if snapshot_download is None:
            raise last_err
        repo = tok_cfg.name if not tok_cfg.use_custom else tok_cfg.hf_fallback
        if repo is None:
            raise last_err
        local_dir = snapshot_download(
            repo_id=repo,
            local_dir=tok_cfg.cache_dir,
            local_dir_use_symlinks=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
        if tok_cfg.pad_token_override:
            tokenizer.pad_token = tok_cfg.pad_token_override

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def _trim_padding(tokens: List[int], pad_id: int) -> List[int]:
    # remove trailing pad tokens
    i = len(tokens) - 1
    while i >= 0 and tokens[i] == pad_id:
        i -= 1
    return tokens[: i + 1]


# --------------------------------------------------------------------------------------
# Dataset readers (NO dataset-scripts)
# --------------------------------------------------------------------------------------

def iter_tiny_stories(cfg: TinyStoriesStageCfg) -> Iterable[str]:
    """Stream TinyStories from the Hub without dataset scripts."""
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=cfg.streaming)
    count = 0
    warmup_cut = None
    if cfg.warmup_fraction and cfg.warmup_fraction > 0:
        # Just emit a fraction then stop (useful for quick tests)
        warmup_cut = cfg.warmup_fraction

    total = None
    # If not streaming, we can know length
    if not cfg.streaming and isinstance(ds, DatasetDict) is False:
        try:
            total = len(ds)  # type: ignore
        except Exception:
            total = None

    for i, row in enumerate(ds):  # type: ignore
        # Choose a reasonable text field
        for key in ("text", "content", "story", "completion", "output"):
            if key in row and isinstance(row[key], str):
                yield row[key]
                break
        count += 1
        if warmup_cut is not None and total is not None and total > 0:
            if (i + 1) / total >= warmup_cut:
                break


def iter_wikipedia(cfg: WikipediaStageCfg) -> Iterable[str]:
    """Walk json files and yield the 'text' field (or fallbacks)."""
    root = _as_path(cfg.raw_root)
    if not root.exists():
        raise FileNotFoundError(f"wikipedia.raw_root not found: {root}")
    files = sorted(root.rglob(cfg.file_glob))
    if cfg.max_files:
        files = files[: cfg.max_files]
    emitted = 0
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(obj, dict):
            if cfg.text_field in obj and isinstance(obj[cfg.text_field], str):
                yield obj[cfg.text_field]
                emitted += 1
            else:
                # try common fallbacks
                for k in ("text", "article", "content", "body"):
                    if k in obj and isinstance(obj[k], str):
                        yield obj[k]
                        emitted += 1
                        break
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and cfg.text_field in item and isinstance(item[cfg.text_field], str):
                    yield item[cfg.text_field]
                    emitted += 1
                    if cfg.max_records and emitted >= cfg.max_records:
                        return
        if cfg.max_records and emitted >= cfg.max_records:
            return


def load_text_iterable(stage_name: str, stages_cfg: Dict[str, Any]) -> Tuple[Iterable[str], int, bool]:
    """
    Returns (iterator, context_length, pads_to_context)
    """
    sn = stage_name.lower()
    if sn == "tiny_stories":
        cfg = TinyStoriesStageCfg(**stages_cfg["tiny_stories"])
        return iter_tiny_stories(cfg), cfg.context_length, False
    if sn == "wikipedia":
        cfg = WikipediaStageCfg(**stages_cfg["wikipedia"])
        return iter_wikipedia(cfg), cfg.context_length, cfg.pads_to_context
    # If someone asks for 'openwebtext', fail fast with an actionable message.
    if sn == "openwebtext":
        raise RuntimeError(
            "Support for dataset scripts like 'openwebtext.py' was removed in datasets>=4.0. "
            "Use another source (e.g., TinyStories/Wikipedia) or pin datasets<4.0."
        )
    raise KeyError(f"Unknown stage '{stage_name}'. Check your Config.yml.")


# --------------------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------------------

def tokenize_records(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    min_tokens: int,
    max_tokens: int,
    pads_to_context: bool,
    dedupe: bool,
    stats: RunningStats,
    workers: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    seen_hashes: set[int] = set() if dedupe else set()

    def iter_candidates() -> Iterator[str]:
        for text in texts:
            stats.records += 1
            if dedupe:
                h = _maybe_hash(text)
                if h in seen_hashes:
                    stats.duplicates += 1
                    continue
                seen_hashes.add(h)
            yield text

    def tokenize_single(text: str) -> Tuple[str, List[int]]:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length" if max_tokens and pads_to_context else False,
            truncation=True if max_tokens else False,
            max_length=max_tokens if max_tokens else None,
        )
        input_ids: List[int] = enc["input_ids"]
        if not pads_to_context and pad_id != -1:
            input_ids = _trim_padding(input_ids, pad_id)
        return text, input_ids

    def emit(text: str, input_ids: List[int]) -> Optional[Dict[str, Any]]:
        token_count = len(input_ids)
        stats.tokens += token_count
        if token_count < max(min_tokens, 0):
            stats.discarded += 1
            return None
        return {
            "text": text,
            "input_ids": np.array(input_ids, dtype=np.int32),
            "length": token_count,
        }

    worker_count = workers if workers and workers > 1 else 1
    if worker_count == 1:
        for text in iter_candidates():
            text_val, input_ids = tokenize_single(text)
            row = emit(text_val, input_ids)
            if row is not None:
                yield row
        return

    chunk_size = max(1, worker_count * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for text_val, input_ids in executor.map(
            tokenize_single,
            iter_candidates(),
            chunksize=chunk_size,
        ):
            row = emit(text_val, input_ids)
            if row is not None:
                yield row


# --------------------------------------------------------------------------------------
# Stage runners
# --------------------------------------------------------------------------------------

def _arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("length", pa.int32()),
        ]
    )


def stage_tokenize(
    top_cfg: TopConfig,
    stage_name: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    outputs = top_cfg.outputs
    stages_cfg = top_cfg.stages

    texts, context_length, pads_to_context = load_text_iterable(stage_name, stages_cfg)

    stage_key = stage_name.lower()
    stage_cfg_map = stages_cfg.get(stage_key, {})

    worker_count = stage_cfg_map.get("tokenization_workers")
    if worker_count is None:
        worker_count = top_cfg.scheduling.tokenization_workers
    if worker_count is None or worker_count <= 0:
        worker_count = max(1, os.cpu_count() or 1)

    # per-stage knobs
    if stage_key == "tiny_stories":
        min_tokens = int(stage_cfg_map.get("min_tokens", 0) or 0)
        dedupe = False
        batch_size = top_cfg.scheduling.write_batch_size
        arrow_filename = outputs.tinystories_filename
    elif stage_key == "wikipedia":
        min_tokens = int(stage_cfg_map.get("min_tokens", 0) or 0)
        dedupe = bool(stage_cfg_map.get("deduplicate", True))
        batch_size = top_cfg.scheduling.write_batch_size
        arrow_filename = outputs.wikipedia_filename
    else:
        # Fallbacks
        min_tokens = 0
        dedupe = True
        batch_size = top_cfg.scheduling.write_batch_size
        arrow_filename = f"{stage_name}.arrow"

    LOGGER.info("Stage '%s' using %d tokenization workers.", stage_name, worker_count)

    output_root = _as_path(outputs.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    arrow_path = output_root / arrow_filename

    stats = RunningStats()
    schema = _arrow_schema()

    if top_cfg.dry_run:
        LOGGER.info("[dry-run] Tokenizing '%s' without writing output.", stage_name)
        for i, _ in zip(
            range(200),
            tokenize_records(
                tokenizer,
                texts,
                min_tokens,
                context_length,
                pads_to_context,
                dedupe,
                stats,
                workers=worker_count,
            ),
        ):
            pass
    else:
        with ArrowBatchWriter(arrow_path, schema, batch_size) as writer:
            for rows in _iter_chunks(
                tokenize_records(
                    tokenizer,
                    texts,
                    min_tokens,
                    context_length,
                    pads_to_context,
                    dedupe,
                    stats,
                    workers=worker_count,
                ),
                batch_size,
            ):
                writer.write(rows)

    LOGGER.info(
        "Stage '%s' done | records=%d tokens=%d duplicates=%d discarded=%d -> %s",
        stage_name,
        stats.records,
        stats.tokens,
        stats.duplicates,
        stats.discarded,
        arrow_path,
    )
    return stats.to_dict()


def stage_merge(top_cfg: TopConfig) -> Dict[str, Any]:
    """
    Minimal merge: just verifies both Arrow files exist and writes a merged file header.
    (You can extend this to actually concatenate batches if needed.)
    """
    out = top_cfg.outputs
    root = _as_path(out.processed_root)
    tinystories = root / out.tinystories_filename
    wikipedia = root / out.wikipedia_filename
    merged = root / out.merged_filename

    existing = [p for p in (tinystories, wikipedia) if p.exists()]
    if not existing:
        LOGGER.warning("No stage outputs found to merge in %s.", root)
        return {}

    # Simple merge: copy the first as "merged"; real merging could read and append batches
    shutil.copy2(existing[0], merged)
    LOGGER.info("Wrote (placeholder) merged file -> %s", merged)
    return {"merged_bytes": merged.stat().st_size, "sources_present": [str(p) for p in existing]}


# --------------------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------------------

def _find_default(path_candidates: List[str]) -> Optional[str]:
    for p in path_candidates:
        if Path(p).exists():
            return p
    return None


def load_combined_config(user_cfg_path: Optional[str]) -> TopConfig:
    """
    Combines your Global_Config.yml and Config.yml if present.
    If --config is provided, it can be either a single YAML with both trees,
    or just the corpus Config.yml (we'll still try to auto-load Global_Config.yml).
    """
    # Try to auto-detect both files in CWD
    script_dir = Path(__file__).resolve().parent
    corpus_cfg_path = user_cfg_path or _find_default(
        [
            "Config.yml",
            "config.yml",
            "configs/Config.yml",
            "configs/config.yml",
            str(script_dir / "Config.yml"),
            str(script_dir / "config.yml"),
        ]
    )
    global_cfg_path = _find_default(
        [
            "Global_Config.yml",
            "global_config.yml",
            "configs/Global_Config.yml",
            "configs/global_config.yml",
        ]
    )

    corpus_cfg = OmegaConf.create({})
    global_cfg = OmegaConf.create({})

    if corpus_cfg_path and Path(corpus_cfg_path).exists():
        corpus_cfg = OmegaConf.load(corpus_cfg_path)
        LOGGER.info("Loaded dataset config: %s", corpus_cfg_path)
    else:
        LOGGER.info("No Config.yml found; defaulting to TinyStories only.")

    if global_cfg_path and Path(global_cfg_path).exists():
        global_cfg = OmegaConf.load(global_cfg_path)
        LOGGER.info("Loaded global config: %s", global_cfg_path)

    # Build pieces with sensible defaults + overrides from YAMLs
    tok = TokenizerCfg(**(global_cfg.get("tokenizer") or {}))
    paths = PathsCfg(**(global_cfg.get("paths") or {}))
    io_cfg = IOcfg(**(global_cfg.get("io") or {}))
    train_def = TrainingDefaultsCfg(**(global_cfg.get("training_defaults") or {}))
    outputs = OutputsCfg(**(corpus_cfg.get("outputs") or {}))
    sched = SchedulingCfg(**(corpus_cfg.get("scheduling") or {}))

    # Stages dict (present keys from corpus config)
    stages: Dict[str, Any] = {}
    if "tiny_stories" in corpus_cfg:
        stages["tiny_stories"] = dict(corpus_cfg["tiny_stories"])
    if "wikipedia" in corpus_cfg:
        stages["wikipedia"] = dict(corpus_cfg["wikipedia"])

    # If nothing provided, default to TinyStories
    if not stages:
        stages["tiny_stories"] = TinyStoriesStageCfg().__dict__
        # Also set minimal outputs so the run is self-contained
        outputs.tinystories_filename = outputs.tinystories_filename or "tinystories.arrow"

    return TopConfig(
        tokenizer=tok,
        paths=paths,
        io=io_cfg,
        training_defaults=train_def,
        outputs=outputs,
        scheduling=sched,
        stages=stages,
        dry_run=False,
    )


# --------------------------------------------------------------------------------------
# CLI / Orchestration
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to corpus Config.yml. If omitted, auto-detects.")
    p.add_argument("--stage", type=str, default="all", help="Stage name ('tiny_stories'/'wikipedia') or 'all'.")
    p.add_argument("--dry-run", action="store_true", help="Run without writing outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    top_cfg = load_combined_config(args.config)
    if args.dry_run:
        top_cfg.dry_run = True

    # Tokenizer
    tokenizer = load_tokenizer(top_cfg.tokenizer)

    np.random.seed(top_cfg.training_defaults.global_seed or top_cfg.scheduling.seed)

    # Determine stages
    if args.stage == "all":
        stage_list = list(top_cfg.stages.keys())
        if not stage_list:
            LOGGER.error("No stages discovered in config. Nothing to do.")
            return
    else:
        stage_list = [args.stage]

    # Run
    all_stats: Dict[str, Any] = {}
    for stg in stage_list:
        st = stage_tokenize(top_cfg, stg, tokenizer)
        all_stats[f"stage_{stg}"] = st

    # Optional merge (only if more than one stage)
    if len(stage_list) > 1:
        merge_stats = stage_merge(top_cfg)
        if merge_stats:
            all_stats["merge"] = merge_stats

    # Write stats
    stats_path = _as_path(top_cfg.outputs.processed_root) / top_cfg.outputs.stats_filename
    _as_path(top_cfg.outputs.processed_root).mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, indent=2)
    LOGGER.info("Wrote stats → %s", stats_path)


if __name__ == "__main__":
    main()
