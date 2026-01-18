"""Sharded Arrow dataset utilities with resumable streaming iterators."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc


@dataclass
class ShardMeta:
    filename: str
    rows: int


class ShardedArrowDataset:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.directory}")
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found in {self.directory}. Please regenerate the dataset with sharding enabled."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.stage = manifest.get("stage", self.directory.name)
        shards = manifest.get("shards", [])
        if not shards:
            raise ValueError(f"Manifest {manifest_path} contains no shard entries")
        self.shards: List[ShardMeta] = [ShardMeta(s["filename"], int(s["rows"])) for s in shards]
        self.total_rows = sum(s.rows for s in self.shards)

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def load_shard(self, shard_index: int, seq_len: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
        meta = self.shards[shard_index]
        path = self.directory / meta.filename
        if not path.exists():
            raise FileNotFoundError(f"Shard file missing: {path}")
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_file(source)
            table = reader.read_all()
        ids_column = table.column("input_ids")
        lengths_column = table.column("length") if "length" in table.column_names else None

        ids_array = ids_column.combine_chunks()
        ids_type = ids_array.type
        if isinstance(ids_type, pa.FixedSizeListType) and ids_type.list_size == seq_len:
            values = ids_array.values.to_numpy(zero_copy_only=False)
            num_rows = len(ids_array)
            tokens = values.reshape(num_rows, seq_len).astype(np.int32, copy=False)
            if lengths_column is not None:
                lengths_arr = lengths_column.combine_chunks()
                lengths = np.asarray(lengths_arr.to_numpy(zero_copy_only=False), dtype=np.int32)
                lengths = np.clip(lengths, 1, seq_len)
            else:
                lengths = np.full(num_rows, seq_len, dtype=np.int32)
            return tokens, lengths

        seq_arrays = ids_column.to_pylist()
        num_rows = len(seq_arrays)
        pad_value = pad_id if pad_id is not None else 0
        tokens = np.full((num_rows, seq_len), pad_value, dtype=np.int32)
        lengths = np.zeros(num_rows, dtype=np.int32)

        for i, seq in enumerate(seq_arrays):
            if not isinstance(seq, list):
                seq = list(seq)
            if len(seq) >= seq_len:
                trimmed = seq[:seq_len]
                tokens[i] = np.asarray(trimmed, dtype=np.int32)
                lengths[i] = seq_len
            else:
                tokens[i, : len(seq)] = np.asarray(seq, dtype=np.int32)
                lengths[i] = len(seq)
        if lengths_column is not None:
            stored_lengths = np.asarray(lengths_column.to_pylist(), dtype=np.int32)
            stored_lengths = np.clip(stored_lengths, 1, seq_len)
            lengths = np.minimum(lengths, stored_lengths)
        lengths = np.clip(lengths, 1, seq_len)
        return tokens, lengths


class StageDataLoader:
    def __init__(
        self,
        dataset: ShardedArrowDataset,
        *,
        batch_size: int,
        seq_len: int,
        shuffle: bool,
        seed: int,
        pad_token_id: int,
        max_rows: Optional[int] = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        max_rows = max_rows if max_rows is not None else dataset.total_rows
        self.total_rows = min(max_rows, dataset.total_rows)
        if self.total_rows < batch_size:
            raise ValueError(
                f"Dataset too small for batch_size. total_rows={self.total_rows}, batch_size={batch_size}"
            )

        self.steps_per_epoch = self.total_rows // batch_size
        self.epoch = 0
        self.step_in_epoch = 0

        self._shard_order: List[int] = []
        self._current_shard_pos = 0
        self._current_data: Optional[np.ndarray] = None
        self._current_lengths: Optional[np.ndarray] = None
        self._row_ptr = 0
        self._rows_consumed = 0

        self._positions = np.arange(self.seq_len)[None, :]
        self._pad_col = np.full((self.batch_size, 1), self.pad_token_id, dtype=np.int32)

        self._prepare_epoch()

    def _epoch_seed(self) -> int:
        return self.seed + self.epoch * 1009

    def _prepare_epoch(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self._epoch_seed())
            self._shard_order = rng.permutation(self.dataset.num_shards).tolist()
        else:
            self._shard_order = list(range(self.dataset.num_shards))
        self._current_shard_pos = 0
        self._current_data = None
        self._current_lengths = None
        self._row_ptr = 0
        self._rows_consumed = 0

    def _load_shard_at(self, pos: int) -> bool:
        if self._rows_consumed >= self.total_rows:
            return False
        while pos < len(self._shard_order):
            shard_idx = self._shard_order[pos]
            data, lengths = self.dataset.load_shard(shard_idx, self.seq_len, self.pad_token_id)
            if data.shape[0] == 0:
                pos += 1
                continue
            remaining = self.total_rows - self._rows_consumed
            if remaining <= 0:
                return False
            if data.shape[0] > remaining:
                data = data[:remaining]
                lengths = lengths[:remaining]
            if self.shuffle:
                rng = np.random.default_rng(self._epoch_seed() ^ shard_idx)
                order = rng.permutation(data.shape[0])
                data = data[order]
                lengths = lengths[order]
            self._current_shard_pos = pos
            self._current_data = data
            self._current_lengths = lengths
            self._row_ptr = 0
            return True
        return False

    def _ensure_batch_available(self) -> None:
        while True:
            if (
                self._current_data is not None
                and self._row_ptr + self.batch_size <= self._current_data.shape[0]
            ):
                return
            next_pos = self._current_shard_pos if self._current_data is None else self._current_shard_pos + 1
            self._current_data = None
            self._current_lengths = None
            self._row_ptr = 0
            if not self._load_shard_at(next_pos):
                self.epoch += 1
                self.step_in_epoch = 0
                self._prepare_epoch()
                continue
            return

    def state_dict(self) -> Dict[str, int]:
        return {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "shard_pos": self._current_shard_pos,
            "row_ptr": self._row_ptr,
            "rows_consumed": self._rows_consumed,
        }

    def load_state(self, state: Dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.step_in_epoch = int(state.get("step_in_epoch", 0))
        shard_pos = int(state.get("shard_pos", 0))
        row_ptr = int(state.get("row_ptr", 0))
        saved_rows_consumed = int(state.get("rows_consumed", 0))
        self._prepare_epoch()
        # _prepare_epoch resets counters; restore consumed rows before loading shard.
        self._rows_consumed = saved_rows_consumed
        if not self._shard_order:
            raise RuntimeError("Dataset contains no shards")
        target_pos = min(max(shard_pos, 0), len(self._shard_order) - 1)
        if not self._load_shard_at(target_pos):
            raise RuntimeError("Failed to load shard while restoring state")
        limit = self._current_data.shape[0] if self._current_data is not None else 0
        self._row_ptr = min(max(row_ptr, 0), limit)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        if self.steps_per_epoch == 0:
            raise StopIteration
        if self.step_in_epoch >= self.steps_per_epoch:
            self.epoch += 1
            self.step_in_epoch = 0
            self._prepare_epoch()
        self._ensure_batch_available()
        data = self._current_data
        lengths = self._current_lengths
        if data is None or lengths is None:
            raise StopIteration
        start = self._row_ptr
        end = start + self.batch_size
        batch_tokens = data[start:end]
        batch_lengths = lengths[start:end]
        self._row_ptr = end
        self._rows_consumed += end - start
        self.step_in_epoch += 1

        seq_len = self.seq_len
        inputs = batch_tokens
        pad_col = self._pad_col.astype(batch_tokens.dtype, copy=False)
        targets = np.concatenate([batch_tokens[:, 1:], pad_col], axis=1)
        eff_lengths = np.clip(batch_lengths, 1, seq_len)
        valid_target_len = np.maximum(eff_lengths - 1, 0)
        positions = self._positions
        mask = (positions < valid_target_len[:, None]).astype(np.float32)

        return {"input": inputs, "target": targets, "mask": mask}


def save_dataloader_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    # Atomic rename avoids partial JSON reads after crashes.
    tmp_path.replace(path)


def load_dataloader_state(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "ShardedArrowDataset",
    "StageDataLoader",
    "save_dataloader_state",
    "load_dataloader_state",
]
