"""Arrow dataset utilities with resumable stage iterators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc


class ArrowDataset:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        self._load()

    def _load(self) -> None:
        with pa.memory_map(str(self.path), "r") as source:
            reader = pa_ipc.open_file(source)
            table = reader.read_all()
        tokens_list = table.column("input_ids").to_pylist()
        self.tokens = np.asarray(tokens_list, dtype=np.int32)
        self.seq_lengths = np.asarray(table.column("seq_length").to_pylist(), dtype=np.int32)
        self.context_lengths = np.asarray(table.column("context_length").to_pylist(), dtype=np.int32)
        self.sources = table.column("source").to_pylist()
        self.doc_ids = table.column("document_id").to_pylist()
        self.window_indices = np.asarray(table.column("window_index").to_pylist(), dtype=np.int32)

    @property
    def num_rows(self) -> int:
        return self.tokens.shape[0]


class StageDataLoader:
    def __init__(
        self,
        dataset: ArrowDataset,
        *,
        batch_size: int,
        seq_len: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

        self.steps_per_epoch = dataset.num_rows // batch_size
        if self.steps_per_epoch == 0:
            raise ValueError("Dataset too small for the specified batch_size")

        self.epoch = 0
        self.step_in_epoch = 0
        self._set_order()

    def _set_order(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(self.dataset.num_rows)
        else:
            perm = np.arange(self.dataset.num_rows)
        usable = self.steps_per_epoch * self.batch_size
        self.order = perm[:usable]

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "step_in_epoch": self.step_in_epoch}

    def load_state(self, state: Dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.step_in_epoch = int(state.get("step_in_epoch", 0))
        self._set_order()

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        if self.step_in_epoch >= self.steps_per_epoch:
            self.epoch += 1
            self.step_in_epoch = 0
            self._set_order()
        start = self.step_in_epoch * self.batch_size
        end = start + self.batch_size
        idxs = self.order[start:end]
        self.step_in_epoch += 1

        tokens = self.dataset.tokens[idxs, : self.seq_len]
        seq_len = self.seq_len
        eff_lengths = np.minimum(self.dataset.seq_lengths[idxs], seq_len)
        # Ensure at least length 2 for shift; shorter sequences were filtered earlier
        eff_lengths = np.maximum(eff_lengths, 2)

        inputs = tokens[:, :seq_len - 1]
        targets = tokens[:, 1:seq_len]
        mask = np.zeros_like(inputs, dtype=np.float32)
        positions = np.arange(seq_len - 1)[None, :]
        valid = (positions < (eff_lengths[:, None] - 1))
        mask[valid] = 1.0

        return {"input": inputs, "target": targets, "mask": mask}


def save_dataloader_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def load_dataloader_state(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["ArrowDataset", "StageDataLoader", "save_dataloader_state", "load_dataloader_state"]
