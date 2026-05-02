from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PackedDataset:
    encoder_tokens: np.ndarray
    decoder_input: np.ndarray
    decoder_target: np.ndarray
    decoder_mask: np.ndarray
    encoder_mask: np.ndarray
    pad_token_id: int


@dataclass
class PackedSplits:
    train: PackedDataset
    val: PackedDataset


def _load_split(data: np.lib.npyio.NpzFile, prefix: str) -> PackedDataset:
    return PackedDataset(
        encoder_tokens=data[f"{prefix}_encoder_tokens"],
        decoder_input=data[f"{prefix}_decoder_input"],
        decoder_target=data[f"{prefix}_decoder_target"],
        decoder_mask=data[f"{prefix}_decoder_mask"],
        encoder_mask=data[f"{prefix}_encoder_mask"],
        pad_token_id=int(data["pad_token_id"]),
    )


def load_packed_dataset(path: str | Path) -> PackedSplits:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file '{path}' not found")
    with np.load(path) as data:
        train = _load_split(data, "train")
        val = _load_split(data, "val")
    return PackedSplits(train=train, val=val)
