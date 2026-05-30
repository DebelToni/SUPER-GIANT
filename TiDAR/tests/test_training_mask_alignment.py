from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from GIANT.v3.data_pipeline.build_corpus import _arrow_schema, _pack_loss_mask
from GIANT.v3.model.arrow_data_loader import ShardedArrowDataset, StageDataLoader
from TiDAR.model.tidar_utils import build_train_batch


def _write_dataset(root: Path, seq_len: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "input_ids": [10, 11, 12, 13, 14, 15],
            "length": 6,
            "loss_mask": _pack_loss_mask([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], seq_len),
        },
        {
            "input_ids": [20, 21, 22, 0, 0, 0],
            "length": 3,
            "loss_mask": _pack_loss_mask([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], seq_len),
        },
    ]
    schema = _arrow_schema(seq_len)
    with pa.OSFile(str(root / "tiny-000000.arrow"), "wb") as sink:
        with pa_ipc.new_file(sink, schema) as writer:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"stage": "tiny", "shards": [{"filename": "tiny-000000.arrow", "rows": len(rows)}]}, handle)


def test_tidar_uses_raw_token_masks() -> None:
    seq_len = 6
    dataset_root = Path(tempfile.mkdtemp()) / "dataset"
    _write_dataset(dataset_root, seq_len)
    dataset = ShardedArrowDataset(dataset_root)

    raw_loader = StageDataLoader(
        dataset,
        batch_size=2,
        seq_len=seq_len,
        shuffle=False,
        seed=0,
        pad_token_id=0,
        mask_target_shift=False,
    )
    raw_batch = next(iter(raw_loader))
    assert raw_batch["mask"].tolist() == [[0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0]]
    assert raw_batch["length"].tolist() == [6, 3]

    train_batch = build_train_batch(
        jnp.asarray(raw_batch["input"]),
        jnp.asarray(raw_batch["length"], dtype=jnp.int32),
        mask_id=999,
        block_len=2,
        token_mask=jnp.asarray(raw_batch["mask"]),
    )
    ntp_mask = np.asarray(train_batch["loss_mask_ntp"])[:, :seq_len].astype(int).tolist()
    diff_mask = np.asarray(train_batch["loss_mask_diff"])[:, seq_len:].astype(int).tolist()

    assert ntp_mask == [[0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0]]
    assert diff_mask == [[0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0]]


def test_causal_target_shift_is_wrong_for_tidar() -> None:
    seq_len = 6
    dataset_root = Path(tempfile.mkdtemp()) / "dataset"
    _write_dataset(dataset_root, seq_len)
    dataset = ShardedArrowDataset(dataset_root)

    causal_loader = StageDataLoader(
        dataset,
        batch_size=2,
        seq_len=seq_len,
        shuffle=False,
        seed=0,
        pad_token_id=0,
    )
    shifted_batch = next(iter(causal_loader))
    assert shifted_batch["mask"].tolist() == [[0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0]]

    train_batch = build_train_batch(
        jnp.asarray(shifted_batch["input"]),
        jnp.asarray(shifted_batch["length"], dtype=jnp.int32),
        mask_id=999,
        block_len=2,
        token_mask=jnp.asarray(shifted_batch["mask"]),
    )
    wrong_diff_mask = np.asarray(train_batch["loss_mask_diff"])[:, seq_len:].astype(int).tolist()
    assert wrong_diff_mask == [[0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0]]
    assert wrong_diff_mask != [[0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0]]


def main() -> None:
    test_tidar_uses_raw_token_masks()
    test_causal_target_shift_is_wrong_for_tidar()
    print("TiDAR mask alignment test passed")


if __name__ == "__main__":
    main()
