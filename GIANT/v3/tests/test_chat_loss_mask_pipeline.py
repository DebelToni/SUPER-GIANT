from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from GIANT.v3.data_pipeline.build_corpus import (
    StageCfg,
    StageSourceCfg,
    StageStats,
    _arrow_schema,
    _pack_loss_mask,
    iter_stage_rows,
)
from GIANT.v3.model.arrow_data_loader import ShardedArrowDataset, StageDataLoader


def _write_chat_records(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sample = {
        "messages": [
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Add 2+2."},
            {"role": "assistant", "content": "It is 4."},
        ]
    }
    with (root / "chat.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(sample) + "\n")


def test_chat_loss_mask_pipeline() -> None:
    tmp_dir = Path(tempfile.mkdtemp())
    chat_root = tmp_dir / "chat"
    _write_chat_records(chat_root)

    source = StageSourceCfg(
        type="json",
        json_root=str(chat_root),
        file_glob="*.jsonl",
        chat_messages_field="messages",
        chat_role_field="role",
        chat_content_field="content",
        chat_assistant_roles=["assistant"],
        chat_role_prefix="### {role}\n",
        chat_turn_suffix="\n",
    )
    stage = StageCfg(
        name="chat",
        sequence_length=64,
        min_tokens=1,
        pack_sequences=False,
        add_eos=False,
        sources=[source],
    )

    class DummyTokenizer:
        eos_token_id = None
        pad_token_id = 0
        sep_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [min(255, ord(ch)) for ch in text]

        def __call__(self, texts, **kwargs):
            input_ids = []
            offsets = []
            for text in texts:
                ids = [min(255, ord(ch)) for ch in text]
                input_ids.append(ids)
                offsets.append([(i, i + 1) for i in range(len(text))])
            return {"input_ids": input_ids, "offset_mapping": offsets}

    tokenizer = DummyTokenizer()

    stats = StageStats()
    rng = np.random.default_rng(0)
    rows = list(iter_stage_rows(stage, tokenizer, rng, stats, batch_size=1))
    assert rows, "Expected at least one sequence"
    row = rows[0]
    assert "loss_mask" in row
    packed = np.frombuffer(row["loss_mask"], dtype=np.uint8)
    mask = np.unpackbits(packed, bitorder="little")[: stage.sequence_length].astype(np.float32)
    assert mask.sum() > 0
    assert mask.sum() < len(mask)

    schema = _arrow_schema(stage.sequence_length)
    table = pa.Table.from_pylist(rows, schema=schema)
    assert "loss_mask" in table.column_names


def test_chat_loss_mask_targets_assistant_only() -> None:
    tmp_dir = Path(tempfile.mkdtemp())
    chat_root = tmp_dir / "chat"
    _write_chat_records(chat_root)

    source = StageSourceCfg(
        type="json",
        json_root=str(chat_root),
        file_glob="*.jsonl",
        chat_messages_field="messages",
        chat_role_field="role",
        chat_content_field="content",
        chat_assistant_roles=["assistant"],
        chat_role_prefix="<{role}>",
        chat_turn_suffix="</turn>",
        chat_loss_on_suffix=True,
    )
    stage = StageCfg(
        name="chat",
        sequence_length=128,
        min_tokens=1,
        pack_sequences=False,
        add_eos=False,
        sources=[source],
    )

    class DummyTokenizer:
        eos_token_id = None
        pad_token_id = 0
        sep_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [min(255, ord(ch)) for ch in text]

    tokenizer = DummyTokenizer()
    stats = StageStats()
    rng = np.random.default_rng(0)
    rows = list(iter_stage_rows(stage, tokenizer, rng, stats, batch_size=1))
    assert rows, "Expected at least one sequence"

    dataset_root = tmp_dir / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema(stage.sequence_length)
    with pa.OSFile(str(dataset_root / "chat-000000.arrow"), "wb") as sink:
        with pa_ipc.new_file(sink, schema) as writer:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    with (dataset_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"stage": "chat", "shards": [{"filename": "chat-000000.arrow", "rows": len(rows)}]}, handle)

    dataset = ShardedArrowDataset(dataset_root)
    loader = StageDataLoader(
        dataset,
        batch_size=1,
        seq_len=stage.sequence_length,
        shuffle=False,
        seed=0,
        pad_token_id=0,
    )
    batch = next(iter(loader))
    masked_targets = batch["target"][0][batch["mask"][0].astype(bool)].tolist()

    raw_loader = StageDataLoader(
        dataset,
        batch_size=1,
        seq_len=stage.sequence_length,
        shuffle=False,
        seed=0,
        pad_token_id=0,
        mask_target_shift=False,
    )
    raw_batch = next(iter(raw_loader))
    masked_input_tokens = raw_batch["input"][0][raw_batch["mask"][0].astype(bool)].tolist()

    expected_text = "Hi there!</turn>It is 4.</turn>"
    expected_targets = tokenizer.encode(expected_text, add_special_tokens=False)
    assert masked_targets == expected_targets
    assert masked_input_tokens == expected_targets


def test_loader_exposes_stored_lengths() -> None:
    tmp_dir = Path(tempfile.mkdtemp())
    seq_len = 4
    dataset_root = tmp_dir / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "input_ids": [5, 0, 6, 0],
            "length": 3,
            "loss_mask": _pack_loss_mask([1.0, 1.0, 1.0, 1.0], seq_len),
        },
        {
            "input_ids": [7, 0, 0, 0],
            "length": 1,
            "loss_mask": _pack_loss_mask([1.0, 1.0, 1.0, 1.0], seq_len),
        },
    ]
    schema = _arrow_schema(seq_len)
    with pa.OSFile(str(dataset_root / "tiny-000000.arrow"), "wb") as sink:
        with pa_ipc.new_file(sink, schema) as writer:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    with (dataset_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"stage": "tiny", "shards": [{"filename": "tiny-000000.arrow", "rows": len(rows)}]}, handle)

    dataset = ShardedArrowDataset(dataset_root)
    loader = StageDataLoader(
        dataset,
        batch_size=2,
        seq_len=seq_len,
        shuffle=False,
        seed=0,
        pad_token_id=0,
        mask_target_shift=False,
    )
    batch = next(iter(loader))
    assert batch["length"].tolist() == [3, 1]
    assert batch["mask"].tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]

    causal_loader = StageDataLoader(
        dataset,
        batch_size=2,
        seq_len=seq_len,
        shuffle=False,
        seed=0,
        pad_token_id=0,
    )
    causal_batch = next(iter(causal_loader))
    assert causal_batch["mask"].tolist() == [[1, 1, 0, 0], [0, 0, 0, 0]]


def main() -> None:
    test_chat_loss_mask_pipeline()
    test_chat_loss_mask_targets_assistant_only()
    test_loader_exposes_stored_lengths()
    print("chat loss mask test passed")


if __name__ == "__main__":
    main()
