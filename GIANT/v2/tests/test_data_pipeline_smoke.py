from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from GIANT.v2.data_pipeline.build_corpus import (
    StageCfg,
    StageStats,
    SequenceEmitter,
    _arrow_schema,
    _iter_jsonl_records,
    _maybe_hash,
    _stream_json_records,
)


def test_hash_deterministic() -> None:
    value = "dedup-check"
    assert _maybe_hash(value) == _maybe_hash(value)


def test_json_parsers(tmp_dir: Path) -> None:
    array_path = tmp_dir / "records.json"
    jsonl_path = tmp_dir / "records.jsonl"

    with array_path.open("w", encoding="utf-8") as handle:
        json.dump([{"text": "alpha"}, {"text": "beta"}], handle)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write("{\"text\": \"gamma\"}\n")
        handle.write("{\"text\": \"delta\"}\n")

    array_rows = [row["text"] for row in _stream_json_records(array_path)]
    jsonl_rows = [row["text"] for row in _iter_jsonl_records(jsonl_path)]

    assert array_rows == ["alpha", "beta"]
    assert jsonl_rows == ["gamma", "delta"]


def test_arrow_schema_fixed_size() -> None:
    schema = _arrow_schema(8)
    field = schema.field("input_ids")
    assert isinstance(field.type, pa.FixedSizeListType)
    assert field.type.list_size == 8
    loss_field = schema.field("loss_mask")
    assert isinstance(loss_field.type, pa.FixedSizeListType)
    assert loss_field.type.list_size == 8


def test_sequence_emitter_random_windows() -> None:
    stage = StageCfg(
        name="test",
        sequence_length=4,
        min_tokens=1,
        random_windows_per_document=3,
    )
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, sep_token_id=None)
    stats = StageStats()
    rng = np.random.default_rng(0)
    emitter = SequenceEmitter(stage, tokenizer, rng, stats)

    rows = list(emitter.consume_tokens(list(range(20))))
    assert len(rows) == stage.random_windows_per_document
    for row in rows:
        assert len(row["input_ids"]) == stage.sequence_length


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp())
    test_hash_deterministic()
    test_json_parsers(tmp_dir)
    test_arrow_schema_fixed_size()
    test_sequence_emitter_random_windows()
    print("data_pipeline smoke tests passed")


if __name__ == "__main__":
    main()
