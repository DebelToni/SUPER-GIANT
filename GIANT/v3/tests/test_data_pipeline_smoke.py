from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from GIANT.v3.data_pipeline.build_corpus import (
    S3UploadCfg,
    StageCfg,
    StageSourceCfg,
    StageStats,
    SequenceEmitter,
    _arrow_schema,
    _build_s3_sync_command,
    _merge_s3_upload_cfg,
    _parse_s3_upload_cfg,
    _iter_jsonl_records,
    _maybe_hash,
    _stream_json_records,
    iter_stage_rows_raw,
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
    assert isinstance(loss_field.type, pa.FixedSizeBinaryType)
    assert loss_field.type.byte_width == 1


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


def test_weighted_source_mix_interleaves_sources(tmp_dir: Path) -> None:
    first_root = tmp_dir / "first"
    second_root = tmp_dir / "second"
    first_root.mkdir(parents=True, exist_ok=True)
    second_root.mkdir(parents=True, exist_ok=True)

    with (first_root / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(12):
            handle.write(json.dumps({"text": f"bg-{idx}"}) + "\n")

    with (second_root / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(12):
            handle.write(json.dumps({"text": f"en-{idx}"}) + "\n")

    stage = StageCfg(
        name="mixed",
        sequence_length=8,
        min_tokens=1,
        source_mix_mode="weighted_random",
        sources=[
            StageSourceCfg(type="json", json_root=str(first_root), file_glob="*.jsonl", sampling_weight=0.65),
            StageSourceCfg(type="json", json_root=str(second_root), file_glob="*.jsonl", sampling_weight=0.35),
        ],
    )

    rows = []
    for source, row in iter_stage_rows_raw(stage, rng=np.random.default_rng(0)):
        rows.append((Path(str(source.json_root)).name, row["text"]))
        if len(rows) >= 8:
            break

    prefixes = {text.split("-", 1)[0] for _, text in rows}
    assert prefixes == {"bg", "en"}


def test_s3_sync_command_uses_env_file_and_size_only(tmp_dir: Path) -> None:
    cfg = S3UploadCfg(enabled=True, destination_root="s3://giant-data/tmp/", env_file="/tmp/fake.env")
    command = _build_s3_sync_command(tmp_dir / "stage", "s3://giant-data/tmp/stage/", cfg)
    assert command[:2] == ["bash", "-lc"]
    script = command[2]
    assert "source /tmp/fake.env" in script
    assert "s5cmd sync --size-only" in script
    assert "s3://giant-data/tmp/stage/" in script


def test_stage_s3_upload_override_preserves_global_defaults() -> None:
    parent = S3UploadCfg(enabled=True, destination_root="s3://giant-data/base/", env_file="/tmp/env", tool="s5cmd")
    child = _parse_s3_upload_cfg({"destination": "s3://giant-data/custom/stage/"}, override=True)
    merged = _merge_s3_upload_cfg(parent, child)
    assert merged.enabled is True
    assert merged.env_file == "/tmp/env"
    assert merged.tool == "s5cmd"
    assert merged.destination == "s3://giant-data/custom/stage/"


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp())
    test_hash_deterministic()
    test_json_parsers(tmp_dir)
    test_arrow_schema_fixed_size()
    test_sequence_emitter_random_windows()
    test_weighted_source_mix_interleaves_sources(tmp_dir)
    test_s3_sync_command_uses_env_file_and_size_only(tmp_dir)
    test_stage_s3_upload_override_preserves_global_defaults()
    print("data_pipeline smoke tests passed")


if __name__ == "__main__":
    main()
