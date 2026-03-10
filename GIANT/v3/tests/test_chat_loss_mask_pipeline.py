from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

from GIANT.v3.data_pipeline.build_corpus import (
    StageCfg,
    StageSourceCfg,
    StageStats,
    SequenceEmitter,
    _arrow_schema,
    iter_stage_rows,
)


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


def main() -> None:
    test_chat_loss_mask_pipeline()
    print("chat loss mask test passed")


if __name__ == "__main__":
    main()
