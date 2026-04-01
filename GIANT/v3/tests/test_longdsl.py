from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from transformers import AutoTokenizer

from GIANT.v3.Long.longdsl import GeneratorConfig, TokenizerSpec, generate_example, save_tokenizer


def test_generate_level1_example() -> None:
    rng = np.random.default_rng(0)
    row = cast(dict[str, object], generate_example(GeneratorConfig(level=1, context_length=128), rng))
    answer = cast(str, row["answer"])
    messages = cast(list[dict[str, str]], row["messages"])
    assert answer.startswith("V")
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"].startswith("ANS ")


def test_generate_level2_example() -> None:
    rng = np.random.default_rng(1)
    row = cast(dict[str, object], generate_example(GeneratorConfig(level=2, context_length=256), rng))
    answer = cast(str, row["answer"])
    messages = cast(list[dict[str, str]], row["messages"])
    assert answer.startswith("V")
    assert "QUERY" in messages[0]["content"]


def test_save_tokenizer_roundtrip() -> None:
    tmp_dir = Path(tempfile.mkdtemp()) / "tok"
    save_tokenizer(tmp_dir, TokenizerSpec())
    tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
    ids = tokenizer.encode("@user LEVEL L1 PROGRAM DEF E001 V001 SEP QUERY ASK E001", add_special_tokens=False)
    assert ids
    decoded = tokenizer.decode(ids)
    assert "E001" in decoded


def main() -> None:
    test_generate_level1_example()
    test_generate_level2_example()
    test_save_tokenizer_roundtrip()
    print("longdsl tests passed")


if __name__ == "__main__":
    main()
