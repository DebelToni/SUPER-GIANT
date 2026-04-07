from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from transformers import AutoTokenizer

from GIANT.v3.Long.longdsl import GeneratorConfig, TokenizerSpec, generate_example, save_tokenizer, surface_tokens


def test_generate_level1_example() -> None:
    rng = np.random.default_rng(0)
    row = cast(dict[str, object], generate_example(GeneratorConfig(level=1, context_length=128), rng))
    assert row["genre"] == "admin_record"
    assert row["relation_key"] in {"locker", "room", "role", "status"}
    assert cast(str, row["question"]).endswith("?")
    assert cast(str, row["answer"]) == surface_tokens(cast(str, row["text"]))[cast(int, row["answer_token_index"])]


def test_generate_level2_example_contains_update_supervision() -> None:
    rng = np.random.default_rng(1)
    row = cast(dict[str, object], generate_example(GeneratorConfig(level=2, context_length=192), rng))
    latent = cast(dict[str, object], row["latent_world"])
    ops = cast(list[list[str]], latent["ops"])
    assert any(op[0] == "SET" for op in ops)
    assert cast(list[int], row["evidence_sentence_indices"])


def test_save_tokenizer_roundtrip() -> None:
    tmp_dir = Path(tempfile.mkdtemp()) / "tok"
    save_tokenizer(tmp_dir, TokenizerSpec())
    tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
    text = "Context: Mira was assigned locker 18. Question: Which locker is currently associated with Nera? Answer: 18"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids
    decoded = tokenizer.decode(ids)
    assert "Mira" in decoded
    assert "18" in decoded


def main() -> None:
    test_generate_level1_example()
    test_generate_level2_example_contains_update_supervision()
    test_save_tokenizer_roundtrip()
    print("long tests passed")


if __name__ == "__main__":
    main()
