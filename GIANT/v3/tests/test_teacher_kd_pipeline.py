from __future__ import annotations

import json
import tempfile
from pathlib import Path

from GIANT.v3.data_curation.build_prompt_bank import PromptSourceCfg, build_prompt_bank, extract_prompt_rows
from GIANT.v3.data_curation.teacher_distill import build_distilled_row, generate_teacher_outputs, run_teacher_distill, TeacherKDCfg


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_extracts_each_user_turn_as_prompt() -> None:
    row = {
        "id": "conv-1",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "Second answer."},
            {"role": "user", "content": "Third question?"},
        ],
    }
    source = PromptSourceCfg(name="unit", min_assistant_messages_before_prompt=1)
    prompts = extract_prompt_rows(row, source, source_index=0)

    assert len(prompts) == 2
    assert prompts[0]["messages"][-1]["content"] == "Second question?"
    assert prompts[1]["messages"][-1]["content"] == "Third question?"
    assert prompts[0]["metadata"]["assistant_messages_before_prompt"] == 1
    assert prompts[1]["metadata"]["assistant_messages_before_prompt"] == 2


def test_prompt_bank_and_mock_teacher_distill() -> None:
    tmp = Path(tempfile.mkdtemp())
    source_root = tmp / "source"
    prompt_out = tmp / "prompts"
    kd_out = tmp / "kd"
    _write_jsonl(
        source_root / "chat.jsonl",
        [
            {
                "id": "conv-2",
                "messages": [
                    {"role": "user", "content": "What is RAM?"},
                    {"role": "assistant", "content": "Memory."},
                    {"role": "user", "content": "Explain storage."},
                ],
            }
        ],
    )

    cfg_path = tmp / "teacher_kd.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "prompt_bank:",
                f"  output_dir: {prompt_out}",
                "  output_filename: prompts.jsonl",
                "  sources:",
                "    - name: local_chat",
                "      type: json",
                f"      json_root: {source_root}",
                "      file_glob: '*.jsonl'",
                "      min_assistant_messages_before_prompt: 0",
                "      max_prompts: 4",
                "teacher_kd:",
                f"  prompt_path: {prompt_out / 'prompts.jsonl'}",
                f"  output_dir: {kd_out}",
                "  output_filename: distilled_chat.jsonl",
                "  model_name: mock-teacher",
                "  batch_size: 2",
                "  mock_response_template: 'Teacher answer to: {user}'",
            ]
        ),
        encoding="utf-8",
    )

    prompt_summary = build_prompt_bank(cfg_path)
    assert prompt_summary["prompts_written"] == 2

    kd_summary = run_teacher_distill(cfg_path)
    assert kd_summary["rows_written"] == 2
    rows = [json.loads(line) for line in (kd_out / "distilled_chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["metadata"]["distillation_type"] == "teacher_output_sft"
    assert rows[0]["metadata"]["trainable_message_indices"] == [len(rows[0]["messages"]) - 1]


def test_distilled_row_records_training_metadata() -> None:
    prompt = {
        "prompt_id": "p1",
        "messages": [{"role": "user", "content": "Hello?"}],
        "metadata": {"source_name": "unit"},
    }
    cfg = TeacherKDCfg(prompt_path="unused", output_dir="unused", model_name="mock")
    row = build_distilled_row(prompt, "Hi.", cfg=cfg)
    assert row["messages"] == [
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hi."},
    ]
    assert row["metadata"]["non_trainable_message_indices"] == [0]
    assert row["metadata"]["trainable_message_indices"] == [1]


def test_teacher_distill_resume_skips_previous_prompts() -> None:
    tmp = Path(tempfile.mkdtemp())
    prompt_path = tmp / "prompts.jsonl"
    _write_jsonl(
        prompt_path,
        [
            {"prompt_id": "p1", "messages": [{"role": "user", "content": "One?"}], "metadata": {}},
            {"prompt_id": "p2", "messages": [{"role": "user", "content": "Two?"}], "metadata": {}},
        ],
    )
    cfg_path = tmp / "teacher_kd.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "teacher_kd:",
                f"  prompt_path: {prompt_path}",
                f"  output_dir: {tmp / 'out'}",
                "  output_filename: distilled_chat.jsonl",
                "  model_name: mock-teacher",
                "  batch_size: 1",
                "  max_prompts: 1",
                "  mock_response_template: 'Answer: {user}'",
            ]
        ),
        encoding="utf-8",
    )

    first = run_teacher_distill(cfg_path)
    second = run_teacher_distill(cfg_path, resume=True)
    rows = [json.loads(line) for line in (tmp / "out" / "distilled_chat.jsonl").read_text(encoding="utf-8").splitlines()]

    assert first["rows_written"] == 1
    assert second["rows_written"] == 2
    assert rows[0]["metadata"]["prompt_id"] == "p1"
    assert rows[1]["metadata"]["prompt_id"] == "p2"


def test_real_tiny_transformers_teacher_if_enabled() -> None:
    import os

    if os.environ.get("GIANT_RUN_REAL_TEACHER_KD_TEST") != "1":
        return
    cfg = TeacherKDCfg(
        prompt_path="unused",
        output_dir="unused",
        model_name="hf-internal-testing/tiny-random-gpt2",
        trust_remote_code=False,
        device="cpu",
        batch_size=1,
        max_new_tokens=8,
    )
    outputs = generate_teacher_outputs(
        [{"prompt_id": "tiny", "messages": [{"role": "user", "content": "Say hello."}]}],
        cfg,
    )
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)


def main() -> None:
    test_extracts_each_user_turn_as_prompt()
    test_prompt_bank_and_mock_teacher_distill()
    test_distilled_row_records_training_metadata()
    test_teacher_distill_resume_skips_previous_prompts()
    test_real_tiny_transformers_teacher_if_enabled()
    print("teacher KD pipeline tests passed")


if __name__ == "__main__":
    main()
