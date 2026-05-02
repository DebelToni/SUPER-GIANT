from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from omegaconf import OmegaConf


@dataclass
class StudentTokenizerCfg:
    name: str = ""
    cache_dir: Optional[str] = None
    use_custom: bool = False
    custom_path: str = ""


@dataclass
class TeacherKDCfg:
    prompt_path: str
    output_dir: str
    output_filename: str = "distilled_chat.jsonl"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = True
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 4
    max_prompts: Optional[int] = None
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    mock_response_template: Optional[str] = None
    assistant_role: str = "assistant"
    prompt_messages_field: str = "messages"
    student_tokenizer: Optional[StudentTokenizerCfg] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher-output KD chat rows from a prompt bank.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_config(path: str | Path) -> TeacherKDCfg:
    cfg = OmegaConf.load(path)
    kd_raw = cfg.get("teacher_kd") or cfg
    kd_dict = OmegaConf.to_container(kd_raw, resolve=True)
    if not isinstance(kd_dict, dict):
        raise ValueError("teacher_kd config must be a mapping")
    student_raw = kd_dict.get("student_tokenizer")
    if student_raw:
        kd_dict["student_tokenizer"] = StudentTokenizerCfg(**dict(student_raw))
    return TeacherKDCfg(**kd_dict)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def render_messages_for_teacher(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return str(
                apply_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    parts = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def load_student_tokenizer(cfg: Optional[StudentTokenizerCfg]):
    if cfg is None:
        return None
    from GIANT.v3.data_pipeline.build_corpus import TokenizerCfg, load_tokenizer

    return load_tokenizer(
        TokenizerCfg(
            name=cfg.name,
            cache_dir=cfg.cache_dir,
            use_custom=cfg.use_custom,
            custom_path=cfg.custom_path,
        )
    )


def student_token_counts(tokenizer: Any, messages: List[Dict[str, str]], assistant_text: str) -> Dict[str, int]:
    if tokenizer is None:
        return {}
    from GIANT.v3.data_pipeline.build_corpus import _tokenize_text

    prompt_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
    full_text = f"{prompt_text}\nassistant: {assistant_text}"
    return {
        "prompt_tokens": len(_tokenize_text(tokenizer, prompt_text)),
        "assistant_tokens": len(_tokenize_text(tokenizer, assistant_text)),
        "full_chat_tokens": len(_tokenize_text(tokenizer, full_text)),
    }


def build_distilled_row(
    prompt_row: Dict[str, Any],
    assistant_text: str,
    *,
    cfg: TeacherKDCfg,
    student_tokenizer: Any = None,
) -> Dict[str, Any]:
    messages = list(prompt_row.get(cfg.prompt_messages_field) or [])
    messages = [dict(message) for message in messages if isinstance(message, dict)]
    messages.append({"role": cfg.assistant_role, "content": assistant_text.strip()})
    trainable_index = len(messages) - 1
    metadata = dict(prompt_row.get("metadata") or {})
    metadata.update(
        {
            "prompt_id": prompt_row.get("prompt_id"),
            "teacher_model": cfg.model_name,
            "distillation_type": "teacher_output_sft",
            "trainable_message_indices": [trainable_index],
            "non_trainable_message_indices": list(range(trainable_index)),
            "generation": {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
            },
        }
    )
    counts = student_token_counts(student_tokenizer, messages[:-1], assistant_text)
    if counts:
        metadata["student_token_counts"] = counts
    if cfg.metadata:
        metadata["run_metadata"] = dict(cfg.metadata)
    return {"messages": messages, "metadata": metadata}


def _mock_responses(prompts: List[Dict[str, Any]], template: str) -> List[str]:
    outputs = []
    for prompt in prompts:
        messages = prompt.get("messages") or []
        last_user = ""
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role") or "") == "user":
                last_user = str(message.get("content") or "")
                break
        outputs.append(template.format(prompt_id=prompt.get("prompt_id", ""), user=last_user))
    return outputs


def generate_teacher_outputs(prompts: List[Dict[str, Any]], cfg: TeacherKDCfg) -> List[str]:
    if cfg.mock_response_template is not None:
        return _mock_responses(prompts, cfg.mock_response_template)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = cfg.tokenizer_name or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if cfg.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif cfg.dtype == "float16":
        dtype = torch.float16
    device_map = "auto" if cfg.device == "auto" else None
    model_kwargs: Dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if not device_map:
        model = model.to(cfg.device)
    model.eval()

    texts = [render_messages_for_teacher(tokenizer, list(prompt.get("messages") or [])) for prompt in prompts]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    if not device_map:
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
    else:
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    gen_kwargs = {
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": bool(cfg.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        gen_kwargs.update({"temperature": float(cfg.temperature), "top_p": float(cfg.top_p)})
    with torch.inference_mode():
        generated = model.generate(**encoded, **gen_kwargs)
    prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
    outputs = []
    for row, prompt_len in zip(generated, prompt_lens):
        new_tokens = row[int(prompt_len) :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def run_teacher_distill(config_path: str | Path, *, resume: bool = False) -> Dict[str, Any]:
    cfg = parse_config(config_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg.output_filename
    state_path = output_dir / "state.json"
    previous = 0
    previous_rows_written = 0
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous = int(state.get("prompts_seen", 0))
        previous_rows_written = int(state.get("rows_written", 0))

    student_tokenizer = load_student_tokenizer(cfg.student_tokenizer)
    mode = "a" if resume else "w"
    prompts_seen = previous
    rows_written = previous_rows_written
    pending: List[Dict[str, Any]] = []

    def flush(handle) -> None:
        nonlocal rows_written, pending
        if not pending:
            return
        outputs = generate_teacher_outputs(pending, cfg)
        for prompt, output in zip(pending, outputs):
            if not output.strip():
                continue
            row = build_distilled_row(prompt, output, cfg=cfg, student_tokenizer=student_tokenizer)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1
        pending = []
        state_path.write_text(json.dumps({"prompts_seen": prompts_seen, "rows_written": rows_written}, indent=2), encoding="utf-8")

    with output_path.open(mode, encoding="utf-8") as handle:
        for input_idx, prompt in enumerate(iter_jsonl(Path(cfg.prompt_path)), start=1):
            if resume and input_idx <= previous:
                continue
            prompts_seen = input_idx
            pending.append(prompt)
            if len(pending) >= max(1, int(cfg.batch_size)):
                flush(handle)
            if cfg.max_prompts is not None and input_idx - previous >= int(cfg.max_prompts):
                break
        flush(handle)

    summary = {
        "prompt_path": cfg.prompt_path,
        "output": str(output_path),
        "teacher_model": cfg.model_name,
        "prompts_seen": prompts_seen,
        "rows_written": rows_written,
        "distillation_type": "teacher_output_sft",
    }
    state_path.write_text(json.dumps({"prompts_seen": prompts_seen, "rows_written": rows_written}, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    run_teacher_distill(args.config, resume=args.resume)
    os._exit(0)


if __name__ == "__main__":
    main()
