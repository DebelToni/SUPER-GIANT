from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from omegaconf import OmegaConf


@dataclass
class PromptSourceCfg:
    type: str = "huggingface"
    name: str = ""
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    data_files: Optional[Any] = None
    json_root: Optional[str] = None
    file_glob: str = "**/*.json*"
    messages_field: str = "messages"
    role_field: str = "role"
    content_field: str = "content"
    user_roles: List[str] = field(default_factory=lambda: ["user"])
    assistant_roles: List[str] = field(default_factory=lambda: ["assistant"])
    system_roles: List[str] = field(default_factory=lambda: ["system"])
    max_rows: Optional[int] = None
    max_prompts: Optional[int] = None
    min_assistant_messages_before_prompt: int = 0
    max_assistant_messages_before_prompt: Optional[int] = None
    min_user_chars: int = 1
    max_user_chars: Optional[int] = None
    include_system: bool = True
    include_history: bool = True
    max_context_messages: Optional[int] = None
    normalize_whitespace: bool = True


@dataclass
class PromptBankCfg:
    output_dir: str
    output_filename: str = "prompts.jsonl"
    sources: List[PromptSourceCfg] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL prompt bank from chat/instruction datasets.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def normalize_text(text: str, *, collapse: bool = True) -> str:
    text = str(text or "").strip()
    if collapse:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def iter_json_rows(root: Path, pattern: str) -> Iterator[Dict[str, Any]]:
    for path in sorted(root.rglob(pattern)):
        suffixes = "".join(path.suffixes).lower()
        with _open_text(path) as handle:
            if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
            else:
                data = json.load(handle)
                rows = data if isinstance(data, list) else [data]
                for row in rows:
                    if isinstance(row, dict):
                        yield row


def iter_source_rows(source: PromptSourceCfg) -> Iterator[Dict[str, Any]]:
    source_type = str(source.type or "huggingface").lower()
    if source_type in {"json", "jsonl", "json_dir"}:
        if not source.json_root:
            raise ValueError(f"Prompt source {source.name or source.type!r} needs json_root")
        yield from iter_json_rows(Path(source.json_root), source.file_glob)
        return

    if source_type in {"hf", "huggingface"}:
        if not source.dataset_name:
            raise ValueError(f"Prompt source {source.name or source.type!r} needs dataset_name")
        from datasets import load_dataset

        kwargs: Dict[str, Any] = {}
        if source.dataset_config:
            kwargs["name"] = source.dataset_config
        if source.data_files is not None:
            kwargs["data_files"] = source.data_files
        ds = load_dataset(source.dataset_name, split=source.split, streaming=bool(source.streaming), **kwargs)
        for row in ds:
            if isinstance(row, dict):
                yield row
        return

    raise ValueError(f"Unsupported prompt source type: {source.type!r}")


def _limit_context(messages: List[Dict[str, str]], source: PromptSourceCfg) -> List[Dict[str, str]]:
    limit = source.max_context_messages
    if not limit or len(messages) <= limit:
        return messages
    if not source.include_system:
        return messages[-limit:]
    leading_system = []
    rest_start = 0
    for idx, message in enumerate(messages):
        if message["role"] in source.system_roles:
            leading_system.append(message)
            rest_start = idx + 1
            continue
        break
    remaining = max(0, int(limit) - len(leading_system))
    return leading_system + messages[rest_start:][-remaining:]


def extract_prompt_rows(
    row: Dict[str, Any],
    source: PromptSourceCfg,
    *,
    source_index: int,
) -> List[Dict[str, Any]]:
    raw_messages = row.get(source.messages_field)
    if not isinstance(raw_messages, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get(source.role_field) or "").strip().lower()
        content = normalize_text(raw.get(source.content_field) or "", collapse=source.normalize_whitespace)
        if not role or not content:
            continue
        if role in source.system_roles and not source.include_system:
            continue
        cleaned.append({"role": role, "content": content})

    prompts: List[Dict[str, Any]] = []
    assistant_count = 0
    row_id = row.get("id") or row.get("conversation_id") or row.get("source_id") or source_index
    for msg_idx, message in enumerate(cleaned):
        role = message["role"]
        if role in source.user_roles:
            user_len = len(message["content"])
            if user_len < int(source.min_user_chars):
                continue
            if source.max_user_chars is not None and user_len > int(source.max_user_chars):
                continue
            if assistant_count < int(source.min_assistant_messages_before_prompt):
                continue
            if (
                source.max_assistant_messages_before_prompt is not None
                and assistant_count > int(source.max_assistant_messages_before_prompt)
            ):
                continue

            if source.include_history:
                prompt_messages = cleaned[: msg_idx + 1]
            else:
                system_messages = [m for m in cleaned[:msg_idx] if m["role"] in source.system_roles]
                prompt_messages = (system_messages if source.include_system else []) + [message]
            prompt_messages = _limit_context(prompt_messages, source)
            if not prompt_messages or prompt_messages[-1]["role"] not in source.user_roles:
                continue

            prompt_id = f"{source.name or source.dataset_name or source.type}:{row_id}:{msg_idx}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "messages": prompt_messages,
                    "metadata": {
                        "source_name": source.name or source.dataset_name or source.type,
                        "source_row_id": row_id,
                        "source_message_index": msg_idx,
                        "assistant_messages_before_prompt": assistant_count,
                        "trainable_message_indices": [],
                    },
                }
            )
        if role in source.assistant_roles:
            assistant_count += 1
    return prompts


def parse_config(path: str | Path) -> PromptBankCfg:
    cfg = OmegaConf.load(path)
    bank_raw = cfg.get("prompt_bank") or cfg
    bank_dict = OmegaConf.to_container(bank_raw, resolve=True)
    if not isinstance(bank_dict, dict):
        raise ValueError("prompt_bank config must be a mapping")
    raw_sources = bank_dict.pop("sources", []) or []
    sources = [PromptSourceCfg(**dict(source)) for source in raw_sources]
    return PromptBankCfg(sources=sources, **bank_dict)


def build_prompt_bank(config_path: str | Path) -> Dict[str, Any]:
    cfg = parse_config(config_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg.output_filename

    counts: Dict[str, int] = {}
    total_rows = 0
    total_prompts = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source in cfg.sources:
            source_name = source.name or source.dataset_name or source.type
            source_prompts = 0
            source_rows = 0
            for row_idx, row in enumerate(iter_source_rows(source)):
                source_rows += 1
                if source.max_rows is not None and source_rows > int(source.max_rows):
                    break
                prompt_rows = extract_prompt_rows(row, source, source_index=row_idx)
                for prompt_row in prompt_rows:
                    if source.max_prompts is not None and source_prompts >= int(source.max_prompts):
                        break
                    handle.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
                    source_prompts += 1
                    total_prompts += 1
                if source.max_prompts is not None and source_prompts >= int(source.max_prompts):
                    break
            counts[str(source_name)] = source_prompts
            total_rows += source_rows

    summary = {
        "output": str(output_path),
        "sources": counts,
        "rows_seen": total_rows,
        "prompts_written": total_prompts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    build_prompt_bank(args.config)


if __name__ == "__main__":
    main()
