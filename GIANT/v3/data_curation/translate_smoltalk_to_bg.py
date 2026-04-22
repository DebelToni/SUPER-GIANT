from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-bg"
DEFAULT_ALLOWED_SOURCES = (
    "smol-magpie-ultra",
    "systemchats-30k",
    "everyday-conversations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate selected SmolTalk conversations into Bulgarian.")
    parser.add_argument("--dataset_name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--dataset_config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output_dir",
        default="/proj/giant-data/GIANT/GIANT-Chat/data_curation/smoltalk_bg_en_v1",
    )
    parser.add_argument("--max_examples", type=int, default=150000)
    parser.add_argument("--conversation_batch_size", type=int, default=16)
    parser.add_argument("--translation_batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_source_tokens", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_messages", type=int, default=6)
    parser.add_argument("--max_message_chars", type=int, default=1200)
    parser.add_argument("--min_cyrillic_ratio", type=float, default=0.20)
    parser.add_argument("--allowed_sources", nargs="*", default=list(DEFAULT_ALLOWED_SOURCES))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyrillic = [ch for ch in letters if "\u0400" <= ch <= "\u04ff"]
    return len(cyrillic) / len(letters)


def looks_formula_heavy(text: str) -> bool:
    markers = ("\\(", "\\)", "\\[", "\\]", "```", "^", "=", "\u2264", "\u2265")
    if any(marker in text for marker in markers):
        return True
    operator_count = sum(text.count(op) for op in ("+", "-", "*", "/", "=", "^"))
    if operator_count >= 6:
        return True
    digit_count = sum(ch.isdigit() for ch in text)
    alpha_count = sum(ch.isalpha() for ch in text)
    if digit_count >= 10 and digit_count > alpha_count:
        return True
    return False


def split_text_into_chunks(text: str, *, max_chars: int = 500) -> List[str]:
    chunks: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        current: List[str] = []
        current_len = 0
        for sentence in sentences or [paragraph]:
            if len(sentence) > max_chars:
                words = sentence.split()
                sentence_parts: List[str] = []
                partial: List[str] = []
                partial_len = 0
                for word in words:
                    extra = len(word) + (1 if partial else 0)
                    if partial and partial_len + extra > max_chars:
                        sentence_parts.append(" ".join(partial))
                        partial = [word]
                        partial_len = len(word)
                    else:
                        partial.append(word)
                        partial_len += extra
                if partial:
                    sentence_parts.append(" ".join(partial))
            else:
                sentence_parts = [sentence]
            for sentence_part in sentence_parts:
                extra = len(sentence_part) + (1 if current else 0)
                if current and current_len + extra > max_chars:
                    chunks.append(" ".join(current))
                    current = [sentence_part]
                    current_len = len(sentence_part)
                else:
                    current.append(sentence_part)
                    current_len += extra
        if current:
            chunks.append(" ".join(current))
    return chunks or [text.strip()]


def iter_rows(args: argparse.Namespace):
    return load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )


def normalize_messages(
    raw_messages: Sequence[dict],
    *,
    max_messages: int,
    max_message_chars: int,
) -> List[dict] | None:
    messages: List[dict] = []
    if not isinstance(raw_messages, Sequence):
        return None
    if len(raw_messages) < 2 or len(raw_messages) > max_messages:
        return None
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        role = str(raw.get("role") or "").strip().lower()
        content = str(raw.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            return None
        if not content or len(content) > max_message_chars or looks_formula_heavy(content):
            return None
        messages.append({"role": role, "content": content})
    has_assistant = any(message["role"] == "assistant" for message in messages)
    has_user = any(message["role"] == "user" for message in messages)
    if not has_assistant or not has_user:
        return None
    return messages


def translate_texts(
    texts: List[str],
    *,
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str,
    batch_size: int,
    num_beams: int,
    max_source_tokens: int,
    max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
        outputs.extend(text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def flush_examples(
    examples: List[dict],
    *,
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str,
    translation_batch_size: int,
    num_beams: int,
    max_source_tokens: int,
    max_new_tokens: int,
    min_cyrillic_ratio: float,
    original_handle,
    translated_handle,
    bilingual_handle,
) -> dict:
    if not examples:
        return {"written": 0, "dropped": 0}

    flat_texts: List[str] = []
    spans: List[List[tuple[str, int, int]]] = []
    for example in examples:
        message_spans: List[tuple[str, int, int]] = []
        for message in example["messages"]:
            start = len(flat_texts)
            flat_texts.extend(split_text_into_chunks(message["content"]))
            message_spans.append((message["role"], start, len(flat_texts)))
        spans.append(message_spans)

    translated_texts = translate_texts(
        flat_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=max(1, translation_batch_size),
        num_beams=max(1, num_beams),
        max_source_tokens=max(32, max_source_tokens),
        max_new_tokens=max(32, max_new_tokens),
    )

    written = 0
    dropped = 0
    for example, message_spans in zip(examples, spans):
        translated_messages = []
        for role, start, end in message_spans:
            translated = "\n\n".join(chunk.strip() for chunk in translated_texts[start:end] if chunk.strip()).strip()
            if not translated:
                translated_messages = []
                break
            translated_messages.append({"role": role, "content": translated})
        if not translated_messages:
            dropped += 1
            continue

        translated_text = "\n".join(message["content"] for message in translated_messages)
        if cyrillic_ratio(translated_text) < min_cyrillic_ratio:
            dropped += 1
            continue

        original_row = {
            "dataset": "HuggingFaceTB/smoltalk",
            "source": example["source"],
            "language": "en",
            "variant": "original_en",
            "messages": example["messages"],
        }
        translated_row = {
            "dataset": "HuggingFaceTB/smoltalk",
            "source": example["source"],
            "language": "bg",
            "variant": "translated_bg",
            "messages": translated_messages,
        }
        original_handle.write(json.dumps(original_row, ensure_ascii=False) + "\n")
        translated_handle.write(json.dumps(translated_row, ensure_ascii=False) + "\n")
        bilingual_handle.write(json.dumps(original_row, ensure_ascii=False) + "\n")
        bilingual_handle.write(json.dumps(translated_row, ensure_ascii=False) + "\n")
        written += 1

    return {"written": written, "dropped": dropped}


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"raw_rows_seen": 0, "accepted_examples": 0, "written_examples": 0, "dropped_examples": 0}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "state.json"
    state = load_state(state_path) if args.resume else {"raw_rows_seen": 0, "accepted_examples": 0, "written_examples": 0, "dropped_examples": 0}
    resume_raw_rows_seen = int(state.get("raw_rows_seen", 0)) if args.resume else 0

    mode = "a" if args.resume else "w"
    original_path = output_dir / "smoltalk_original_en.jsonl"
    translated_path = output_dir / "smoltalk_translated_bg.jsonl"
    bilingual_path = output_dir / "smoltalk_bilingual.jsonl"

    device = select_device()
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name).to(device)
    model.eval()

    allowed_sources = {str(value) for value in args.allowed_sources}
    pending: List[dict] = []

    with original_path.open(mode, encoding="utf-8") as original_handle, translated_path.open(mode, encoding="utf-8") as translated_handle, bilingual_path.open(mode, encoding="utf-8") as bilingual_handle:
        for row in iter_rows(args):
            state["raw_rows_seen"] += 1
            if args.resume and state["raw_rows_seen"] <= resume_raw_rows_seen:
                continue
            if state["accepted_examples"] >= args.max_examples:
                break

            source = str(row.get("source") or "")
            if allowed_sources and source not in allowed_sources:
                continue

            messages = normalize_messages(
                row.get("messages") or [],
                max_messages=max(2, args.max_messages),
                max_message_chars=max(64, args.max_message_chars),
            )
            if messages is None:
                continue

            pending.append({"source": source, "messages": messages})
            state["accepted_examples"] += 1
            if len(pending) < max(1, args.conversation_batch_size):
                continue

            result = flush_examples(
                pending,
                tokenizer=tokenizer,
                model=model,
                device=device,
                translation_batch_size=args.translation_batch_size,
                num_beams=args.num_beams,
                max_source_tokens=args.max_source_tokens,
                max_new_tokens=args.max_new_tokens,
                min_cyrillic_ratio=args.min_cyrillic_ratio,
                original_handle=original_handle,
                translated_handle=translated_handle,
                bilingual_handle=bilingual_handle,
            )
            state["written_examples"] += int(result["written"])
            state["dropped_examples"] += int(result["dropped"])
            save_state(state_path, state)
            pending = []
            if state["accepted_examples"] >= args.max_examples:
                break

        if pending and state["accepted_examples"] <= args.max_examples:
            result = flush_examples(
                pending,
                tokenizer=tokenizer,
                model=model,
                device=device,
                translation_batch_size=args.translation_batch_size,
                num_beams=args.num_beams,
                max_source_tokens=args.max_source_tokens,
                max_new_tokens=args.max_new_tokens,
                min_cyrillic_ratio=args.min_cyrillic_ratio,
                original_handle=original_handle,
                translated_handle=translated_handle,
                bilingual_handle=bilingual_handle,
            )
            state["written_examples"] += int(result["written"])
            state["dropped_examples"] += int(result["dropped"])

    summary = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "allowed_sources": sorted(allowed_sources),
        "raw_rows_seen": state["raw_rows_seen"],
        "accepted_examples": state["accepted_examples"],
        "written_examples": state["written_examples"],
        "dropped_examples": state["dropped_examples"],
        "rows_written_bilingual": state["written_examples"] * 2,
        "translation_model": args.model_name,
        "device": device,
        "outputs": {
            "original_en": str(original_path),
            "translated_bg": str(translated_path),
            "bilingual": str(bilingual_path),
        },
    }
    save_state(state_path, state)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    os._exit(0)
