from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-bg"
DEFAULT_TEXT_FIELDS = ("text", "text_lead", "text_full_truncated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate curated booster rows and emit a bilingual JSONL.")
    parser.add_argument(
        "--input",
        default="/proj/giant-data/GIANT/GIANT-Chat/data_curation/strong56_wikipedia_v2/top3_candidates.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="/proj/giant-data/GIANT/GIANT-Chat/data_curation/curated_booster_bg_en_v1",
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_source_tokens", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_text(row: dict, fields: Iterable[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(text.strip() for text in decoded)
    return outputs


def split_text_into_chunks(text: str, *, max_chars: int = 600) -> List[str]:
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if args.max_rows is not None:
        rows = rows[: max(0, int(args.max_rows))]
    source_rows = []
    texts = []
    chunk_spans: List[tuple[int, int]] = []
    for row in rows:
        text = pick_text(row, DEFAULT_TEXT_FIELDS)
        if not text:
            continue
        source_rows.append(row)
        start = len(texts)
        texts.extend(split_text_into_chunks(text))
        chunk_spans.append((start, len(texts)))

    device = select_device()
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name).to(device)
    model.eval()

    translations = translate_texts(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=max(1, args.batch_size),
        num_beams=max(1, args.num_beams),
        max_source_tokens=max(32, args.max_source_tokens),
        max_new_tokens=max(32, args.max_new_tokens),
    )

    bilingual_path = output_dir / "bilingual_booster.jsonl"
    with bilingual_path.open("w", encoding="utf-8") as handle:
        for row, (start, end) in zip(source_rows, chunk_spans):
            en_text = "\n\n".join(texts[start:end]).strip()
            bg_text = "\n\n".join(translations[start:end]).strip()
            common = {
                "target_id": row.get("target_id"),
                "canonical_name": row.get("canonical_name"),
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "url": row.get("url"),
            }
            handle.write(
                json.dumps(
                    {
                        **common,
                        "language": "en",
                        "variant": "original_en",
                        "text": en_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {
                        **common,
                        "language": "bg",
                        "variant": "translated_bg",
                        "text": bg_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "input": str(input_path),
        "output": str(bilingual_path),
        "rows_read": len(rows),
        "rows_kept": len(source_rows),
        "rows_written": len(source_rows) * 2,
        "translation_model": args.model_name,
        "device": device,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
