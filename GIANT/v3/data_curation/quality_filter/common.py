from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
CYRILLIC_RE = re.compile(r"[А-Яа-яЁёЍѝ]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letter_count = sum(ch.isalpha() for ch in text)
    if letter_count == 0:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / float(letter_count)


def tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def hash_token(token: str, vocab_size: int) -> int:
    return 1 + (stable_hash(token) % max(vocab_size - 1, 1))


def split_paragraph_windows(text: str, *, min_chars: int, max_chars: int, target_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    windows: list[str] = []
    current: list[str] = []
    current_chars = 0

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if len(paragraph) > max_chars:
            for start in range(0, len(paragraph), target_chars):
                chunk = paragraph[start : start + max_chars].strip()
                if len(chunk) >= min_chars:
                    windows.append(chunk)
            continue

        projected = current_chars + len(paragraph) + (1 if current else 0)
        if current and projected > max_chars:
            candidate = normalize_text("\n\n".join(current))
            if len(candidate) >= min_chars:
                windows.append(candidate)
            current = [paragraph]
            current_chars = len(paragraph)
            continue

        current.append(paragraph)
        current_chars = projected
        if current_chars >= target_chars:
            candidate = normalize_text("\n\n".join(current))
            if len(candidate) >= min_chars:
                windows.append(candidate)
            current = []
            current_chars = 0

    if current:
        candidate = normalize_text("\n\n".join(current))
        if len(candidate) >= min_chars:
            windows.append(candidate)
    return windows


def make_text(row: dict[str, Any], source_cfg: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for field in source_cfg.get("prefix_fields", []):
        value = row.get(field)
        if value is not None:
            text = normalize_text(str(value))
            if text:
                parts.append(text)
    for field in source_cfg.get("text_fields", []):
        value = row.get(field)
        if value is not None:
            text = normalize_text(str(value))
            if text:
                parts.append(text)
    if not parts and source_cfg.get("text_field"):
        value = row.get(source_cfg["text_field"])
        if value is not None:
            text = normalize_text(str(value))
            if text:
                parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def row_matches(row: dict[str, Any], source_cfg: dict[str, Any]) -> bool:
    equals = source_cfg.get("field_equals", {}) or {}
    for key, expected in equals.items():
        if row.get(key) != expected:
            return False
    includes = source_cfg.get("field_in", {}) or {}
    for key, options in includes.items():
        if row.get(key) not in set(options):
            return False
    excludes = source_cfg.get("field_not_in", {}) or {}
    for key, options in excludes.items():
        if row.get(key) in set(options):
            return False
    return True


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows: Sequence[dict[str, Any]], *, seed: int, split_fracs: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["label_name"])].append(dict(row))

    rng = random.Random(seed)
    out = {name: [] for name in split_fracs}
    split_names = list(split_fracs)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        total = len(bucket_rows)
        start = 0
        for idx, split_name in enumerate(split_names):
            if idx == len(split_names) - 1:
                end = total
            else:
                end = start + int(round(total * float(split_fracs[split_name])))
            out[split_name].extend(bucket_rows[start:end])
            start = end
    for split_name in out:
        rng.shuffle(out[split_name])
    return out


def summarize_labels(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["label_name"]) for row in rows)
    return dict(sorted(counts.items()))
