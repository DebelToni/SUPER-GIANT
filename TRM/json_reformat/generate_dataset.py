from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from urllib import request

def build_vocab(pairs: List[Tuple[str, str]]) -> List[str]:
    chars = set()
    for raw, fixed in pairs:
        chars.update(raw)
        chars.update(fixed)
    return ["<pad>", "<unk>"] + sorted(chars)


def save_vocab(chars: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"chars": chars, "pad_token": "<pad>", "unk_token": "<unk>"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


SYSTEM_PROMPT = (
    "You generate messy JSON-like records and their cleaned JSON. "
    "Return only a JSON array, nothing else."
)

USER_PROMPT_TEMPLATE = """
Generate {count} examples. Each element is an object with keys:
- raw: a single-line, ASCII, JSON-ish string with inconsistent formatting (e.g., single quotes, trailing commas, unquoted keys, wrong boolean case, inconsistent date formats).
- fixed: the corrected strict JSON (double quotes, lowercase true/false/null, ISO dates like YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ), preserving the same data.
Rules:
- No tabs or newlines in raw/fixed.
- ASCII only.
- Max {max_chars} characters for raw and fixed.
- Make each example meaningfully different.
""".strip()


@dataclass
class Example:
    raw: str
    fixed: str


def _normalize(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.strip().split())


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _parse_json_array(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: parse line-by-line objects
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items if items else None


def call_deepseek(api_key: str, base_url: str, *, count: int, max_chars: int, temperature: float) -> List[dict]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(count=count, max_chars=max_chars)},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"DeepSeek error: {resp.status}")
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = _parse_json_array(content)
    if parsed is None:
        raise ValueError("Could not parse JSON array from response")
    if isinstance(parsed, dict) and "examples" in parsed:
        parsed = parsed["examples"]
    if not isinstance(parsed, list):
        raise ValueError("Response JSON is not a list")
    return parsed


def filter_examples(raw_items: List[dict], *, max_chars: int) -> List[Example]:
    results: List[Example] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw = _normalize(str(item.get("raw", "")))
        fixed = _normalize(str(item.get("fixed", "")))
        if not raw or not fixed:
            continue
        if not _is_ascii(raw) or not _is_ascii(fixed):
            continue
        if len(raw) > max_chars or len(fixed) > max_chars:
            continue
        if raw == fixed:
            continue
        try:
            fixed_obj = json.loads(fixed)
        except json.JSONDecodeError:
            continue
        fixed = json.dumps(fixed_obj, separators=(",", ":"), sort_keys=True)
        if len(fixed) > max_chars:
            continue
        results.append(Example(raw=raw, fixed=fixed))
    return results


def write_jsonl(path: Path, examples: List[Example], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps({"raw": ex.raw, "fixed": ex.fixed}, ensure_ascii=True))
            handle.write("\n")


def load_existing(path: Path) -> List[Example]:
    if not path.exists():
        return []
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = _normalize(str(obj.get("raw", "")))
            fixed = _normalize(str(obj.get("fixed", "")))
            if raw and fixed:
                examples.append(Example(raw=raw, fixed=fixed))
    return examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate JSON reformat dataset with DeepSeek")
    p.add_argument("--out_dir", default="dataset_artifacts/json_reformat")
    p.add_argument("--train_size", type=int, default=10)
    p.add_argument("--val_size", type=int, default=0)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--max_chars", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com").strip()
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set")

    total = int(args.train_size) + int(args.val_size)
    if total <= 0:
        raise ValueError("train_size + val_size must be > 0")

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    vocab_path = out_dir / "vocab.json"

    if args.overwrite:
        if train_path.exists():
            train_path.unlink()
        if val_path.exists():
            val_path.unlink()

    existing_train = load_existing(train_path)
    existing_val = load_existing(val_path)

    seen = {ex.raw for ex in existing_train + existing_val}
    collected: List[Example] = existing_train + existing_val
    train_count = len(existing_train)
    val_count = len(existing_val)
    retries = 0

    while train_count < args.train_size or val_count < args.val_size:
        remaining = (args.train_size - train_count) + (args.val_size - val_count)
        need = min(args.batch, remaining)
        try:
            items = call_deepseek(api_key, base_url, count=need, max_chars=args.max_chars, temperature=args.temperature)
        except Exception as exc:
            retries += 1
            if retries > args.max_retries:
                raise RuntimeError(f"Failed after {args.max_retries} retries: {exc}") from exc
            time.sleep(args.sleep)
            continue

        retries = 0
        filtered = filter_examples(items, max_chars=args.max_chars)
        for ex in filtered:
            if ex.raw in seen:
                continue
            seen.add(ex.raw)
            collected.append(ex)
            if train_count < args.train_size:
                write_jsonl(train_path, [ex], append=True)
                train_count += 1
            elif val_count < args.val_size:
                write_jsonl(val_path, [ex], append=True)
                val_count += 1
            if train_count >= args.train_size and val_count >= args.val_size:
                break

        print(
            f"[progress] train={train_count}/{args.train_size} val={val_count}/{args.val_size} "
            f"kept={len(filtered)}"
        )
        time.sleep(args.sleep)

    if not val_path.exists():
        write_jsonl(val_path, [], append=False)

    vocab = build_vocab([(ex.raw, ex.fixed) for ex in collected])
    save_vocab(vocab, vocab_path)

    print(f"[dataset] train={train_count} val={val_count}")
    print(f"[dataset] saved: {train_path}")
    print(f"[dataset] vocab: {vocab_path} size={len(vocab)}")
    if collected:
        print("[sample raw]", collected[0].raw)
        print("[sample fixed]", collected[0].fixed)


if __name__ == "__main__":
    main()
