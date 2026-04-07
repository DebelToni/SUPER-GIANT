from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from GIANT.v3.Long.longdsl import surface_tokens


BENCHMARK_INSTRUCTIONS = """You are solving the LongGIANT hidden-world benchmark.
The records describe a small world in natural language.
Important rules:
- names can be aliases of the same entity
- later updates override earlier values
- the question asks for the current final value tied to the queried entity
- the correct answer is always exactly one token already supported by the record
Return only the final one-token answer. Do not explain your reasoning. Do not add punctuation or labels."""

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ANSWER_HINT_RE = re.compile(
    r"(?:final answer|answer|output|return)\s*(?:is|:)?\s*[\"'`]?([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
QUOTED_TOKEN_RE = re.compile(r"[\"'`]([A-Za-z0-9_-]+)[\"'`]")


@dataclass
class EvalResult:
    index: int
    gold_answer: str
    raw_output: str
    parsed_answer: str
    strict_correct: bool
    parsed_correct: bool
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_user_message(row: dict[str, Any]) -> str:
    if "messages" in row:
        for message in row["messages"]:
            if str(message.get("role")) == "user":
                return str(message.get("content", "")).strip()
    if "context" in row and "question" in row:
        return f"Context: {row['context']} Question: {row['question']}"
    if "text" in row:
        text = str(row["text"])
        marker = " Answer: "
        if marker in text:
            return text.split(marker, 1)[0]
        return text
    raise KeyError("Could not extract user prompt from row")


def extract_gold_answer(row: dict[str, Any]) -> str:
    if "answer" in row:
        return str(row["answer"]).strip()
    if "messages" in row:
        for message in row["messages"]:
            if str(message.get("role")) == "assistant":
                content = str(message.get("content", "")).strip()
                if content.lower().startswith("answer:"):
                    return content.split(":", 1)[1].strip()
                return content
    raise KeyError("Could not extract gold answer from row")


def build_prompt(example_row: dict[str, Any], eval_row: dict[str, Any]) -> str:
    example_user = extract_user_message(example_row)
    example_answer = extract_gold_answer(example_row)
    eval_user = extract_user_message(eval_row)
    return (
        "Solved example:\n"
        f"{example_user}\n"
        f"Answer: {example_answer}\n\n"
        "Now solve this new example. Reply with only the one-token answer.\n"
        f"{eval_user}"
    )


def parse_answer(raw_output: str) -> str:
    text = THINK_RE.sub(" ", raw_output).strip().strip("`")
    if not text:
        return ""
    answer_hints = ANSWER_HINT_RE.findall(text)
    if answer_hints:
        return answer_hints[-1]
    quoted = QUOTED_TOKEN_RE.findall(text)
    if quoted:
        return quoted[-1]
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = (nonempty_lines[-1] if nonempty_lines else text).strip()
    lower = candidate.lower()
    if lower.startswith("answer"):
        parts = candidate.split(":", 1)
        if len(parts) == 2:
            candidate = parts[1].strip()
        else:
            candidate = candidate[len("answer") :].strip(" :-")
    tokens = [tok for tok in surface_tokens(candidate) if tok not in {":", "-"}]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return tokens[-1]


def call_openai(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    max_retries: int,
    api_mode: str,
    disable_thinking: bool,
) -> tuple[str, dict[str, int | None]]:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if api_mode == "chat":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_output_tokens,
                    temperature=0,
                    extra_body=({"think": False} if disable_thinking else None),
                )
                usage = getattr(response, "usage", None)
                message = response.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning", None) or ""
                combined_text = str(content).strip() or str(reasoning).strip()
                usage_dict = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                return combined_text, usage_dict

            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
            )
            usage = getattr(response, "usage", None)
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            return response.output_text.strip(), usage_dict
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"OpenAI request failed after {max_retries} attempts: {last_error}")


def call_ollama_native(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    max_retries: int,
    disable_thinking: bool,
) -> tuple[str, dict[str, int | None]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_output_tokens},
    }
    if disable_thinking:
        payload["think"] = False

    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            message = body.get("message", {})
            usage_dict = {
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
                "total_tokens": (body.get("prompt_eval_count") or 0) + (body.get("eval_count") or 0),
            }
            return str(message.get("content", "")).strip(), usage_dict
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Ollama native request failed after {max_retries} attempts: {last_error}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            done.add(int(payload["index"]))
    return done


def summarize(
    results_path: Path,
    summary_path: Path,
    *,
    model: str,
    example_index: int,
    limit: int,
    api_mode: str,
    base_url: str | None,
) -> dict[str, Any]:
    results = load_jsonl(results_path)
    strict_correct = sum(1 for row in results if bool(row["strict_correct"]))
    parsed_correct = sum(1 for row in results if bool(row["parsed_correct"]))
    input_tokens = sum(int(row["input_tokens"]) for row in results if row.get("input_tokens") is not None)
    output_tokens = sum(int(row["output_tokens"]) for row in results if row.get("output_tokens") is not None)
    total_tokens = sum(int(row["total_tokens"]) for row in results if row.get("total_tokens") is not None)
    summary = {
        "model": model,
        "api_mode": api_mode,
        "base_url": base_url,
        "n_examples": len(results),
        "requested_examples": limit,
        "example_index": example_index,
        "strict_exact_match": strict_correct / max(len(results), 1),
        "strict_exact_match_pct": 100.0 * strict_correct / max(len(results), 1),
        "parsed_exact_match": parsed_correct / max(len(results), 1),
        "parsed_exact_match_pct": 100.0 * parsed_correct / max(len(results), 1),
        "n_strict_correct": strict_correct,
        "n_parsed_correct": parsed_correct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def default_output_dir(model: str, limit: int, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_")
    return Path("/proj/giant-data/GIANT/Long/benchmarks/openai") / f"{stamp}_{safe_model}_val{limit}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LongGIANT rows with an OpenAI model")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--train_jsonl", default="/proj/giant-data/GIANT/Long/records/raw/level1_ctx512/train.jsonl")
    parser.add_argument("--eval_jsonl", default="/proj/giant-data/GIANT/Long/records/raw/level1_ctx512/val.jsonl")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example_index", type=int, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--api_mode", choices=["auto", "responses", "chat", "ollama_native"], default="auto")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if args.base_url:
        api_key = api_key or "ollama"
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    train_rows = load_jsonl(Path(args.train_jsonl))
    eval_rows = load_jsonl(Path(args.eval_jsonl), limit=args.limit)
    if not train_rows:
        raise RuntimeError("No train rows found")
    if not eval_rows:
        raise RuntimeError("No eval rows found")

    rng = random.Random(args.seed)
    example_index = args.example_index if args.example_index is not None else rng.randrange(len(train_rows))
    example_row = train_rows[example_index]

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model, args.limit, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    prompt_path = output_dir / "prompt_scaffold.txt"

    prompt_path.write_text(
        BENCHMARK_INSTRUCTIONS
        + "\n\n"
        + build_prompt(example_row, eval_rows[0]),
        encoding="utf-8",
    )

    completed = load_completed_indices(results_path) if args.resume else set()
    api_mode = args.api_mode
    if api_mode == "auto":
        api_mode = "chat" if args.base_url else "responses"

    client = None
    if api_mode != "ollama_native":
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)

    for idx, row in enumerate(eval_rows):
        if idx in completed:
            continue
        user_prompt = build_prompt(example_row, row)
        if api_mode == "ollama_native":
            if not args.base_url:
                raise RuntimeError("--base_url is required for api_mode=ollama_native")
            raw_output, usage = call_ollama_native(
                base_url=args.base_url,
                model=args.model,
                system_prompt=BENCHMARK_INSTRUCTIONS,
                user_prompt=user_prompt,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                disable_thinking=args.disable_thinking,
            )
        else:
            raw_output, usage = call_openai(
                client,
                model=args.model,
                system_prompt=BENCHMARK_INSTRUCTIONS,
                user_prompt=user_prompt,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                api_mode=api_mode,
                disable_thinking=args.disable_thinking,
            )
        gold_answer = extract_gold_answer(row)
        parsed_answer = parse_answer(raw_output)
        result = EvalResult(
            index=idx,
            gold_answer=gold_answer,
            raw_output=raw_output,
            parsed_answer=parsed_answer,
            strict_correct=(raw_output.strip() == gold_answer),
            parsed_correct=(parsed_answer == gold_answer),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
        )
        append_jsonl(results_path, asdict(result))
        print(
            f"[openai-long] {idx + 1}/{len(eval_rows)} gold={gold_answer} "
            f"parsed={parsed_answer} correct={result.parsed_correct}"
        )

    summary = summarize(
        results_path,
        summary_path,
        model=args.model,
        example_index=example_index,
        limit=args.limit,
        api_mode=api_mode,
        base_url=args.base_url,
    )
    print(json.dumps(summary, indent=2))
    print(f"[openai-long] results={results_path}")
    print(f"[openai-long] summary={summary_path}")


if __name__ == "__main__":
    main()
