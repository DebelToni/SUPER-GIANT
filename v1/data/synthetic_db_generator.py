"""
synthetic_db_generator.py
Generate a synthetic text dataset by querying a local LLM that exposes
an OpenAI-compatible HTTP API (e.g. Ollama with --api-base).

Example
-------
$ python synthetic_db_generator.py \
    --prompt "Generate a short 300-word story wrapped in triple back-ticks." \
    --model "llama3:8b" \
    --num_samples 1_000 \
    --out_dir data/tiny_stories \
    --base_url http://localhost:11434/v1 \
    --api_key "ollama"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

from datasets import Dataset, Features, Value
from openai import OpenAI
from tqdm.auto import tqdm

# ────────────────────────────────
# helper: pull the first ``` … ``` block (or fall back to full text)
# ────────────────────────────────
_BLOCK_RE = re.compile(r"```(?:[\w+-]*\n)?(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ────────────────────────────────
# main
# ────────────────────────────────
def main() -> None:
    cli = argparse.ArgumentParser("Synthetic dataset generator")
    cli.add_argument("--prompt", required=True, help="User prompt.")
    cli.add_argument("-m", "--model", default="llama3", help="Model name.")
    cli.add_argument("-n", "--num_samples", type=int, default=100,
                     help="How many rows to generate.")
    cli.add_argument("--out_dir", type=Path, default=Path("synthetic_ds"),
                     help="Destination directory (HF Arrow format).")
    cli.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL",
                     "http://localhost:11434/v1"),
                     help="OpenAI-compatible endpoint.")
    cli.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "ollama"),
                     help="Dummy key used by most local servers.")
    cli.add_argument("--batch_size", type=int, default=4,
                     help="Parallel requests in flight.")
    cli.add_argument("--max_retries", type=int, default=3,
                     help="Retries per sample on failure.")
    args = cli.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # client per latest openai-python SDK (v1.x)
    client = OpenAI(base_url=args.base_url,
                    api_key=args.api_key,
                    timeout=60.0)

    rows: List[dict] = []
    pbar = tqdm(total=args.num_samples, desc="Generating", unit="sample")

    i = 0
    while i < args.num_samples:
        batch = min(args.batch_size, args.num_samples - i)
        payloads = [{
            "model": args.model,
            "messages": [
                {"role": "system",
                 "content": "You are a high-quality synthetic data generator."},
                {"role": "user", "content": args.prompt}
            ]
        } for _ in range(batch)]

        responses = []
        for item in payloads:
            err = None
            for attempt in range(args.max_retries):
                try:
                    responses.append(client.chat.completions.create(**item))
                    break
                except Exception as exc:     # noqa: BLE001
                    err = exc
                    time.sleep(2 ** attempt)
            else:  # ran out of retries
                raise RuntimeError(f"Failed after {args.max_retries} retries") from err

        for r in responses:
            text = r.choices[0].message.content
            rows.append({"text": extract_code(text)})
            i += 1
            pbar.update(1)

    pbar.close()

    # pack into a HF Arrow dataset
    ds = Dataset.from_list(rows, features=Features({"text": Value("string")}))
    ds.save_to_disk(str(args.out_dir))

    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "prompt": args.prompt,
            "model": args.model,
            "base_url": args.base_url,
            "num_samples": args.num_samples,
            "created_utc": datetime.utcnow().isoformat() + "Z"
        }, fh, indent=2)

    print(f"✓ Saved {len(ds):,} rows to {args.out_dir}")


if __name__ == "__main__":
    main()

