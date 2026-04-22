from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split filtered JSONL into tier-specific views.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "high_quality": output_dir / "high_quality.jsonl",
        "acceptable": output_dir / "acceptable.jsonl",
        "keep_all": output_dir / "keep_all.jsonl",
    }
    counts = Counter()

    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        with input_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                tier = str(row.get("tier") or "")
                if tier not in {"high_quality", "acceptable"}:
                    continue
                counts[tier] += 1
                handles[tier].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles["keep_all"].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "input": str(input_path),
        "counts": dict(counts),
        "keep_all": int(counts["high_quality"] + counts["acceptable"]),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "tier_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
