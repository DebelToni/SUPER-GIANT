from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-N curated candidate rows per target.")
    parser.add_argument("--input", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--per-target", type=int, default=1, help="Rows to keep per target")
    parser.add_argument("--min-heuristic", type=float, default=None, help="Drop rows below this heuristic score")
    parser.add_argument(
        "--drop-may-refer-to",
        action="store_true",
        help="Drop rows whose lead looks like a disambiguation page ('may refer to').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    may_refer_to_re = re.compile(r"\bmay refer to\b", re.IGNORECASE)

    kept: Dict[str, List[dict]] = {}
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            heuristic = row.get("score_heuristic")
            if args.min_heuristic is not None and (heuristic is None or float(heuristic) < float(args.min_heuristic)):
                continue
            if args.drop_may_refer_to:
                lead = str(row.get("text_lead") or "")
                if may_refer_to_re.search(lead):
                    continue
            target_id = str(row.get("target_id"))
            bucket = kept.setdefault(target_id, [])
            if len(bucket) < args.per_target:
                bucket.append(row)

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for target_id in sorted(kept):
            for row in kept[target_id]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()
