from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from flax import serialization
from huggingface_hub import HfApi, hf_hub_download

from GIANT.v3.data_curation.quality_filter.common import hash_token, load_yaml, normalize_text, tokenize_words
from GIANT.v3.data_curation.quality_filter.train_quality_filter import FastTextClassifier


TOTAL_PRETRAIN_ROWS = 26_278_393


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-filter the lumees Bulgarian pretrain corpus with the trained fastText baseline.")
    parser.add_argument("--run_dir", required=True, help="Path under /proj/giant-data/GIANT containing model_params.msgpack")
    parser.add_argument("--sample_fraction", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--high_quality_threshold", type=float, default=0.9)
    parser.add_argument("--keep_threshold", type=float, default=0.5)
    parser.add_argument("--repo_id", default="lumees/bulgarian-corpus-33b")
    parser.add_argument("--subset", default="pretrain")
    parser.add_argument("--output_subdir", default="filter_runs/lumees_pretrain_fasttext_5pct")
    return parser.parse_args()


def encode_batch(texts: list[str], *, vocab_size: int, max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.zeros((len(texts), max_tokens), dtype=np.int32)
    mask = np.zeros((len(texts), max_tokens), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = tokenize_words(text)[:max_tokens]
        ids = [hash_token(token, vocab_size) for token in tokens]
        if not ids:
            ids = [1]
        token_ids[row_idx, : len(ids)] = np.asarray(ids, dtype=np.int32)
        mask[row_idx, : len(ids)] = 1.0
    return token_ids, mask


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata["model_type"] != "fasttext":
        raise ValueError(f"This script currently supports only fasttext runs, got {metadata['model_type']!r}")

    labels = list(metadata["labels"])
    if "keep" not in labels or "reject" not in labels:
        raise ValueError(f"Expected labels to contain keep/reject, got {labels}")
    keep_idx = labels.index("keep")
    vocab_size = int(metadata["features"]["vocab_size"])
    max_tokens = int(metadata["features"]["max_tokens"])
    model_cfg = metadata["models"]["fasttext"]

    model = FastTextClassifier(
        vocab_size=vocab_size,
        embed_dim=int(model_cfg["embed_dim"]),
        num_classes=len(labels),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
    )
    params = serialization.from_bytes(None, (run_dir / "model_params.msgpack").read_bytes())

    @jax.jit
    def predict(token_ids, mask):
        logits = model.apply({"params": params}, token_ids, mask, train=False)
        return jax.nn.softmax(logits, axis=-1)

    api = HfApi()
    shard_paths = sorted(path for path in api.list_repo_files(args.repo_id, repo_type="dataset") if path.startswith(f"{args.subset}/") and path.endswith(".parquet"))
    sample_count = max(1, int(round(len(shard_paths) * float(args.sample_fraction))))
    shard_indices = sorted(set(int(round(x)) for x in np.linspace(0, len(shard_paths) - 1, num=sample_count)))
    selected_shards = [shard_paths[idx] for idx in shard_indices]

    output_root = run_dir.parent / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    kept_path = output_root / "kept.jsonl"

    total_rows = 0
    tier_counts = Counter()
    source_counts = defaultdict(Counter)
    kept_rows = 0
    start = time.perf_counter()

    with kept_path.open("w", encoding="utf-8") as out:
        for shard_name in selected_shards:
            local_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=shard_name)
            parquet = pq.ParquetFile(local_path)
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["text", "source", "meta"]):
                data = batch.to_pydict()
                texts = [normalize_text(text or "") for text in data["text"]]
                token_ids, mask = encode_batch(texts, vocab_size=vocab_size, max_tokens=max_tokens)
                probs = np.asarray(predict(jnp.asarray(token_ids), jnp.asarray(mask)))
                keep_probs = probs[:, keep_idx]

                for idx, keep_prob in enumerate(keep_probs):
                    tier = "reject"
                    if keep_prob >= float(args.high_quality_threshold):
                        tier = "high_quality"
                    elif keep_prob >= float(args.keep_threshold):
                        tier = "acceptable"
                    tier_counts[tier] += 1
                    source = str(data["source"][idx])
                    source_counts[source][tier] += 1
                    total_rows += 1

                    if tier == "reject":
                        continue

                    kept_rows += 1
                    payload = {
                        "source": source,
                        "tier": tier,
                        "keep_probability": float(keep_prob),
                        "text": texts[idx],
                        "meta": data["meta"][idx],
                        "shard": shard_name,
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start
    docs_per_second = total_rows / max(elapsed, 1e-6)
    sample_fraction_actual = total_rows / float(TOTAL_PRETRAIN_ROWS)
    full_eta_seconds = TOTAL_PRETRAIN_ROWS / max(docs_per_second, 1e-6)
    summary = {
        "repo_id": args.repo_id,
        "subset": args.subset,
        "selected_shards": selected_shards,
        "num_selected_shards": len(selected_shards),
        "rows_processed": total_rows,
        "rows_kept": kept_rows,
        "sample_fraction_actual": sample_fraction_actual,
        "wall_time_s": elapsed,
        "docs_per_second": docs_per_second,
        "estimated_full_runtime_s": full_eta_seconds,
        "estimated_full_runtime_h": full_eta_seconds / 3600.0,
        "tier_counts": dict(tier_counts),
        "tier_ratios": {key: value / max(total_rows, 1) for key, value in tier_counts.items()},
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "thresholds": {
            "high_quality_threshold": args.high_quality_threshold,
            "keep_threshold": args.keep_threshold,
        },
        "note": "high_quality/acceptable/reject are heuristic score bands derived from a binary keep-vs-reject fasttext model",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
