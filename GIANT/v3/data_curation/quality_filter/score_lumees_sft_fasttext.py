from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from flax import serialization
from huggingface_hub import HfApi, hf_hub_download

from GIANT.v3.data_curation.quality_filter.common import hash_token, normalize_text, tokenize_words
from GIANT.v3.data_curation.quality_filter.train_quality_filter import FastTextClassifier


TOTAL_SFT_ROWS = 8_663_195


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score lumees SFT conversations with the trained fastText filter.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--repo_id", default="lumees/bulgarian-corpus-33b")
    parser.add_argument("--subset", default="sft")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--download_workers", type=int, default=3)
    parser.add_argument("--high_quality_threshold", type=float, default=0.9)
    parser.add_argument("--keep_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
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


def flatten_messages(messages) -> str:
    parts = []
    for message in messages or []:
        role = normalize_text(str(message.get("role") or ""))
        content = normalize_text(str(message.get("content") or ""))
        if not content:
            continue
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


def download_one(repo_id: str, shard_name: str) -> dict[str, object]:
    started = time.perf_counter()
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=shard_name)
    ended = time.perf_counter()
    return {
        "shard": shard_name,
        "local_path": local_path,
        "download_time_s": ended - started,
        "size_bytes": Path(local_path).stat().st_size,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
    labels = list(metadata["labels"])
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
    selected_shards = sorted(path for path in api.list_repo_files(args.repo_id, repo_type="dataset") if path.startswith(f"{args.subset}/") and path.endswith(".parquet"))

    samples = {"high_quality": [], "acceptable": [], "reject": []}
    tier_counts = Counter()
    source_counts = defaultdict(Counter)
    download_stats = []
    process_stats = []
    rows_processed = 0
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        future_map = {pool.submit(download_one, args.repo_id, shard): shard for shard in selected_shards}
        for future in as_completed(future_map):
            download_info = future.result()
            download_stats.append(download_info)
            shard_name = str(download_info["shard"])
            process_start = time.perf_counter()
            parquet = pq.ParquetFile(str(download_info["local_path"]))
            shard_rows = 0
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["messages", "source"]):
                data = batch.to_pydict()
                texts = [flatten_messages(messages) for messages in data["messages"]]
                token_ids, mask = encode_batch(texts, vocab_size=vocab_size, max_tokens=max_tokens)
                probs = np.asarray(predict(jnp.asarray(token_ids), jnp.asarray(mask)))
                keep_probs = probs[:, keep_idx]
                for idx, keep_prob in enumerate(keep_probs):
                    if keep_prob >= float(args.high_quality_threshold):
                        tier = "high_quality"
                    elif keep_prob >= float(args.keep_threshold):
                        tier = "acceptable"
                    else:
                        tier = "reject"
                    tier_counts[tier] += 1
                    source = str(data["source"][idx])
                    source_counts[source][tier] += 1
                    rows_processed += 1
                    shard_rows += 1
                    if len(samples[tier]) < 10:
                        samples[tier].append(
                            {
                                "source": source,
                                "keep_probability": float(keep_prob),
                                "text": texts[idx][:1600],
                            }
                        )
            process_stats.append({
                "shard": shard_name,
                "rows": shard_rows,
                "process_time_s": time.perf_counter() - process_start,
            })

    wall_time_s = time.perf_counter() - overall_start
    docs_per_second = rows_processed / max(wall_time_s, 1e-6)
    summary = {
        "repo_id": args.repo_id,
        "subset": args.subset,
        "num_selected_shards": len(selected_shards),
        "rows_processed": rows_processed,
        "sample_fraction_actual": rows_processed / float(TOTAL_SFT_ROWS),
        "wall_time_s": wall_time_s,
        "docs_per_second": docs_per_second,
        "tier_counts": dict(tier_counts),
        "tier_ratios": {key: value / max(rows_processed, 1) for key, value in tier_counts.items()},
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "download_stats": download_stats,
        "process_stats": process_stats,
        "thresholds": {
            "high_quality_threshold": args.high_quality_threshold,
            "keep_threshold": args.keep_threshold,
        },
        "note": "Conversation rows were flattened as role-prefixed text and scored by a fastText model trained on Bulgarian quality filtering.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Lumees SFT FastText Samples",
        "",
        f"Rows processed: `{rows_processed}`",
        "",
    ]
    for tier in ["high_quality", "acceptable", "reject"]:
        lines.append(f"## {tier}")
        lines.append("")
        for idx, sample in enumerate(samples[tier], start=1):
            lines.append(f"### Example {idx}")
            lines.append(f"- source: `{sample['source']}`")
            lines.append(f"- keep_probability: `{sample['keep_probability']:.4f}`")
            lines.append("")
            lines.append(sample["text"])
            lines.append("")
    (output_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
