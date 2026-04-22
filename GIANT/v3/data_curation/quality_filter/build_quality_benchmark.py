from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from datasets import load_dataset

from GIANT.v3.data_curation.quality_filter.common import (
    cyrillic_ratio,
    load_yaml,
    make_text,
    merge_dicts,
    normalize_text,
    row_matches,
    split_paragraph_windows,
    stable_hash,
    stratified_split,
    summarize_labels,
    write_jsonl,
)


OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled Bulgarian text-quality benchmark.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _resolve_output_root(base_cfg: dict, cfg: dict) -> Path:
    data_root = Path(base_cfg["paths"]["data_root"])
    rel = cfg["output"]["benchmark_dir"]
    return data_root / rel


def _set_hf_cache(base_cfg: dict) -> None:
    hf_root = base_cfg["paths"].get("hf_cache_root")
    if hf_root:
        os.environ["HF_HOME"] = hf_root
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_root) / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_root) / "transformers")


def _iter_rows(source_cfg: dict):
    if source_cfg.get("source_type") == "oai_pmh":
        return _iter_oai_rows(source_cfg)
    kwargs = {}
    config_name = source_cfg.get("config_name")
    if config_name is not None:
        kwargs["name"] = config_name
    split = source_cfg.get("split", "train")
    streaming = bool(source_cfg.get("streaming", False))
    return load_dataset(
        source_cfg["dataset"],
        split=split,
        streaming=streaming,
        **kwargs,
    )


def _pick_bulgarian_variant(values: list[str]) -> str | None:
    best = None
    best_ratio = -1.0
    for raw in values:
        if raw is None:
            continue
        candidates = [raw]
        if "///" in raw:
            candidates.extend(part.strip() for part in raw.split("///") if part.strip())
        for candidate in candidates:
            text = normalize_text(candidate)
            if not text:
                continue
            ratio = cyrillic_ratio(text)
            if ratio > best_ratio:
                best_ratio = ratio
                best = text
    return best


def _iter_oai_rows(source_cfg: dict):
    endpoint = source_cfg["endpoint"]
    metadata_prefix = source_cfg.get("metadata_prefix", "oai_dc")
    timeout = int(source_cfg.get("timeout", 60))
    resumption_token = None

    while True:
        if resumption_token:
            query = urlencode({"verb": "ListRecords", "resumptionToken": resumption_token})
        else:
            query = urlencode({"verb": "ListRecords", "metadataPrefix": metadata_prefix})
        with urlopen(f"{endpoint}?{query}", timeout=timeout) as response:
            root = ET.fromstring(response.read())

        for record in root.findall(".//oai:record", OAI_NS):
            header = record.find("oai:header", OAI_NS)
            if header is not None and header.attrib.get("status") == "deleted":
                continue
            metadata = record.find("oai:metadata", OAI_NS)
            if metadata is None:
                continue

            titles = [elem.text or "" for elem in metadata.findall(".//dc:title", OAI_NS)]
            descriptions = [elem.text or "" for elem in metadata.findall(".//dc:description", OAI_NS)]
            identifiers = [elem.text or "" for elem in metadata.findall(".//dc:identifier", OAI_NS)]
            yield {
                "title_bg": _pick_bulgarian_variant(titles),
                "abstract_bg": _pick_bulgarian_variant(descriptions),
                "identifier": next((item for item in identifiers if item), None),
            }

        token_elem = root.find(".//oai:resumptionToken", OAI_NS)
        resumption_token = None if token_elem is None else (token_elem.text or "").strip() or None
        if not resumption_token:
            break


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(Path(__file__).resolve().parent / "Config.yml")
    cfg = merge_dicts(base_cfg, load_yaml(args.config))
    _set_hf_cache(base_cfg)

    benchmark_cfg = cfg["benchmark"]
    output_root = _resolve_output_root(base_cfg, cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    label_order = list(cfg["labels"])
    label_to_id = {label: idx for idx, label in enumerate(label_order)}

    min_chars = int(benchmark_cfg["text"]["min_chars"])
    max_chars = int(benchmark_cfg["text"]["max_chars"])
    target_chars = int(benchmark_cfg["text"]["target_chars"])
    min_cyr = float(benchmark_cfg["text"].get("min_cyrillic_ratio", 0.55))

    dedup: set[int] = set()
    records: list[dict] = []
    per_source_stats: dict[str, dict[str, int]] = {}

    for source_cfg in cfg["sources"]:
        source_name = source_cfg["name"]
        label_name = source_cfg["label"]
        limit = int(source_cfg["limit"])
        kept = 0
        seen_rows = 0
        skipped_short = 0
        skipped_non_bg = 0
        skipped_dup = 0

        for row in _iter_rows(source_cfg):
            row = dict(row)
            seen_rows += 1
            if not row_matches(row, source_cfg):
                continue
            text = make_text(row, source_cfg)
            if not text:
                continue
            windows = split_paragraph_windows(
                text,
                min_chars=min_chars,
                max_chars=max_chars,
                target_chars=target_chars,
            )
            if not windows:
                skipped_short += 1
                continue
            for window_idx, window in enumerate(windows):
                norm = normalize_text(window)
                if len(norm) < min_chars:
                    skipped_short += 1
                    continue
                if cyrillic_ratio(norm) < min_cyr:
                    skipped_non_bg += 1
                    continue
                fp = stable_hash(norm[:512])
                if fp in dedup:
                    skipped_dup += 1
                    continue
                dedup.add(fp)
                records.append(
                    {
                        "id": f"{source_name}-{seen_rows}-{window_idx}",
                        "label_name": label_name,
                        "label_id": label_to_id[label_name],
                        "source_name": source_name,
                        "dataset": source_cfg.get("dataset") or source_cfg.get("endpoint") or source_name,
                        "text": norm,
                    }
                )
                kept += 1
                if kept >= limit:
                    break
            if kept >= limit:
                break

        per_source_stats[source_name] = {
            "label": label_name,
            "seen_rows": seen_rows,
            "kept": kept,
            "skipped_short": skipped_short,
            "skipped_non_bg": skipped_non_bg,
            "skipped_dup": skipped_dup,
        }
        print(f"[benchmark] {source_name}: kept={kept} seen_rows={seen_rows}")

    splits = stratified_split(records, seed=int(benchmark_cfg["seed"]), split_fracs=benchmark_cfg["splits"])
    write_jsonl(output_root / "all.jsonl", records)
    for split_name, rows in splits.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    stats = {
        "labels": label_to_id,
        "total_records": len(records),
        "all_counts": summarize_labels(records),
        "split_counts": {name: summarize_labels(rows) for name, rows in splits.items()},
        "sources": per_source_stats,
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[benchmark] wrote {len(records)} records to {output_root}")


if __name__ == "__main__":
    main()
