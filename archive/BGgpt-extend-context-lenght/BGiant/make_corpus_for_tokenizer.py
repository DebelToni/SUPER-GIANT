#!/usr/bin/env python3
"""
Build a mixed text corpus from several Hugging Face datasets.

Datasets:
- EN: agentlans/high-quality-english-sentences (split: train, field: text)
- BG: climb-mao/Bulgarian-BabyLM (split: train, field: text)
- PY: Muennighoff/mbpp (subset: sanitized, split: test, field: code)

Usage (example):
  python make_corpus.py \
      --size-mb 200 \
      --output mixed_corpus.txt \
      --seed 42
"""

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from datasets import load_dataset
from huggingface_hub import login


# -------------------------
# Config for each domain
# -------------------------

@dataclass
class DomainConfig:
    name: str
    dataset_id: str
    subset: Optional[str]  # e.g. "sanitized" for mbpp, None if default
    split: str             # e.g. "train" or "test"
    text_key: str          # column with the text/code
    weight: float          # target share of bytes (0–1)


DOMAIN_CONFIGS: List[DomainConfig] = [
    DomainConfig(
        name="en",
        dataset_id="agentlans/high-quality-english-sentences",
        subset=None,            # only "default" subset; None works fine
        split="train",
        text_key="text",
        weight=0.45,
    ),
    DomainConfig(
        name="bg",
        dataset_id="climb-mao/Bulgarian-BabyLM",
        subset=None,
        split="train",
        text_key="text",
        weight=0.45,
    ),
    DomainConfig(
        name="py",
        dataset_id="google-research-datasets/mbpp",  # was "Muennighoff/mbpp"
        subset="sanitized",                          # keep sanitized subset
        split="train",                               # use train split here
        text_key="code",
        weight=0.10,
    ),
]


# -------------------------
# Domain sampler
# -------------------------

class DomainSampler:
    """
    Wraps a Hugging Face dataset and yields cleaned text samples.
    Keeps track of how many bytes have been written from this domain.
    """

    def __init__(self, config: DomainConfig):
        self.config = config
        self.dataset = self._load_dataset()
        self.iterator = iter(self.dataset)
        self.bytes_written: int = 0

    def _load_dataset(self):
        """Load the HF dataset (non-streaming, memory-mapped to disk)."""
        if self.config.subset is not None:
            ds = load_dataset(
                self.config.dataset_id,
                self.config.subset,
                split=self.config.split,
            )
        else:
            ds = load_dataset(
                self.config.dataset_id,
                split=self.config.split,
            )
        return ds

    def _raw_next_row(self) -> dict:
        """Get the next row, re-instantiating the iterator when exhausted."""
        while True:
            try:
                row = next(self.iterator)
                return row
            except StopIteration:
                # Start over from the beginning
                self.dataset = self._load_dataset()
                self.iterator = iter(self.dataset)

    def next_sample(self) -> str:
        """
        Return a single non-empty text sample (string) from the dataset.
        Performs very light cleaning only.
        """
        while True:
            row = self._raw_next_row()
            val = row.get(self.config.text_key, "")

            if not isinstance(val, str):
                continue

            text = val.strip()
            if not text:
                continue

            # Normalise newlines slightly
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return text


# -------------------------
# HF login helper
# -------------------------

def maybe_login_from_env():
    """
    Optionally login to Hugging Face Hub if HF_TOKEN is set in env.
    This lets you access any gated datasets, if needed later.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        return

    try:
        print("Detected HF_TOKEN in environment, logging into Hugging Face Hub...")
        login(token=token)
    except Exception as exc:
        print(f"Warning: could not login to Hugging Face Hub: {exc}")


# -------------------------
# Corpus builder
# -------------------------

def build_corpus(
    output_path: Path,
    target_mb: float,
    seed: int = 42,
) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes: int = int(target_mb * 1024 * 1024)
    print(f"Target size: ~{target_mb:.2f} MB ({target_bytes} bytes)")

    # Instantiate samplers
    samplers: List[DomainSampler] = [DomainSampler(cfg) for cfg in DOMAIN_CONFIGS]

    # Compute per-domain byte targets (by weight)
    total_weight = sum(cfg.weight for cfg in DOMAIN_CONFIGS)
    norm_weights: Dict[str, float] = {
        cfg.name: (cfg.weight / total_weight) for cfg in DOMAIN_CONFIGS
    }
    domain_targets: Dict[str, int] = {
        cfg.name: int(target_bytes * norm_weights[cfg.name]) for cfg in DOMAIN_CONFIGS
    }

    print("Per-domain target bytes (approx):")
    for cfg in DOMAIN_CONFIGS:
        mb = domain_targets[cfg.name] / (1024 * 1024)
        print(f"  - {cfg.name}: {mb:.2f} MB (weight={cfg.weight:.2f})")

    total_bytes_written = 0
    progress_step = 10 * 1024 * 1024  # print every ~10 MB
    next_progress_threshold = progress_step

    with output_path.open("w", encoding="utf-8") as f:
        while total_bytes_written < target_bytes:
            # Domains that haven't hit their share yet
            underfilled = [
                s for s in samplers
                if s.bytes_written < domain_targets[s.config.name]
            ]

            # Once all hit their target, we just use all of them
            candidate_samplers = underfilled or samplers

            # Selection weights based on original weights
            weights = [s.config.weight for s in candidate_samplers]
            chosen: DomainSampler = random.choices(candidate_samplers, weights=weights, k=1)[0]

            sample = chosen.next_sample()
            # Separate samples with a blank line
            text_to_write = sample.strip() + "\n\n"

            encoded = text_to_write.encode("utf-8")
            n_bytes = len(encoded)

            # Write and update counters
            f.write(text_to_write)
            total_bytes_written += n_bytes
            chosen.bytes_written += n_bytes

            # Progress output
            if total_bytes_written >= next_progress_threshold:
                mb_done = total_bytes_written / (1024 * 1024)
                print(f"Wrote {mb_done:.1f} MB...", end="\r", flush=True)
                next_progress_threshold += progress_step

    print()  # newline after progress line
    final_mb = total_bytes_written / (1024 * 1024)
    print(f"Done. Wrote {final_mb:.2f} MB to '{output_path}'")

    print("Final per-domain shares (by bytes):")
    for s in samplers:
        pct = (s.bytes_written / total_bytes_written) * 100 if total_bytes_written > 0 else 0.0
        mb = s.bytes_written / (1024 * 1024)
        print(f"  - {s.config.name}: {mb:.2f} MB ({pct:.1f}%)")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mixed EN/BG/Python corpus from Hugging Face datasets."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mixed_corpus.txt",
        help="Output file path (UTF-8 text).",
    )
    parser.add_argument(
        "--size-mb",
        "-s",
        type=float,
        default=200.0,
        help="Approximate target size in megabytes (100–400 recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_login_from_env()
    output_path = Path(args.output)
    build_corpus(output_path=output_path, target_mb=args.size_mb, seed=args.seed)


if __name__ == "__main__":
    main()

