import os, random, math, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ── tweakables ────────────────────────────────────────────────────────────────
DATASET_NAME   = "roneneldan/TinyStories"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
CACHE_DIR      = Path("tiny_cached")          # kept on the Colab VM disk
VAL_SPLIT_PCT  = 3.0                          # 97 / 3 train-val split
DTYPE          = np.uint16                    # 65 535 > 50 000-token vocab


# ── helpers ───────────────────────────────────────────────────────────────────
def _flush(buf: List[List[int]], path: Path):
    if not buf:
        return
    np.save(path, np.asarray(buf, dtype=DTYPE))
    buf.clear()


def _encode_stream(tokenizer, ctx: int,
                   subset_pct: float, chunk_pct: float) -> Tuple[List[Path], List[Path]]:
    ds              = load_dataset(DATASET_NAME, split="train")       # Arrow table, lazy
    total_examples  = len(ds)
    limit_examples  = math.ceil(total_examples * subset_pct / 100)
    per_shard_input = max(1, math.floor(limit_examples * chunk_pct / 100))

    train_files, val_files          = [], []
    train_buf: List[List[int]] = []
    val_buf:   List[List[int]] = []
    shard = 0

    pbar = tqdm(total=limit_examples, desc="tokenising", unit="example")
    for i, rec in enumerate(ds):
        if i >= limit_examples:
            break
        pbar.update()

        ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        for s in range(0, len(ids) - ctx + 1, ctx):
            window = ids[s:s + ctx]
            (val_buf if random.random() < VAL_SPLIT_PCT / 100 else train_buf).append(window)

        if (i + 1) % per_shard_input == 0:
            _flush(train_buf, CACHE_DIR / f"train_tokens_{shard:03d}.npy")
            _flush(val_buf,   CACHE_DIR / f"val_tokens_{shard:03d}.npy")
            train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
            val_files.append(CACHE_DIR / f"val_tokens_{shard:03d}.npy")
            shard += 1

    _flush(train_buf, CACHE_DIR / f"train_tokens_{shard:03d}.npy")
    _flush(val_buf,   CACHE_DIR / f"val_tokens_{shard:03d}.npy")
    if train_buf:
        train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
    if val_buf:
        val_files.append(CACHE_DIR / f"val_tokens_{shard:03d}.npy")
    pbar.close()
    return train_files, val_files


def _concat(shards):
    return np.concatenate([np.load(p, mmap_mode="r") for p in shards], axis=0)


def get_data(*, subset_pct: float = 100.0, chunk_pct: float = 10.0,
             context_length: int = 256):
    """Main entry-point used from Run_training.py"""
    CACHE_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train_shards = sorted(CACHE_DIR.glob("train_tokens_*.npy"))
    val_shards   = sorted(CACHE_DIR.glob("val_tokens_*.npy"))
    if train_shards:
        print("▶ Using cached shards found in", CACHE_DIR)
        return _concat(train_shards), _concat(val_shards), tokenizer

    print("▶ No cache found – streaming encode begins…")
    train_shards, val_shards = _encode_stream(
        tokenizer, context_length, subset_pct, chunk_pct)
    return _concat(train_shards), _concat(val_shards), tokenizer


# ── minimal CLI for experimentation ───────────────────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--subset_pct", type=float, default=100,
                     help="Percent of TinyStories to encode [0–100]")
    cli.add_argument("--chunk_pct", type=float, default=10,
                     help="Percent interval between output shards")
    cli.add_argument("--ctx", type=int, default=256,
                     help="Sequence length (context window)")
    a = cli.parse_args()
    tr, va, _ = get_data(subset_pct=a.subset_pct,
                         chunk_pct=a.chunk_pct,
                         context_length=a.ctx)
    print("train_tokens", tr.shape, "val_tokens", va.shape)

