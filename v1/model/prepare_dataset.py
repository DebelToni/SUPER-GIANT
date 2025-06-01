# DATASET_NAME = "OpenWebText"
# DATASET_NAME   = "roneneldan/TinyStories"

"""
prepare_dataset.py

Streaming preprocessing utility for language‑model training.

Key fixes vs. original version
──────────────────────────────
* Drops empty or whitespace‑only WikiText records early.
* Skips paragraphs shorter than the context window – no more
  filling an entire window with pad tokens.
* Adds a real pad token to the tokenizer if it does not exist
  and propagates its id everywhere.
* _chunk(...) no longer pads; it only returns full‑length windows.
* Sanity check warns if the final dataset still contains too many pads.
"""
import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ────────────────────────────────
# Configuration
# ────────────────────────────────
DATASET_NAME   = "wikitext-103-v1"
DATASET_VENDOR = "wikitext"

TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
CACHE_DIR      = Path("tiny_cached")

VAL_SPLIT_PCT  = 3.0              # percentage of windows that go to validation
DTYPE          = np.uint16        # np.uint16 is enough for vocab < 65 535

STRIDE_FRAC    = 0.5              # overlap expressed as fraction of ctx
VAL_EVERY_N_WIN = 33              # deterministic interleaving train/val
FLUSH_EVERY     = 4_096

MIN_DOC_TOKENS  = 64             # drop docs shorter than this many tokens


# ────────────────────────────────
# Tokeniser & PAD token handling
# ────────────────────────────────
print("▶ Loading tokenizer …")

# we load once at import time; it is cheap relative to the rest
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if _tokenizer.pad_token is None:
    _tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

PAD_TOKEN_ID = _tokenizer.pad_token_id


# ────────────────────────────────
# Helper functions
# ────────────────────────────────

def _chunk(doc_ids: List[int], ctx: int, stride: int) -> List[List[int]]:
    """Slice *one* tokenised document into ≥1 windows of exactly ctx tokens.

    • Docs shorter than ctx are *skipped* (no padding).
    • Discards the final partial window of a long doc.
    """
    n = len(doc_ids)
    if n < ctx:
        return []

    windows = []
    start = 0
    while start + ctx <= n:
        windows.append(doc_ids[start:start + ctx])
        start += stride
    return windows


def _encode_stream(
    ctx: int,
    subset_pct: float,
    chunk_pct: float,
) -> Tuple[List[Path], List[Path]]:
    """Tokenise, window, and shard the dataset on the fly."""

    ds = load_dataset(DATASET_VENDOR, DATASET_NAME, split="train")
    stride = max(1, int(ctx * STRIDE_FRAC))

    total_examples  = len(ds)
    limit_examples  = math.ceil(total_examples * subset_pct / 100)
    per_shard_input = max(1, math.floor(limit_examples * chunk_pct / 100))

    shard, win_cnt = 0, 0
    train_files, val_files = [], []

    def _new_memmap(prefix: str):
        path = CACHE_DIR / f"{prefix}_{shard:03d}.npy"
        mm   = np.memmap(path, dtype=DTYPE, mode="w+",
                         shape=(per_shard_input * 10, ctx))
        return path, mm

    train_path, train_mm = _new_memmap("train_tokens")
    val_path,   val_mm   = _new_memmap("val_tokens")
    train_pos = val_pos = 0

    pbar = tqdm(total=limit_examples, desc="tokenising", unit="doc")
    for i, rec in enumerate(ds):
        if i >= limit_examples:
            break

        text = rec["text"]
        if not text or not text.strip():
            continue  # drop empty lines

        ids = _tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < MIN_DOC_TOKENS:
            continue  # skip micro‑docs to avoid pad‑heavy windows

        windows = _chunk(ids, ctx, stride)
        if not windows:
            continue

        pbar.update()
        for w in windows:
            target_mm, target_pos = (
                (val_mm, val_pos) if (win_cnt % VAL_EVERY_N_WIN == 0)
                else (train_mm, train_pos)
            )
            target_mm[target_pos] = w
            if target_mm is train_mm:
                train_pos += 1
            else:
                val_pos += 1
            win_cnt += 1

            # flush periodically to keep mem usage bounded
            if win_cnt % FLUSH_EVERY == 0:
                train_mm.flush(); val_mm.flush()

            # roll over to a new shard when either split fills up
            if (train_pos >= per_shard_input) or (val_pos >= per_shard_input):
                train_mm.flush(); val_mm.flush()

                for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                    if rows:
                        tmp = np.memmap(path, dtype=DTYPE, mode="r",
                                        shape=(per_shard_input * 10, ctx))[:rows]
                        np.savez_compressed(path.with_suffix(".npz"), data=tmp)
                        os.remove(path)

                train_files.append(train_path.with_suffix(".npz"))
                val_files.append(val_path.with_suffix(".npz"))

                shard += 1
                train_path, train_mm = _new_memmap("train_tokens")
                val_path,   val_mm   = _new_memmap("val_tokens")
                train_pos = val_pos = 0

    # final shard
    train_mm.flush(); val_mm.flush()
    for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
        if rows:
            tmp = np.memmap(path, dtype=DTYPE, mode="r",
                            shape=(per_shard_input * 10, ctx))[:rows]
            np.savez_compressed(path.with_suffix(".npz"), data=tmp)
            os.remove(path)
    if train_pos:
        train_files.append(train_path.with_suffix(".npz"))
    if val_pos:
        val_files.append(val_path.with_suffix(".npz"))
    pbar.close()
    return train_files, val_files


def _concat(shards):
    arrays = []
    for p in shards:
        if p.suffix == ".npz":
            with np.load(p, mmap_mode="r") as z:
                arrays.append(z["data"])
        else:
            arrays.append(np.load(p, mmap_mode="r"))
    return np.concatenate(arrays, axis=0)


def _sanity_check(arr: np.ndarray):
    pad_frac = (arr == PAD_TOKEN_ID).mean()
    if pad_frac > 0.05:
        raise RuntimeError(
            f"Dataset contains {pad_frac:.2%} pad tokens – "
            f"check preprocessing parameters.")


# ────────────────────────────────
# Public API
# ────────────────────────────────

def get_data(*, subset_pct: float = 100.0, chunk_pct: float = 10.0,
             context_length: int = 256):
    """Main entry point used by Run_training.py"""
    CACHE_DIR.mkdir(exist_ok=True)

    train_shards = sorted(CACHE_DIR.glob("train_tokens_*.npz"))
    val_shards   = sorted(CACHE_DIR.glob("val_tokens_*.npz"))
    if train_shards and val_shards:
        print("▶ Using cached shards found in", CACHE_DIR)
        train_tokens = _concat(train_shards)
        val_tokens   = _concat(val_shards)
        _sanity_check(train_tokens)
        return train_tokens, val_tokens, _tokenizer

    print("▶ No cache found – streaming encode begins…")
    train_shards, val_shards = _encode_stream(
        context_length, subset_pct, chunk_pct)
    train_tokens = _concat(train_shards)
    val_tokens   = _concat(val_shards)
    _sanity_check(train_tokens)
    return train_tokens, val_tokens, _tokenizer


# ────────────────────────────────
# CLI usage
# ────────────────────────────────
if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Prepare WikiText‑103 for GPT training.")
    cli.add_argument("--subset_pct", type=float, default=100,
                     help="Percent of WikiText‑103 to encode [0–100]")
    cli.add_argument("--chunk_pct", type=float, default=10,
                     help="Percent interval between output shards")
    cli.add_argument("--ctx", type=int, default=256,
                     help="Sequence length (context window)")
    args = cli.parse_args()

    tr, va, _ = get_data(subset_pct=args.subset_pct,
                         chunk_pct=args.chunk_pct,
                         context_length=args.ctx)
    print("train_tokens", tr.shape, "val_tokens", va.shape)

