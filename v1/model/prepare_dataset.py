# DATASET_NAME = "OpenWebText"
# DATASET_NAME   = "roneneldan/TinyStories"

"""
prepare_dataset.py – **stream‑packing** variant
============================================
A robust data‑prep helper that turns WikiText‑103 (or any other text‑line
corpus) into fixed‑length windows without drowning everything in padding.

Major differences vs. v2
────────────────────────
1. **Streaming packer** – short lines are concatenated into one rolling
   buffer so we never throw away usable tokens. Windows are emitted with
   a fixed *stride* (default 50 % overlap).
2. **Constant‑size shards** – each shard holds `SHARD_ROWS` windows,
   making memory use predictable.
3. **Simpler bookkeeping** – no more mixing “examples” (docs) with
   “windows” when sizing memmaps.
4. **Pad token sanity check** retained.

Usage stays the same for `Run_training.py`.
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
DATASET_VENDOR = "Salesforce/wikitext"
DATASET_NAME   = "wikitext-2-raw-v1"

TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
CACHE_DIR      = Path("tiny_cached")

VAL_EVERY_N_WIN = 33          # deterministic interleaving train/val
STRIDE_FRAC     = 0.5         # 50 % overlap between windows
SHARD_ROWS      = 10_000      # windows per .npz file
DTYPE           = np.uint16   # works for vocab < 65 535
PAD_FRAC_LIMIT  = 0.05        # sanity‑check threshold


# ────────────────────────────────
# Tokeniser & pad token handling
# ────────────────────────────────
print("▶ Loading tokenizer …")
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if _tokenizer.pad_token is None:
    _tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
PAD_TOKEN_ID = _tokenizer.pad_token_id


# ────────────────────────────────
# Helper – rolling buffer windowiser
# ────────────────────────────────

def _iter_windows(ds, ctx: int, stride: int):
    """Yield fixed‑length windows from a line corpus.

    Concatenates successive non‑empty lines into a buffer. Whenever the
    buffer has `ctx` tokens or more, we pop the first `ctx` tokens as a
    window and advance the buffer by `stride` tokens.
    """
    buf: List[int] = []
    for rec in ds:
        text = rec["text"].strip()
        if not text:
            continue
        buf.extend(_tokenizer.encode(text, add_special_tokens=False))

        while len(buf) >= ctx:
            yield buf[:ctx]
            del buf[:stride]


# ────────────────────────────────
# Core encoding/sharding routine
# ────────────────────────────────

def _encode_stream(ctx: int, subset_pct: float) -> Tuple[List[Path], List[Path]]:
    ds = load_dataset(DATASET_VENDOR, DATASET_NAME, split="train")
    stride = max(1, int(ctx * STRIDE_FRAC))

    # Restrict dataset size if requested
    if subset_pct < 100.0:
        subset_size = math.ceil(len(ds) * subset_pct / 100)
        ds = ds.select(range(subset_size))

    train_files, val_files = [], []

    def _new_memmap(prefix: str, shard_idx: int):
        path = CACHE_DIR / f"{prefix}_{shard_idx:03d}.npy"
        mm   = np.memmap(path, dtype=DTYPE, mode="w+", shape=(SHARD_ROWS, ctx))
        return path, mm

    shard = 0
    train_path, train_mm = _new_memmap("train_tokens", shard)
    val_path,   val_mm   = _new_memmap("val_tokens",   shard)
    train_pos = val_pos = 0

    pbar = tqdm(total=len(ds), desc="packing", unit="line")
    win_cnt = 0
    for window in _iter_windows(ds, ctx, stride):
        target_mm, target_pos = (
            (val_mm, val_pos) if (win_cnt % VAL_EVERY_N_WIN == 0)
            else (train_mm, train_pos)
        )
        target_mm[target_pos] = window

        if target_mm is train_mm:
            train_pos += 1
        else:
            val_pos += 1
        win_cnt += 1

        # rollover when shard full
        if train_pos >= SHARD_ROWS or val_pos >= SHARD_ROWS:
            train_mm.flush(); val_mm.flush()

            # compress and close shard
            for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                if rows:
                    tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(SHARD_ROWS, ctx))[:rows]
                    np.savez_compressed(path.with_suffix(".npz"), data=tmp)
                    os.remove(path)

            train_files.append(train_path.with_suffix(".npz"))
            val_files.append(val_path.with_suffix(".npz"))

            shard += 1
            train_path, train_mm = _new_memmap("train_tokens", shard)
            val_path,   val_mm   = _new_memmap("val_tokens",   shard)
            train_pos = val_pos = 0
        pbar.update(0)  # keep tqdm alive (we’re iterating our own gen)

    # final flush
    train_mm.flush(); val_mm.flush()
    for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
        if rows:
            tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(SHARD_ROWS, ctx))[:rows]
            np.savez_compressed(path.with_suffix(".npz"), data=tmp)
            os.remove(path)
    if train_pos:
        train_files.append(train_path.with_suffix(".npz"))
    if val_pos:
        val_files.append(val_path.with_suffix(".npz"))
    pbar.close()
    return train_files, val_files


# ────────────────────────────────
# Utilities
# ────────────────────────────────

def _concat(shards):
    arrays = []
    for p in shards:
        with np.load(p, mmap_mode="r") as z:
            arrays.append(z["data"])
    return np.concatenate(arrays, axis=0)


def _sanity_check(arr: np.ndarray):
    pad_frac = (arr == PAD_TOKEN_ID).mean()
    if pad_frac > PAD_FRAC_LIMIT:
        raise RuntimeError(
            f"Dataset contains {pad_frac:.2%} pad tokens – preprocessing error.")


# ────────────────────────────────
# Public API
# ────────────────────────────────

def get_data(*, subset_pct: float = 100.0, context_length: int = 256):
    """Return (train_tokens, val_tokens, tokenizer)."""
    CACHE_DIR.mkdir(exist_ok=True)

    train_shards = sorted(CACHE_DIR.glob("train_tokens_*.npz"))
    val_shards   = sorted(CACHE_DIR.glob("val_tokens_*.npz"))
    if train_shards and val_shards:
        print("▶ Using cached shards found in", CACHE_DIR)
        tr = _concat(train_shards); va = _concat(val_shards)
        _sanity_check(tr)
        return tr, va, _tokenizer

    print("▶ No cache found – streaming encode begins…")
    train_shards, val_shards = _encode_stream(context_length, subset_pct)
    tr = _concat(train_shards); va = _concat(val_shards)
    _sanity_check(tr)
    return tr, va, _tokenizer


# ────────────────────────────────
# CLI helper
# ────────────────────────────────
if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser("Prepare dataset")
    cli.add_argument("--subset_pct", type=float, default=100)
    cli.add_argument("--ctx", type=int, default=256)
    args = cli.parse_args()

    tr, va, _ = get_data(subset_pct=args.subset_pct, context_length=args.ctx)
    print("train_tokens", tr.shape, "val_tokens", va.shape)

