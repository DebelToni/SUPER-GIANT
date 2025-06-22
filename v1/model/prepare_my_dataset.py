# DATASET_NAME = "OpenWebText"
# DATASET_NAME   = "roneneldan/TinyStories"

"""
prepare_dataset.py – streaming packer with **chunk_pct** alias
=============================================================
This version packs a line‑based corpus (e.g. WikiText‑103) into fixed‑
length token windows while:

* preserving virtually every token – short paragraphs are concatenated
  instead of padded out or dropped,
* writing predictable, constant‑size shards (10 000 windows each), and
* guaranteeing <5 % pad tokens in the final arrays.

**New / restored CLI flags**
────────────────────────────
* ``--subset_pct`` *(float)* – percent of the corpus to read (default 100).
* ``--chunk_pct`` *(float)* – **alias** of `subset_pct` kept for backward
  compatibility with older `Run_training.py` files. If both are given we
  favour `chunk_pct` and print a warning.

So the old call
```python
get_data(subset_pct=Config.dataset_percent,
         chunk_pct =Config.chunk_percent,
         context_length=Config.context_length)
```
still works.
"""
import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

# ────────────────────────────────
# Configuration constants
# ────────────────────────────────
# DATASET_VENDOR = "Salesforce/wikitext"
# DATASET_NAME   = "wikitext-2-raw-v1"
DATASET_VENDOR = Config.dataset_vendor
DATASET_NAME   = Config.dataset_name

# TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
TOKENIZER_NAME = Config.tokenizer_name
# CACHE_DIR      = Path("tiny_cached")
CACHE_DIR      = Path(Config.tokenizer_path)

VAL_EVERY_N_WIN = 33          # deterministic interleaving train/val
STRIDE_FRAC     = 0.5         # 50 % overlap between successive windows
SHARD_ROWS      = 10_000      # windows per compressed shard
DTYPE           = np.uint16   # enough for vocab < 65 535
PAD_FRAC_LIMIT  = 0.05        # sanity‑check threshold


# ────────────────────────────────
# Load our pre-baked math dataset from Arrow
# ────────────────────────────────
from datasets import load_from_disk
DATA_PATH = Path(Config.dataset_path or "math_dataset_arrow")
print(f"▶ Loading math dataset from {DATA_PATH} …")
ds_main = load_from_disk(str(DATA_PATH))
if Config.use_custom_tokenizer:
    _tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Config.custom_tokenizer_path
    )
else:
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if _tokenizer.pad_token is None:
    _tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
PAD_TOKEN_ID = _tokenizer.pad_token_id


# ────────────────────────────────
# Rolling‑buffer windowiser
# ────────────────────────────────

def _iter_windows(ds, ctx: int, stride: int):
    """Yield fixed‑length windows with overlap from a line corpus."""
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
# Encoding & sharding
# ────────────────────────────────

def _encode_stream(ctx: int, subset_pct: float) -> Tuple[List[Path], List[Path]]:
    global ds_main
    ds = ds_main
    # if Config.dataset_has_vendor:
    #     ds = load_dataset(DATASET_VENDOR, DATASET_NAME, split="train")
    # else:
    #     ds = load_dataset(DATASET_NAME, split="train")
    stride = max(1, int(ctx * STRIDE_FRAC))

    # optional subset
    if subset_pct < 100.0:
        subset_size = math.ceil(len(ds) * subset_pct / 100)
        ds = ds.select(range(subset_size))
        print(f"▶ Using {subset_pct:.1f}% → {subset_size:,} lines from dataset")

    def _new_mm(prefix: str, shard_idx: int):
        path = CACHE_DIR / f"{prefix}_{shard_idx:03d}.npy"
        mm   = np.memmap(path, dtype=DTYPE, mode="w+", shape=(SHARD_ROWS, ctx))
        return path, mm

    train_files, val_files = [], []
    shard = 0
    train_path, train_mm = _new_mm("train_tokens", shard)
    val_path,   val_mm   = _new_mm("val_tokens",   shard)
    train_pos = val_pos = 0
    win_cnt = 0

    pbar = tqdm( desc="packing", unit="window")
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

        # roll over to next shard when either split is full
        if train_pos >= SHARD_ROWS or val_pos >= SHARD_ROWS:
            train_mm.flush(); val_mm.flush()
            for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                if rows:
                    tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(SHARD_ROWS, ctx))[:rows]
                    np.savez_compressed(path.with_suffix(".npz"), data=tmp)
                    os.remove(path)
            train_files.append(train_path.with_suffix(".npz"))
            val_files.append(val_path.with_suffix(".npz"))
            shard += 1
            train_path, train_mm = _new_mm("train_tokens", shard)
            val_path,   val_mm   = _new_mm("val_tokens",   shard)
            train_pos = val_pos = 0
        # pbar.update(stride)  # update tqdm by the number of tokens processed
        pbar.update(1)
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
# Concatenate shards and sanity check
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
        raise RuntimeError(f"Dataset contains {pad_frac:.2%} pad tokens – preprocessing error.")


# ────────────────────────────────
# Public API
# ────────────────────────────────

def get_data(*, subset_pct: float | None = None, chunk_pct: float | None = None,
             context_length: int = 256):
    """Return `(train_tokens, val_tokens, tokenizer)`.

    Parameters
    ----------
    subset_pct : float | None
        Percent of dataset lines to read. Ignored if ``chunk_pct`` is
        given.
    chunk_pct  : float | None
        *Alias* of ``subset_pct`` kept for backward compatibility. If both
        are provided the alias wins.
    context_length : int
        Token window length (``ctx`` in the code).
    """
    # resolve aliasing
    if chunk_pct is not None:
        if subset_pct is not None and subset_pct != 100.0:
            print("⚠ Both subset_pct and chunk_pct provided – using chunk_pct")
        subset_pct = chunk_pct
    if subset_pct is None:
        subset_pct = 100.0

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
    cli = argparse.ArgumentParser("prepare_dataset packing util")
    cli.add_argument("--subset_pct", type=float, default=100,
                     help="Percent of corpus to encode [0‑100]")
    cli.add_argument("--chunk_pct" , type=float,
                     help="Alias of --subset_pct (older scripts)")
    cli.add_argument("--ctx", type=int, default=256,
                     help="Token window length")
    args = cli.parse_args()

    tr, va, _ = get_data(subset_pct=args.subset_pct,
                         chunk_pct=args.chunk_pct,
                         context_length=args.ctx)
    print("train_tokens", tr.shape, "val_tokens", va.shape)

