import os
import math
import re
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent / "Config.yml"
Config = OmegaConf.load(CONFIG_PATH)

# Constants
DATASET_VENDOR = Config.dataset_vendor
DATASET_NAME   = Config.dataset_name
TOKENIZER_NAME = Config.tokenizer_name
CACHE_DIR      = Path(Config.tokenizer_path)

VAL_EVERY_N_WIN = 33            # interleave train/val every N windows
SHARD_ROWS      = 10_000        # windows per compressed shard
DTYPE           = np.uint16     # token dtype
# PAD_FRAC_LIMIT  = 0.05          # max allowable pad fraction
PAD_FRAC_LIMIT = 25

# Load or build tokenizer
print("▶ Loading tokenizer …")
if Config.use_custom_tokenizer:
    _tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Config.custom_tokenizer_path
    )
else:
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# Ensure special tokens exist
if _tokenizer.pad_token is None:
    _tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
PAD_TOKEN_ID = _tokenizer.pad_token_id

# Utility: clean raw text of noisy symbols/sequences

def clean_text(text: str) -> str:
    # remove wiki links [[...]], HTML tags, URLs, non-ASCII/control chars
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]+", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Utility: split a record into paragraph strings

def iter_paragraphs(ds) -> Iterator[str]:
    buffer: List[str] = []
    for rec in ds:
        raw = rec.get("text", "")
        # if dataset has embedded paragraphs, split on double-newline
        parts = raw.split("\n\n") if "\n\n" in raw else [raw]
        for p in parts:
            p = p.strip()
            if p:
                yield p

# Generate fixed-length windows at paragraph starts

def _iter_windows(ds, ctx: int) -> Iterator[List[int]]:
    for paragraph in iter_paragraphs(ds):
        # clean noise
        cleaned = clean_text(paragraph)
        if not cleaned:
            continue
        # tokenize
        ids = _tokenizer.encode(cleaned, add_special_tokens=False)
        if not ids:
            continue
        # drop very short paragraphs to limit padding
        if len(ids) < ctx // 2:
            continue
        # take first ctx tokens, pad if needed
        if len(ids) >= ctx:
            window = ids[:ctx]
        else:
            window = ids + [PAD_TOKEN_ID] * (ctx - len(ids))
        yield window

# Shard & write encoded windows

def _encode_stream(ctx: int, subset_pct: float) -> Tuple[List[Path], List[Path]]:
    # load dataset
    if Config.use_custom_dataset:
        ds = load_from_disk(str(Config.dataset_path))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
    else:
        ds = (
            load_dataset(DATASET_VENDOR, DATASET_NAME, split="train")
            if Config.dataset_has_vendor
            else load_dataset(DATASET_NAME, split="train")
        )

    # optionally subset lines/records
    if subset_pct < 100.0:
        size = math.ceil(len(ds) * subset_pct / 100)
        ds = ds.select(range(size))
        print(f"▶ Using {subset_pct:.1f}% → {size:,} records from dataset")

    def _new_mm(prefix: str, shard_idx: int):
        path = CACHE_DIR / f"{prefix}_{shard_idx:03d}.npy"
        mm   = np.memmap(path, dtype=DTYPE, mode="w+", shape=(SHARD_ROWS, ctx))
        return path, mm

    train_files, val_files = [], []
    shard = 0
    train_path, train_mm = _new_mm("train_tokens", shard)
    val_path,   val_mm   = _new_mm("val_tokens",   shard)
    train_pos = val_pos = win_cnt = 0

    pbar = tqdm(desc="packing", unit="window")
    for window in _iter_windows(ds, ctx):
        # choose split
        if win_cnt % VAL_EVERY_N_WIN == 0:
            target_mm, target_pos = val_mm, val_pos
        else:
            target_mm, target_pos = train_mm, train_pos
        target_mm[target_pos] = window
        if target_mm is train_mm:
            train_pos += 1
        else:
            val_pos += 1
        win_cnt += 1

        # shard rollover
        if train_pos >= SHARD_ROWS or val_pos >= SHARD_ROWS:
            for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                if rows:
                    data = np.memmap(path, dtype=DTYPE, mode="r", shape=(SHARD_ROWS, ctx))[:rows]
                    np.savez_compressed(path.with_suffix(".npz"), data=data)
                    os.remove(path)
            train_files.append(train_path.with_suffix(".npz"))
            val_files.append(val_path.with_suffix(".npz"))
            shard += 1
            train_path, train_mm = _new_mm("train_tokens", shard)
            val_path,   val_mm   = _new_mm("val_tokens",   shard)
            train_pos = val_pos = 0
        pbar.update(1)
    pbar.close()

    # final flush
    for path, mm, pos, out_list in [
        (train_path, train_mm, train_pos, train_files),
        (val_path,   val_mm,   val_pos,   val_files)
    ]:
        if pos:
            data = np.memmap(path, dtype=DTYPE, mode="r", shape=(SHARD_ROWS, ctx))[:pos]
            np.savez_compressed(path.with_suffix(".npz"), data=data)
            os.remove(path)
            out_list.append(path.with_suffix(".npz"))

    return train_files, val_files

# Concatenate shards & check padding

def _concat(shards):
    arrays = [np.load(p, mmap_mode="r")["data"] for p in shards]
    return np.concatenate(arrays, axis=0)


def _sanity_check(arr: np.ndarray):
    pad_frac = (arr == PAD_TOKEN_ID).mean()
    if pad_frac > PAD_FRAC_LIMIT:
        raise RuntimeError(f"Dataset contains {pad_frac:.2%} pad tokens – preprocessing error.")

# Public API

def get_data(*, subset_pct: float | None = None, chunk_pct: float | None = None,
             context_length: int = 256):
    if chunk_pct is not None:
        if subset_pct is not None and subset_pct != 100.0:
            print("⚠ Both subset_pct and chunk_pct provided – using chunk_pct")
        subset_pct = chunk_pct
    subset_pct = 100.0 if subset_pct is None else subset_pct

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

# CLI
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
