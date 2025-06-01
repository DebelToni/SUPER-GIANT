import os, random, math, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ── tweakables ────────────────────────────────────────────────────────────────
# DATASET_NAME   = "TinyStories"
# DATASET_VENDOR = "roneneldan"
DATASET_NAME = "wikitext-103-v1"
DATASET_VENDOR = "wikitext"  
# DATASET_NAME = "OpenWebText"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
CACHE_DIR      = Path("tiny_cached")          # kept on the Colab VM disk
VAL_SPLIT_PCT  = 3.0                          # 97 / 3 train-val split
DTYPE          = np.uint16                    # 65 535 > 50 000-token vocab


# ── constants ────────────────────────────────────────────────────────────────
STRIDE_FRAC     = 0.5        # 0.5 ⇒ 50 % overlap. Set 1.0 for no overlap.
VAL_EVERY_N_WIN = 33         # deterministic ~3 % val split
PAD_TOKEN_ID    = 0          # GPT-Neo uses 50256 (= eod), change if needed
FLUSH_EVERY     = 4_096      # write to disk after this many windows

# ── chunker ──────────────────────────────────────────────────────────────────
def _chunk(document_ids: List[int], ctx: int, stride: int) -> List[List[int]]:
    """ Split one tokenised document into ctx-sized windows with overlap. """
    n = len(document_ids)
    if n <= ctx:
        # Pad on the right so every sequence is exactly ctx long.
        window = document_ids + [PAD_TOKEN_ID] * (ctx - n)
        return [window]

    windows, start = [], 0
    while start < n:
        end = start + ctx
        window = document_ids[start:end]
        if len(window) < ctx:
            # Discard tiny tail; alternatively pad it – up to you.
            break
        windows.append(window)
        start += stride
    return windows



# ── helpers ───────────────────────────────────────────────────────────────────
def _flush(buf: List[List[int]], path: Path):
    if not buf:
        return
    np.save(path, np.asarray(buf, dtype=DTYPE))
    buf.clear()

def _encode_stream(
    tokenizer,
    ctx: int,
    subset_pct: float,
    chunk_pct: float,
) -> Tuple[List[Path], List[Path]]:
    ds     = load_dataset(DATASET_VENDOR, DATASET_NAME, split="train",) # streaming=True)
    stride = max(1, int(ctx * STRIDE_FRAC))

    # Compute quotas
    # total_examples  = ds.info.dataset_size  # works in streaming mode
    total_examples  = len(ds)  
    limit_examples  = math.ceil(total_examples * subset_pct / 100)
    per_shard_input = max(1, math.floor(limit_examples * chunk_pct / 100))

    # Shard bookkeeping
    shard, win_cnt = 0, 0
    train_files, val_files = [], []

    # Open two memmaps for the first shard
    def _new_memmap(prefix: str):
        path = CACHE_DIR / f"{prefix}_{shard:03d}.npy"
        mm   = np.memmap(path, dtype=DTYPE, mode="w+", shape=(per_shard_input * 10, ctx))
        return path, mm

    train_path, train_mm = _new_memmap("train_tokens")
    val_path,   val_mm   = _new_memmap("val_tokens")
    train_pos, val_pos   = 0, 0

    pbar = tqdm(total=limit_examples, desc="tokenising", unit="doc")
    for i, rec in enumerate(ds):
        if i >= limit_examples:
            break
        pbar.update()

        windows = _chunk(
            tokenizer.encode(rec["text"], add_special_tokens=False),
            ctx, stride,
        )

        for w in windows:
            target_mm, target_pos = (val_mm, val_pos) if (win_cnt % VAL_EVERY_N_WIN == 0) else (train_mm, train_pos)
            target_mm[target_pos] = w
            if target_mm is train_mm:
                train_pos += 1
            else:
                val_pos += 1
            win_cnt += 1

            # Flush memmap if necessary
            if win_cnt % FLUSH_EVERY == 0:
                train_mm.flush(); val_mm.flush()

            # Rotate shard when we reach the quota
            if (train_pos >= per_shard_input) or (val_pos >= per_shard_input):
                train_mm.flush(); val_mm.flush()
                # Trim unused rows and compress
                # for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                #     if rows:
                #         tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(per_shard_input * 10, ctx))[:rows]
                #         # np.savez_compressed(path.with_suffix(".npz"), tmp)
                #         np.savez_compressed(path.with_suffix(".npz"), data=tmp)
                #         os.remove(path)  # remove the raw memmap
                # # train_files.append(train_path.with_suffix(".npz"))
                # train_files.append(path.with_suffix(".npz"))
                # val_files.append(val_path.with_suffix(".npz"))
                for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
                    if rows:
                        tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(per_shard_input * 10, ctx))[:rows]
                        np.savez_compressed(path.with_suffix(".npz"), data=tmp)
                        os.remove(path)

                # Append each shard explicitly (do not reuse the loop variable `path`)
                train_files.append(train_path.with_suffix(".npz"))
                val_files.append(val_path.with_suffix(".npz"))

                # Start new shard
                shard += 1
                train_path, train_mm = _new_memmap("train_tokens")
                val_path,   val_mm   = _new_memmap("val_tokens")
                train_pos, val_pos   = 0, 0

    # Final shard cleanup (same logic as above)
    train_mm.flush(); val_mm.flush()
    for path, rows in [(train_path, train_pos), (val_path, val_pos)]:
        if rows:
            tmp = np.memmap(path, dtype=DTYPE, mode="r", shape=(per_shard_input * 10, ctx))[:rows]
            # np.savez_compressed(path.with_suffix(".npz"), tmp)
            np.savez_compressed(path.with_suffix(".npz"), data=tmp)
            os.remove(path)
    if train_pos:
        # train_files.append(train_path.with_suffix(".npz"))
        train_files.append(train_path.with_suffix(".npz"))
    if val_pos:
        val_files.append(val_path.with_suffix(".npz"))
    pbar.close()
    return train_files, val_files


# def _encode_stream(tokenizer, ctx: int,
#                    subset_pct: float, chunk_pct: float) -> Tuple[List[Path], List[Path]]:
#     # ds              = load_dataset(DATASET_NAME, split="train")       # this does not work
#     ds              = load_dataset("wikitext", "wikitext-103-v1", split="train") # this gets us to tokenizing
#     total_examples  = len(ds)
#     limit_examples  = math.ceil(total_examples * subset_pct / 100)
#     per_shard_input = max(1, math.floor(limit_examples * chunk_pct / 100))
#
#     train_files, val_files          = [], []
#     train_buf: List[List[int]] = []
#     val_buf:   List[List[int]] = []
#     shard = 0
#
#     pbar = tqdm(total=limit_examples, desc="tokenising", unit="example")
#     for i, rec in enumerate(ds):
#         if i >= limit_examples:
#             break
#         pbar.update()
#
#         ids = tokenizer.encode(rec["text"], add_special_tokens=False)
#         for s in range(0, len(ids) - ctx + 1, ctx):
#             window = ids[s : s + ctx]
#             # random train/val split per-window:
#             (val_buf if random.random() < VAL_SPLIT_PCT / 100 else train_buf).append(window)
#
#         if (i + 1) % per_shard_input == 0:
#             # === Only flush & append if each buffer has something in it ===
#             train_path = CACHE_DIR / f"train_tokens_{shard:03d}.npy"
#             val_path   = CACHE_DIR / f"val_tokens_{shard:03d}.npy"
#
#             if train_buf:
#                 _flush(train_buf, train_path)
#                 train_files.append(train_path)
#
#             if val_buf:
#                 _flush(val_buf, val_path)
#                 val_files.append(val_path)
#
#             # Only bump `shard` if we wrote at least one file
            # if train_buf or val_buf:
            #     shard += 1

    # # (the “end-of-loop” cleanup remains the same)
    # _flush(train_buf, CACHE_DIR / f"train_tokens_{shard:03d}.npy")
    # _flush(val_buf,   CACHE_DIR / f"val_tokens_{shard:03d}.npy")
    # if train_buf:
    #     train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
    # if val_buf:
    #     val_files.append(CACHE_DIR / f"val_tokens_{shard:03d}.npy")
    # pbar.close()
    # return train_files, val_files

# def _encode_stream(tokenizer, ctx: int,
#                    subset_pct: float, chunk_pct: float) -> Tuple[List[Path], List[Path]]:
#     # ds              = load_dataset(DATASET_NAME, split="train")       # this does not work
#     ds              = load_dataset("wikitext", "wikitext-103-v1", split="train") # this gets us to tokenizing
#     total_examples  = len(ds)
#     limit_examples  = math.ceil(total_examples * subset_pct / 100)
#     per_shard_input = max(1, math.floor(limit_examples * chunk_pct / 100))
#
#     train_files, val_files          = [], []
#     train_buf: List[List[int]] = []
#     val_buf:   List[List[int]] = []
#     shard = 0
#
#     pbar = tqdm(total=limit_examples, desc="tokenising", unit="example")
#     for i, rec in enumerate(ds):
#         if i >= limit_examples:
#             break
#         pbar.update()
#
#         ids = tokenizer.encode(rec["text"], add_special_tokens=False)
#         for s in range(0, len(ids) - ctx + 1, ctx):
#             window = ids[s:s + ctx]
#             (val_buf if random.random() < VAL_SPLIT_PCT / 100 else train_buf).append(window)
#
#         if (i + 1) % per_shard_input == 0:
#             _flush(train_buf, CACHE_DIR / f"train_tokens_{shard:03d}.npy")
#             _flush(val_buf,   CACHE_DIR / f"val_tokens_{shard:03d}.npy")
#             train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
#             val_files.append(CACHE_DIR / f"val_tokens_{shard:03d}.npy")
#             shard += 1
#
#     _flush(train_buf, CACHE_DIR / f"train_tokens_{shard:03d}.npy")
#     _flush(val_buf,   CACHE_DIR / f"val_tokens_{shard:03d}.npy")
#     if train_buf:
#         train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
#     if val_buf:
#         val_files.append(CACHE_DIR / f"val_tokens_{shard:03d}.npy")
#     pbar.close()
#     return train_files, val_files


# def _concat(shards):
#     return np.concatenate([np.load(p, mmap_mode="r") for p in shards], axis=0)
def _concat(shards):
    """Load a list of .npy / .npz shards and return one big (N, ctx) array."""
    arrays = []
    for p in shards:
        if p.suffix == ".npz":
            with np.load(p, mmap_mode="r") as z:
                # by construction each shard contains one array named "data"
                arrays.append(z["data"])
        else:  # .npy – the legacy uncompressed format
            arrays.append(np.load(p, mmap_mode="r"))
    return np.concatenate(arrays, axis=0)



def get_data(*, subset_pct: float = 100.0, chunk_pct: float = 10.0,
             context_length: int = 256):
    """Main entry-point used from Run_training.py"""
    CACHE_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # train_shards = sorted(CACHE_DIR.glob("train_tokens_*.npy"))
    # val_shards   = sorted(CACHE_DIR.glob("val_tokens_*.npy"))
    train_shards = sorted(CACHE_DIR.glob("train_tokens_*.npz"))
    val_shards   = sorted(CACHE_DIR.glob("val_tokens_*.npz"))
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

