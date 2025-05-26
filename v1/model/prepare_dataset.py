# # from datasets      import load_dataset
# # from transformers  import AutoTokenizer
# # import numpy as np, Config
# #
# # _TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
# #
# # def get_data():
# #     print("Entering get_data()")
# #     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
# #     dataset   = load_dataset("roneneldan/TinyStories", split="all")
# #     print(f"Loaded dataset with {len(dataset)} examples")
# #
# #     dataset = dataset.select(range(int(0.1 * len(dataset))))  # use only 10% for testing
# #
# #     # Encode every story, then chunk into context_length tokens
# #     ctx = Config.context_length
# #     sequences = []
# #     for ex in dataset["text"]:
# #         # print(f"Encoding story: {ex[:50]}...")  # print first 50 chars for context
# #         ids = tokenizer.encode(ex, add_special_tokens=False)
# #         # pad or chunk to fixed length
# #         # print(f"Encoding story of length {len(ids)} tokens")
# #         for i in range(0, len(ids), ctx):
# #             chunk = ids[i:i+ctx]
# #             if len(chunk) < ctx:
# #                 chunk += [tokenizer.eos_token_id]*(ctx-len(chunk))
# #             sequences.append(chunk)
# #
# #     print(f"Encoded {len(sequences)} sequences of length {ctx} tokens")
# #
# #     sequences = np.array(sequences, dtype=np.int32)
# #     np.random.shuffle(sequences)
# #     split = int(0.97*len(sequences))     # 97 % train / 3 % val
# #     return sequences[:split], sequences[split:], tokenizer
# #######################
# # prepare_dataset.py
# # from datasets import load_dataset, IterableDataset
# # from transformers import AutoTokenizer
# # import numpy as np, Config, os, tqdm, pyarrow as pa, pyarrow.parquet as pq
# #
# # _TOKENIZER = "EleutherAI/gpt-neo-125M"
# # _CACHE_DIR = "./tiny_cached"                 # <-- stays on the VM’s drive
# #
# # def _tokenise_batch(batch, tokenizer, ctx):
# #     # batch["text"] is a list[str]
# #     out = []
# #     for t in batch["text"]:
# #         ids = tokenizer.encode(t, add_special_tokens=False)
# #         for i in range(0, len(ids), ctx):
# #             chunk = ids[i : i+ctx]
# #             if len(chunk) < ctx:
# #                 chunk += [tokenizer.eos_token_id] * (ctx - len(chunk))
# #             out.append(chunk)
# #     # ‘out’ is now list[list[int]] length ctx
# #     return {"tokens": out}
# #
# # def get_data():
# #     if os.path.exists(_CACHE_DIR):                 #  ⬅︎ reuse
# #         print("▶ Using cached Arrow tensors")
# #         tbl = pq.read_table(f"{_CACHE_DIR}/data.parquet")
# #         token_arr = np.asarray(tbl["tokens"].combine_chunks(), dtype=np.int32)
# #         tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
# #         return token_arr, tokenizer                # train≡val split later
# #
# #     print("▶ First run – streaming encode")
# #     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
# #     ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
# #
# #     ctx = Config.context_length
# #     batched = ds.map(
# #         lambda b: _tokenise_batch(b, tokenizer, ctx),
# #         batched=True,
# #         remove_columns=["text"],
# #     )
# #
# #     # Materialise to Arrow on disk once
# #     # rows = []
# #     rows = []
# #     for batch in tqdm.tqdm(batched, desc="encoding", unit="story"):
# #         tk = batch["tokens"]
# #         # tk is either a list[list[int]] or list[int]; detect and wrap.
# #         if tk and isinstance(tk[0], int):
# #             rows.append(tk)              # already one full chunk
# #         else:
# #             rows.extend(tk)              # many chunks → extend
# #
# #     # for batch in tqdm.tqdm(batched, desc="encoding", unit="story"):
# #     #     rows.extend(batch["tokens"])
# #     tbl = pa.Table.from_arrays([pa.array(rows, type=pa.list_(pa.int32()))],
# #                                names=["tokens"])
# #     os.makedirs(_CACHE_DIR, exist_ok=True)
# #     pq.write_table(tbl, f"{_CACHE_DIR}/data.parquet")
# #
# #     token_arr = np.asarray(rows, dtype=np.int32)
# #
# #     # train_tokens = token_arr[:int(0.97 * len(token_arr))]
# #     # val_tokens = token_arr[int(0.97 * len(token_arr)):]
# #     return token_arr, tokenizer
# #     # return train_tokens, val_tokens, tokenizer
# #
# # prepare_dataset.py  – robust, compact version
# import os, tqdm, numpy as np
# from datasets import load_dataset
# from transformers import AutoTokenizer
# import Config                                        # your config file
#
# _TOKENIZER = "EleutherAI/gpt-neo-125M"
# _CACHE_DIR = "./tiny_cached"
# _NPY_FILE  = os.path.join(_CACHE_DIR, "tokens.npy")
#
# def _encode_stream(tokenizer, ctx):
#     ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
#     rows = []
#     for ex in tqdm.tqdm(ds, desc="encoding", unit="story"):
#         ids = tokenizer.encode(ex["text"], add_special_tokens=False)
#         for i in range(0, len(ids), ctx):
#             chunk = ids[i : i + ctx]
#             chunk += [tokenizer.eos_token_id] * (ctx - len(chunk))
#             rows.append(chunk)
#     return np.asarray(rows, dtype=np.int32)          # 2-D
#
# def get_data():
#     os.makedirs(_CACHE_DIR, exist_ok=True)
#     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
#
#     if os.path.exists(_NPY_FILE):
#         print("▶ Using cached .npy tensors")
#         token_arr = np.load(_NPY_FILE, mmap_mode="r")    # zero-copy
#     else:
#         print("▶ First run – streaming encode")
#         token_arr = _encode_stream(tokenizer, Config.context_length)
#         np.save(_NPY_FILE, token_arr)
#
#     # 97 / 3 split like your original code
#     split = int(0.97 * len(token_arr))
#     return token_arr[:split], token_arr[split:], tokenizer
#
"""
prepare_dataset.py

Stream-tokenises the TinyStories corpus into fixed-length sequences
and writes a shard to disk every *chunk_pct* percent so you never blow
your Colab RAM.

Public API
----------
    train_tokens, val_tokens, tokenizer = get_data(
        subset_pct = 50,   # encode only 50 % of the corpus
        chunk_pct  = 10,   # write a shard every 10 %
        context_length = 256,
    )
"""
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
    train_files.append(CACHE_DIR / f"train_tokens_{shard:03d}.npy")
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

