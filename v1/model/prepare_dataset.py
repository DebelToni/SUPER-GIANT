# from datasets      import load_dataset
# from transformers  import AutoTokenizer
# import numpy as np, Config
#
# _TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
#
# def get_data():
#     print("Entering get_data()")
#     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
#     dataset   = load_dataset("roneneldan/TinyStories", split="all")
#     print(f"Loaded dataset with {len(dataset)} examples")
#
#     dataset = dataset.select(range(int(0.1 * len(dataset))))  # use only 10% for testing
#
#     # Encode every story, then chunk into context_length tokens
#     ctx = Config.context_length
#     sequences = []
#     for ex in dataset["text"]:
#         # print(f"Encoding story: {ex[:50]}...")  # print first 50 chars for context
#         ids = tokenizer.encode(ex, add_special_tokens=False)
#         # pad or chunk to fixed length
#         # print(f"Encoding story of length {len(ids)} tokens")
#         for i in range(0, len(ids), ctx):
#             chunk = ids[i:i+ctx]
#             if len(chunk) < ctx:
#                 chunk += [tokenizer.eos_token_id]*(ctx-len(chunk))
#             sequences.append(chunk)
#
#     print(f"Encoded {len(sequences)} sequences of length {ctx} tokens")
#
#     sequences = np.array(sequences, dtype=np.int32)
#     np.random.shuffle(sequences)
#     split = int(0.97*len(sequences))     # 97 % train / 3 % val
#     return sequences[:split], sequences[split:], tokenizer
#######################
# prepare_dataset.py
# from datasets import load_dataset, IterableDataset
# from transformers import AutoTokenizer
# import numpy as np, Config, os, tqdm, pyarrow as pa, pyarrow.parquet as pq
#
# _TOKENIZER = "EleutherAI/gpt-neo-125M"
# _CACHE_DIR = "./tiny_cached"                 # <-- stays on the VM’s drive
#
# def _tokenise_batch(batch, tokenizer, ctx):
#     # batch["text"] is a list[str]
#     out = []
#     for t in batch["text"]:
#         ids = tokenizer.encode(t, add_special_tokens=False)
#         for i in range(0, len(ids), ctx):
#             chunk = ids[i : i+ctx]
#             if len(chunk) < ctx:
#                 chunk += [tokenizer.eos_token_id] * (ctx - len(chunk))
#             out.append(chunk)
#     # ‘out’ is now list[list[int]] length ctx
#     return {"tokens": out}
#
# def get_data():
#     if os.path.exists(_CACHE_DIR):                 #  ⬅︎ reuse
#         print("▶ Using cached Arrow tensors")
#         tbl = pq.read_table(f"{_CACHE_DIR}/data.parquet")
#         token_arr = np.asarray(tbl["tokens"].combine_chunks(), dtype=np.int32)
#         tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
#         return token_arr, tokenizer                # train≡val split later
#
#     print("▶ First run – streaming encode")
#     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
#     ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
#
#     ctx = Config.context_length
#     batched = ds.map(
#         lambda b: _tokenise_batch(b, tokenizer, ctx),
#         batched=True,
#         remove_columns=["text"],
#     )
#
#     # Materialise to Arrow on disk once
#     # rows = []
#     rows = []
#     for batch in tqdm.tqdm(batched, desc="encoding", unit="story"):
#         tk = batch["tokens"]
#         # tk is either a list[list[int]] or list[int]; detect and wrap.
#         if tk and isinstance(tk[0], int):
#             rows.append(tk)              # already one full chunk
#         else:
#             rows.extend(tk)              # many chunks → extend
#
#     # for batch in tqdm.tqdm(batched, desc="encoding", unit="story"):
#     #     rows.extend(batch["tokens"])
#     tbl = pa.Table.from_arrays([pa.array(rows, type=pa.list_(pa.int32()))],
#                                names=["tokens"])
#     os.makedirs(_CACHE_DIR, exist_ok=True)
#     pq.write_table(tbl, f"{_CACHE_DIR}/data.parquet")
#
#     token_arr = np.asarray(rows, dtype=np.int32)
#
#     # train_tokens = token_arr[:int(0.97 * len(token_arr))]
#     # val_tokens = token_arr[int(0.97 * len(token_arr)):]
#     return token_arr, tokenizer
#     # return train_tokens, val_tokens, tokenizer
#
# prepare_dataset.py  – robust, compact version
import os, tqdm, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import Config                                        # your config file

_TOKENIZER = "EleutherAI/gpt-neo-125M"
_CACHE_DIR = "./tiny_cached"
_NPY_FILE  = os.path.join(_CACHE_DIR, "tokens.npy")

def _encode_stream(tokenizer, ctx):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    rows = []
    for ex in tqdm.tqdm(ds, desc="encoding", unit="story"):
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        for i in range(0, len(ids), ctx):
            chunk = ids[i : i + ctx]
            chunk += [tokenizer.eos_token_id] * (ctx - len(chunk))
            rows.append(chunk)
    return np.asarray(rows, dtype=np.int32)          # 2-D

def get_data():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)

    if os.path.exists(_NPY_FILE):
        print("▶ Using cached .npy tensors")
        token_arr = np.load(_NPY_FILE, mmap_mode="r")    # zero-copy
    else:
        print("▶ First run – streaming encode")
        token_arr = _encode_stream(tokenizer, Config.context_length)
        np.save(_NPY_FILE, token_arr)

    # 97 / 3 split like your original code
    split = int(0.97 * len(token_arr))
    return token_arr[:split], token_arr[split:], tokenizer

