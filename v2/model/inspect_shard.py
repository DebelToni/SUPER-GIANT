# inspect_shard.py
import sys, numpy as np
from pathlib import Path
from transformers import AutoTokenizer

TOK_PATH = "neo-english-cust"            # folder you saved earlier
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)

def main(npz_file: Path, row: int = 0):
    with np.load(npz_file, mmap_mode="r") as z:
        arr = z["data"]                   # shape = (N_windows, ctx)
    ctx_len = arr.shape[1]
    pad_id  = tokenizer.pad_token_id
    pad_pct = (arr == pad_id).mean()*100

    print(f"Loaded {npz_file.name}  →  {arr.shape}  (context={ctx_len})")
    print(f"PAD tokens overall: {pad_pct:.2f}%\n")

    # pick a window to display (default row 0)
    toks = arr[row]
    print("IDs :", toks[:24], "...", toks[-6:])  # show a slice

    print("\nDecoded w/ specials:")
    print(tokenizer.decode(toks, skip_special_tokens=False))

    print("\nDecoded clean (skip_special_tokens=True):")
    print(tokenizer.decode(toks, skip_special_tokens=True))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python inspect_shard.py train_tokens_000.npz [row]")
    npz_path = Path(sys.argv[1])
    row_idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(npz_path, row_idx)
