#!/usr/bin/env python3
"""Inspect UltraChat shard for NaN/zero-mask rows and mask validity.

Checks:
- loss_mask length matches seq_len
- loss_mask values are finite and in [0, 1]
- no rows with all-zero loss_mask
- no rows with all-zero token_mask after length_mask applied
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def load_tokenizer():
    cfg_path = Path(__file__).resolve().parent / "Global_Config.yml"
    cfg = OmegaConf.load(cfg_path)
    tok_cfg = cfg.get("tokenizer", {})
    name = tok_cfg.get("name")
    cache_dir = tok_cfg.get("cache_dir")
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id set")
    return tokenizer


def main() -> None:
    shard_path = Path(
        "/proj/giant-data/TiDAR/Sweep/Data/ultrachat_rehearsal_1m/"
        "ultrachat_rehearsal_1m-000000.arrow"
    )
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    tokenizer = load_tokenizer()
    pad_id = int(tokenizer.pad_token_id)

    reader = pa.ipc.RecordBatchFileReader(open(shard_path, "rb"))
    table = reader.read_all()
    tokens = table["input_ids"].to_pylist()
    loss_masks = table["loss_mask"].to_pylist()

    total_rows = len(tokens)
    if total_rows == 0:
        print("No rows found")
        return

    seq_len = len(tokens[0])
    positions = np.arange(seq_len, dtype=np.int32)

    invalid_len = []
    non_finite_mask = []
    out_of_range_mask = []
    zero_loss_mask = []
    zero_token_mask = []

    mask_sums = []
    token_mask_sums = []

    for idx, (row_tokens, row_mask) in enumerate(zip(tokens, loss_masks)):
        if len(row_tokens) != seq_len or len(row_mask) != seq_len:
            invalid_len.append(idx)
            continue

        mask_arr = np.asarray(row_mask, dtype=np.float32)
        if not np.isfinite(mask_arr).all():
            non_finite_mask.append(idx)
        if (mask_arr < 0).any() or (mask_arr > 1).any():
            out_of_range_mask.append(idx)

        mask_sum = float(mask_arr.sum())
        mask_sums.append(mask_sum)
        if mask_sum == 0:
            zero_loss_mask.append(idx)

        length_from_tokens = int(np.sum(np.asarray(row_tokens) != pad_id))
        valid_target_len = max(length_from_tokens - 1, 1)
        length_mask = (positions < valid_target_len).astype(np.float32)
        token_mask = length_mask * mask_arr
        token_mask_sum = float(token_mask.sum())
        token_mask_sums.append(token_mask_sum)
        if token_mask_sum == 0:
            zero_token_mask.append(idx)

    print("UltraChat shard inspection")
    print(f"- rows: {total_rows}")
    print(f"- seq_len: {seq_len}")
    print(f"- pad_id: {pad_id}")
    print(f"- invalid length rows: {len(invalid_len)}")
    print(f"- non-finite loss_mask rows: {len(non_finite_mask)}")
    print(f"- out-of-range loss_mask rows: {len(out_of_range_mask)}")
    print(f"- all-zero loss_mask rows: {len(zero_loss_mask)}")
    print(f"- all-zero token_mask rows: {len(zero_token_mask)}")
    print(f"- loss_mask sum min/max: {min(mask_sums):.1f}/{max(mask_sums):.1f}")
    print(f"- token_mask sum min/max: {min(token_mask_sums):.1f}/{max(token_mask_sums):.1f}")

    if zero_loss_mask:
        print(f"- example zero loss_mask row: {zero_loss_mask[0]}")
    if zero_token_mask:
        print(f"- example zero token_mask row: {zero_token_mask[0]}")


if __name__ == "__main__":
    main()
