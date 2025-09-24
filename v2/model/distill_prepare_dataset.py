# prepare_dataset.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple

import numpy as np
import jax.numpy as jnp
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from transformers import AutoTokenizer
from omegaconf import OmegaConf

Config = OmegaConf.load("Config.yml")

@dataclass
class Batch:
    input: np.ndarray         # (B, T) int32
    target: np.ndarray        # (B, T) int32
    mask: np.ndarray          # (B, T) float32
    topk_ids: np.ndarray      # (B, T, K) int32   (-1 where N/A)
    topk_logprobs: np.ndarray # (B, T, K) float32 (-inf where N/A)

def _open_answers_table() -> pa.Table:
    base = Path(Config.dataset_path)
    fname = getattr(Config, "teacher_answers_filename", "answers.arrow")
    path = base / fname
    if not path.exists():
        raise FileNotFoundError(f"Teacher answers Arrow not found: {path}")
    with path.open("rb") as f:
        reader = pa_ipc.open_file(f)
        tbl = reader.read_all()
    return tbl

def _parse_topk_str(s: str, k: int) -> Tuple[List[int], List[float]]:
    """
    's' is a compact JSON produced by teacher_generate.parse_topk_json
    It may be:
      - dict: {token: logprob, ...}
      - list: [{token: "...", "logprob": -0.1}, ...]
    Returns lists of token-ids and logprobs (length <= k).
    """
    if not s:
        return [], []
    try:
        obj = json.loads(s)
    except Exception:
        return [], []
    toks: List[str] = []
    lps:  List[float] = []
    if isinstance(obj, dict):
        for tok, lp in obj.items():
            toks.append(tok)
            lps.append(float(lp))
    elif isinstance(obj, list):
        for d in obj:
            if not isinstance(d, dict):
                continue
            toks.append(d.get("token", ""))
            lps.append(float(d.get("logprob", float("-inf"))))
    # cut to k
    toks = toks[:k]; lps = lps[:k]
    return toks, lps

# def _align_topk_for_record(
#     tokenizer, ids: List[int], loss_mask: List[int], topk_json_per_token: List[str], k: int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Build per-position (len=seq_len) topk ids/logprobs.
#     For non-answer tokens (mask==0) fill with -1 / -inf.
#     topk_json_per_token is only for generated answer tokens and is iterated
#     in order across positions where loss_mask==1.
#     """
#     T = len(ids)
#     topk_ids = np.full((T, k), -1, dtype=np.int32)
#     topk_lp  = np.full((T, k), -np.inf, dtype=np.float32)
#
#     n = min(len(topk_json_per_token), len(answer_positions))
#     if n == 0:
#         return topk_ids, topk_lp  # nothing to align
#     for pos, json_str in zip(answer_positions[:n], topk_json_per_token[:n]):
#         toks, lps = _parse_topk_str(json_str, k)
#
#         toks, lps = _parse_topk_str(json_str, k)
#         if not toks:
#             continue
#         # Convert teacher tokens → student token ids (byte-level & merges already in tokenizer)
#         # NOTE: vLLM tokens are string pieces; AutoTokenizer.decode/encode may be needed.
#         # We try encode without special tokens; we also handle cases where 'toks' is already a single-piece token.
#         ids_k: List[int] = []
#         for t in toks:
#             if t == "":
#                 continue
#             # robust path: encode the piece exactly as-is
#             enc = tokenizer.encode(t, add_special_tokens=False)
#             if len(enc) == 1:
#                 ids_k.append(int(enc[0]))
#             else:
#                 # If it maps to multiple pieces, we skip it (KD would be noisy).
#                 continue
#         # cut/align to k
#         ids_k = ids_k[:k]
#         lps_k = lps[: len(ids_k)]
#         if not ids_k:
#             continue
#         topk_ids[pos, :len(ids_k)] = np.asarray(ids_k, dtype=np.int32)
#         topk_lp[pos, :len(ids_k)]  = np.asarray(lps_k, dtype=np.float32)
#
#     return topk_ids, topk_lp

def _iter_windows_for_record(
    ids: List[int],
    roles: List[int],
    mask: List[int],
    topk_ids_full: np.ndarray,
    topk_lp_full: np.ndarray,
    ctx: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Turn one variable-length sequence into many (ctx)-length windows for LM training.
    We create input/target by shifting 1, and slice topk arrays accordingly.
    """
    # We need at least ctx+1 tokens to make (ctx) targets after shift
    if len(ids) < ctx + 1:
        return
    # stride = ctx (no overlap). You can change to smaller stride if desired.
    stride = ctx
    L = len(ids) - 1  # because of shift
    for start in range(0, L - ctx + 1, stride):
        s = start
        e = start + ctx + 1  # include next token for target
        chunk_ids   = np.asarray(ids[s:e], dtype=np.int32)
        chunk_mask  = np.asarray(mask[s:e], dtype=np.int8)
        chunk_topkI = topk_ids_full[s:e]
        chunk_topkL = topk_lp_full[s:e]

        inp    = chunk_ids[:-1]
        tgt    = chunk_ids[1:]
        msk    = chunk_mask[1:]        # mask next-token prediction (answer region only)
        topkI  = chunk_topkI[1:]
        topkL  = chunk_topkL[1:]
        yield inp, tgt, msk, topkI, topkL

def _split_indices(n: int, train_pct: float = 0.9):
    # simple split using qid parity or round-robin if present else 90/10 by index
    n_train = int(n * train_pct)
    return set(range(n_train)), set(range(n_train, n))

def _read_column(tbl, name):
    return tbl[name].to_pylist() if name in tbl.column_names else []

def _materialize_batches(tbl, ctx, k, batch_size, subset_pct):
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name, use_fast=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    ids_list   = _read_column(tbl, "student_input_ids")
    roles_list = _read_column(tbl, "student_role_ids")
    mask_list  = _read_column(tbl, "student_loss_mask")
    topk_list  = _read_column(tbl, "topk_json_per_token")

    assert len(ids_list) == len(roles_list) == len(mask_list) == len(topk_list), \
        "Arrow columns length mismatch. Re-generate with teacher_generate.py"

    total = len(ids_list)
    if subset_pct and 0 < subset_pct < 100:
        total = int(total * (subset_pct / 100.0))

    train_idx, val_idx = _split_indices(total)

    def _iterator(indexes: set[int]) -> Iterator[Batch]:
        B = batch_size; K = k
        buf_inp, buf_tgt, buf_msk, buf_kI, buf_kL = [], [], [], [], []
        emitted = 0
        for i in range(total):
            if i not in indexes:
                continue
            ids   = ids_list[i]
            msk   = mask_list[i]
            kd    = topk_list[i]

            # --- tolerant KD alignment (EOS/off-by-one safe) ---
            topkI_full = np.full((len(ids), K), -1, dtype=np.int32)
            topkL_full = np.full((len(ids), K), -np.inf, dtype=np.float32)
            answer_positions = [p for p, m in enumerate(msk) if m == 1]
            n = min(len(kd), len(answer_positions))
            for pos, json_str in zip(answer_positions[:n], kd[:n]):
                toks, lps = _parse_topk_str(json_str, K)
                if not toks:
                    continue
                ids_k = []
                for t in toks[:K]:
                    enc = tokenizer.encode(t, add_special_tokens=False)
                    if len(enc) == 1:
                        ids_k.append(int(enc[0]))
                if len(ids_k) == 0:
                    continue
                Lk = min(len(ids_k), len(lps))
                topkI_full[pos, :Lk] = np.asarray(ids_k[:Lk], np.int32)
                topkL_full[pos, :Lk] = np.asarray(lps[:Lk], np.float32)
            # ----------------------------------------------------

            result = _make_single_padded_window(
                ids, msk, topkI_full, topkL_full, ctx=ctx, eos_id=eos_id, pad_id=pad_id
            )
            if result is None:
                continue  # Skip sequences that are too short
            inp, tgt, m, kI, kL = result
            buf_inp.append(inp); buf_tgt.append(tgt); buf_msk.append(m); buf_kI.append(kI); buf_kL.append(kL)

            if len(buf_inp) == B:
                yield Batch(
                    input=np.stack(buf_inp, 0),
                    target=np.stack(buf_tgt, 0),
                    mask=np.stack(buf_msk, 0),
                    topk_ids=np.stack(buf_kI, 0),
                    topk_logprobs=np.stack(buf_kL, 0),
                )
                emitted += 1
                buf_inp.clear(); buf_tgt.clear(); buf_msk.clear(); buf_kI.clear(); buf_kL.clear()

        if emitted == 0:
            print("[prepare_dataset] WARNING: 0 batches emitted for this split — "
                  "check context_length or dataset_percent.")
    return _iterator(train_idx), _iterator(val_idx), tokenizer

# Public API used by Run_training.py
def get_data(*, subset_pct: float, context_length: int, batch_size: int, **_):
    tbl = _open_answers_table()

    # Debug: Check data size first
    ids_list = _read_column(tbl, "student_input_ids")
    print(f"[DEBUG] Total records in Arrow table: {len(ids_list)}")
    if ids_list:
        print(f"[DEBUG] Sample sequence length: {len(ids_list[0])}")

    train_it, val_it, tokenizer = _materialize_batches(
        tbl, ctx=context_length, k=getattr(Config, "distill_topk", 8),
        batch_size=batch_size, subset_pct=subset_pct
    )

    return train_it, val_it, tokenizer

def data_loader(iterator: Iterator[Batch]) -> Iterator[Dict[str, np.ndarray]]:
    # present batches as dicts for Training_step
    for b in iterator:
        yield {
            "input": b.input,
            "target": b.target,
            "mask": b.mask,
            "topk_ids": b.topk_ids,
            "topk_logprobs": b.topk_logprobs,
        }

def _make_single_padded_window(
    ids: List[int],
    loss_mask: List[int],
    topk_ids_full: np.ndarray,
    topk_lp_full: np.ndarray,
    ctx: int,
    eos_id: int | None,
    pad_id: int,
):
    """
    Returns a single (input[T], target[T], mask[T], topk_ids[T,K], topk_logprobs[T,K])
    with T = ctx (your context_length). We build targets by shifting by 1.
    If the sequence is longer than ctx+1, we truncate from the LEFT (keep the most recent tail).
    If shorter, we LEFT-pad with pad_id, zeros for mask, and -1/-inf for KD.
    We also try to ensure an EOS at the end: if eos_id is provided and the last token is not EOS,
    we append one before truncation/padding (so next-token prediction has a sensible end).
    """
    # Optionally append EOS once
    if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
        ids = ids + [eos_id]
        loss_mask = loss_mask + [loss_mask[-1] if len(loss_mask) > 0 else 0]
        topk_ids_full = np.concatenate([topk_ids_full, np.full((1, topk_ids_full.shape[1]), -1, np.int32)], axis=0)
        topk_lp_full  = np.concatenate([topk_lp_full,  np.full((1, topk_lp_full.shape[1]), -np.inf, np.float32)], axis=0)

    # We need ctx+1 tokens to create ctx targets after a 1-step shift.
    need = ctx + 1
    L = len(ids)

    # If the sequence is too short for meaningful training, skip it
    # But allow sequences that can be padded to the required length
    if L < 2:  # Need at least 2 tokens for input/target
        return None

    # Build arrays aligned with ids
    ids_arr  = np.asarray(ids, dtype=np.int32)
    msk_arr  = np.asarray(loss_mask, dtype=np.int8)
    kI_arr   = topk_ids_full
    kL_arr   = topk_lp_full

    # Truncate or pad so that len == need
    if L >= need:
        # keep the most recent tail
        ids_arr = ids_arr[-need:]
        msk_arr = msk_arr[-need:]
        kI_arr  = kI_arr[-need:]
        kL_arr  = kL_arr[-need:]
    else:
        pad = need - L
        ids_arr = np.pad(ids_arr, (pad, 0), constant_values=pad_id)
        msk_arr = np.pad(msk_arr, (pad, 0), constant_values=0)
        kI_arr  = np.pad(kI_arr,  ((pad,0),(0,0)), constant_values=-1)
        kL_arr  = np.pad(kL_arr,  ((pad,0),(0,0)), constant_values=-np.inf)

    # Now shift to build input/target of length ctx
    inp   = ids_arr[:-1]
    tgt   = ids_arr[1:]
    mask  = msk_arr[1:].astype(np.float32)
    topkI = kI_arr[1:]
    topkL = kL_arr[1:]
    assert inp.shape[0] == ctx and tgt.shape[0] == ctx

    return inp, tgt, mask, topkI, topkL

