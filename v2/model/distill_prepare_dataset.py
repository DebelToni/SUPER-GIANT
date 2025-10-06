# prepare_dataset.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple

import numpy as np
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

def _answers_path() -> Path:
    base = Path(Config.dataset_path)
    fname = getattr(Config, "teacher_answers_filename", "answers.arrow")
    path = base / fname
    if not path.exists():
        raise FileNotFoundError(f"Teacher answers Arrow not found: {path}")
    return path

def _parse_topk_str(s: str, k: int) -> Tuple[List[int], List[float]]:
    """
    's' is a compact JSON produced by teacher_generate.parse_topk_json
    It may be:
      - dict: {token: logprob, ...}
      - list: [{token: "...", "logprob": -0.1}, ...]
      - list of lists: [["token", logprob], ...]  # new format from teacher
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
            if isinstance(d, dict):
                # Format: [{"token": "...", "logprob": -0.1}, ...]
                toks.append(d.get("token", ""))
                lps.append(float(d.get("logprob", float("-inf"))))
            elif isinstance(d, list) and len(d) >= 2:
                # Format: [["token", logprob], ...] - new format from teacher
                token_str = str(d[0]) if d[0] is not None else ""
                logprob_val = float(d[1]) if len(d) > 1 else float("-inf")
                toks.append(token_str)
                lps.append(logprob_val)
            else:
                # Skip invalid entries
                continue
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

# NOTE: legacy multi-window function removed. We use exactly one window per entry.


def _stream_iterator(*, split: str, ctx: int, k: int, batch_size: int, subset_pct: float) -> Iterator[Batch]:
    """
    Stream the Arrow file record-batch by record-batch and yield training batches.
    No full-table materialization. Deterministic subsampling + split via row index.
    """
    path = _answers_path()
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name, use_fast=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    # Deterministic approx-subset: keep rows where (row_idx % 100) < subset_mod
    subset_mod = int(round(subset_pct)) if (subset_pct and 0 < subset_pct < 100) else 100
    subset_mod = max(1, min(100, subset_mod))

    B = batch_size; K = k
    buf_inp, buf_tgt, buf_msk, buf_kI, buf_kL = [], [], [], [], []
    emitted = 0

    with path.open("rb") as f:
        reader = pa_ipc.open_file(f)
        row_idx = 0
        # Iterate record-batches without loading entire table
        for bi in range(reader.num_record_batches):
            rb = reader.get_batch(bi)
            # Extract columns as Python lists per *batch* (not whole file)
            ids_col  = rb.column(rb.schema.get_field_index("student_input_ids")).to_pylist()
            mask_col = rb.column(rb.schema.get_field_index("student_loss_mask")).to_pylist()
            kd_col   = rb.column(rb.schema.get_field_index("topk_json_per_token")).to_pylist()

            for ids, msk, kd in zip(ids_col, mask_col, kd_col):
                # subset gating
                if (row_idx % 100) >= subset_mod:
                    row_idx += 1
                    continue
                # split gating: 90/10 via modulo (train: indices not ≡9 mod 10; val: ≡9 mod 10)
                is_val = (row_idx % 10) == 9
                if (split == "train" and is_val) or (split == "val" and not is_val):
                    row_idx += 1
                    continue

                # ---- KD alignment (tolerant to off-by-1 / EOS) ----
                topkI_full = np.full((len(ids), K), -1, dtype=np.int32)
                topkL_full = np.full((len(ids), K), -np.inf, dtype=np.float32)
                answer_positions = [p for p, m in enumerate(msk) if m == 1]

                # Handle both formats: single JSON strings or lists of JSON strings
                if isinstance(kd, list) and kd and isinstance(kd[0], list):
                    # New format: kd is a list of lists of JSON strings
                    # Flatten the list of lists
                    flat_kd = []
                    for sublist in kd:
                        if isinstance(sublist, list):
                            flat_kd.extend(sublist)
                        else:
                            flat_kd.append(sublist)
                    kd = flat_kd

                n = min(len(kd), len(answer_positions))
                for pos, json_str in zip(answer_positions[:n], kd[:n]):
                    toks, lps = _parse_topk_str(json_str, K)
                    if not toks:
                        continue
                    ids_k = []
                    for t in toks[:K]:
                        if not t:
                            continue
                        enc = tokenizer.encode(t, add_special_tokens=False)
                        if len(enc) == 1:
                            ids_k.append(int(enc[0]))
                    if not ids_k:
                        continue
                    Lk = min(len(ids_k), len(lps))
                    topkI_full[pos, :Lk] = np.asarray(ids_k[:Lk], np.int32)
                    topkL_full[pos, :Lk] = np.asarray(lps[:Lk], np.float32)
                # ---------------------------------------------------

                result = _make_single_padded_window(
                    ids, msk, topkI_full, topkL_full, ctx=ctx, eos_id=eos_id, pad_id=pad_id
                )
                row_idx += 1
                if result is None:
                    continue  # L < 2 → skip by design

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
        print(f"[prepare_dataset] WARNING: 0 batches emitted for split={split} — "
              "check context_length, dataset_percent, or data coverage.")

# Public API used by Run_training.py
def get_data(*, subset_pct: float, context_length: int, batch_size: int, **_):
    """
    Returns *factory functions* that create fresh streaming iterators
    (avoids generator exhaustion). No full in-RAM materialization.
    """
    k = int(getattr(Config, "distill_topk", 8))
    # Small probe to print tokenizer info (and ensure it's available)
    tok = AutoTokenizer.from_pretrained(Config.tokenizer_name, use_fast=True)
    print(f"[DEBUG] tokenizer vocab_size={len(tok)}")

    def train_factory():
        return _stream_iterator(split="train", ctx=context_length, k=k, batch_size=batch_size, subset_pct=subset_pct)
    def val_factory():
        return _stream_iterator(split="val",   ctx=context_length, k=k, batch_size=batch_size, subset_pct=subset_pct)
    # one quick probe to report KD coverage on a few batches
    it = train_factory()
    try:
        b = next(it)
        has_any = (b.topk_ids >= 0).any(axis=-1)   # (B,T)

        # Calculate padding percentage
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        is_padding = (b.input == pad_id)
        padding_percentage = is_padding.mean() * 100.0

        # Calculate KD coverage excluding padding positions
        non_padding_positions = ~is_padding
        if non_padding_positions.sum() > 0:
            coverage_non_padding = (has_any & non_padding_positions).sum() / non_padding_positions.sum() * 100.0
        else:
            coverage_non_padding = 0.0

        coverage_total = has_any.mean() * 100.0

        print(f"[DEBUG] KD coverage on first batch: {coverage_non_padding:.1f}% of non-padding positions have teacher candidates")
        print(f"[DEBUG] Dataset padding: {padding_percentage:.1f}% of positions are padding tokens")
        print(f"[DEBUG] Raw KD coverage (including padding): {coverage_total:.1f}%")
    except StopIteration:
        pass
    return train_factory, val_factory, tok

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

