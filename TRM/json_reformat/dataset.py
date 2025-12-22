from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, List, Tuple

import numpy as np


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class JsonVocab:
    chars: List[str]
    stoi: dict[str, int]
    pad_id: int
    unk_id: int


@dataclass(frozen=True)
class JsonReformatDataset:
    train_input: np.ndarray
    train_target: np.ndarray
    train_mask: np.ndarray
    val_input: np.ndarray
    val_target: np.ndarray
    val_mask: np.ndarray
    vocab: JsonVocab


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.strip().split())


def load_pairs(path: str | Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = _normalize_text(str(obj["raw"]))
            fixed = _normalize_text(str(obj["fixed"]))
            pairs.append((raw, fixed))
    return pairs


def build_vocab(pairs: Iterable[Tuple[str, str]]) -> JsonVocab:
    chars = set()
    for raw, fixed in pairs:
        chars.update(raw)
        chars.update(fixed)
    sorted_chars = sorted(chars)
    tokens = [PAD_TOKEN, UNK_TOKEN] + sorted_chars
    stoi = {ch: idx for idx, ch in enumerate(tokens)}
    return JsonVocab(chars=tokens, stoi=stoi, pad_id=stoi[PAD_TOKEN], unk_id=stoi[UNK_TOKEN])


def save_vocab(vocab: JsonVocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "chars": vocab.chars,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def load_vocab(path: str | Path) -> JsonVocab:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    chars = list(data["chars"])
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    pad_id = stoi.get(data.get("pad_token", PAD_TOKEN), 0)
    unk_id = stoi.get(data.get("unk_token", UNK_TOKEN), 1)
    return JsonVocab(chars=chars, stoi=stoi, pad_id=pad_id, unk_id=unk_id)


def encode(text: str, vocab: JsonVocab, *, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.full((max_len,), vocab.pad_id, dtype=np.int32)
    mask = np.zeros((max_len,), dtype=np.float32)

    text = _normalize_text(text)
    for i, ch in enumerate(text[:max_len]):
        ids[i] = vocab.stoi.get(ch, vocab.unk_id)
        mask[i] = 1.0
    return ids, mask


def decode(ids: np.ndarray, vocab: JsonVocab) -> str:
    chars = []
    for idx in ids.tolist():
        if idx == vocab.pad_id:
            break
        chars.append(vocab.chars[idx] if 0 <= idx < len(vocab.chars) else "?")
    return "".join(chars)


def _build_arrays(pairs: List[Tuple[str, str]], vocab: JsonVocab, *, max_len: int):
    n = len(pairs)
    src = np.full((n, max_len), vocab.pad_id, dtype=np.int32)
    tgt = np.full((n, max_len), vocab.pad_id, dtype=np.int32)
    mask = np.zeros((n, max_len), dtype=np.float32)

    for i, (raw, fixed) in enumerate(pairs):
        src_ids, _src_mask = encode(raw, vocab, max_len=max_len)
        tgt_ids, tgt_mask = encode(fixed, vocab, max_len=max_len)
        src[i] = src_ids
        tgt[i] = tgt_ids
        mask[i] = tgt_mask

    return src, tgt, mask


def load_dataset(
    *,
    train_path: str | Path,
    val_path: str | Path,
    vocab_path: str | Path,
    max_len: int,
) -> JsonReformatDataset:
    train_pairs = load_pairs(train_path)
    val_pairs = load_pairs(val_path)

    vocab_file = Path(vocab_path)
    if vocab_file.exists():
        vocab = load_vocab(vocab_file)
    else:
        vocab = build_vocab(train_pairs + val_pairs)
        save_vocab(vocab, vocab_file)

    train_input, train_target, train_mask = _build_arrays(train_pairs, vocab, max_len=max_len)
    val_input, val_target, val_mask = _build_arrays(val_pairs, vocab, max_len=max_len)

    return JsonReformatDataset(
        train_input=train_input,
        train_target=train_target,
        train_mask=train_mask,
        val_input=val_input,
        val_target=val_target,
        val_mask=val_mask,
        vocab=vocab,
    )
