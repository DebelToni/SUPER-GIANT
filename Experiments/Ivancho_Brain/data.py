from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_cfg(path: str):
    return OmegaConf.load(path)


def tokenize_tinystories(config_path: str, *, max_tokens: int | None = None) -> Path:
    cfg = load_cfg(config_path)
    out_path = Path(cfg.paths.tokenized_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.tokenizer_path, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None or eos_id < 0:
        raise ValueError("Tokenizer needs an EOS token or <|endoftext|> token")

    token_budget = int(max_tokens or cfg.data.max_tokens)
    dataset = load_dataset(cfg.data.dataset_name, split=cfg.data.split)
    text_field = str(cfg.data.text_field)
    chunks: list[np.ndarray] = []
    total = 0
    for row in tqdm(dataset, desc="tokenizing TinyStories"):
        text = row.get(text_field)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(int(eos_id))
        remaining = token_budget - total
        if remaining <= 0:
            break
        if len(ids) > remaining:
            ids = ids[:remaining]
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if total >= token_budget:
            break

    if not chunks:
        raise RuntimeError("No tokens produced")
    tokens = np.concatenate(chunks)
    np.save(out_path, tokens)
    print(f"saved {tokens.size:,} tokens to {out_path}")
    return out_path


class TokenBatcher:
    def __init__(self, tokens: np.ndarray, *, seq_len: int, batch_size: int, seed: int):
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        if self.tokens.size <= self.seq_len + 1:
            raise ValueError("Token array is too small for requested seq_len")

    def next_batch(self) -> np.ndarray:
        max_start = self.tokens.size - self.seq_len - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        return np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int32)


def load_train_val_tokens(config_path: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_cfg(config_path)
    path = Path(cfg.paths.tokenized_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenized data not found: {path}. Run data.py tokenize first.")
    tokens = np.load(path)
    val_n = int(cfg.data.validation_tokens)
    if tokens.size <= val_n + int(cfg.data.seq_len) + 1:
        raise ValueError("Not enough tokens for train/validation split")
    return tokens[:-val_n], tokens[-val_n:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tokenize"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("Config.yml")))
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.command == "tokenize":
        tokenize_tinystories(args.config, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
