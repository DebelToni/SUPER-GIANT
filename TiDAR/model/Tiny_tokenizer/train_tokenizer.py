#!/usr/bin/env python3
"""
Train a Byte-Level BPE tokenizer on TinyStories using HuggingFace Tokenizers.

Output:
  - vocab.json, merges.txt
  - tokenizer.json (full tokenizer config for PreTrainedTokenizerFast)

Usage:
  python TiDAR/model/Tiny_tokenizer/train_tokenizer.py --out_dir /proj/giant-data/TiDAR/Tiny_tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train TinyStories Byte-Level BPE tokenizer")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/proj/giant-data/TiDAR/Tiny_tokenizer",
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8192,
        help="Vocabulary size (default: 8192)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges (default: 2)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode (default: True)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for training (for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] Training Byte-Level BPE tokenizer")
    print(f"[tokenizer] vocab_size={args.vocab_size}, min_frequency={args.min_frequency}")
    print(f"[tokenizer] Output: {out_dir}")

    # Special tokens - order matters for IDs
    # <pad>=0, <bos>=1, <eos>=2, <unk>=3
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Load TinyStories dataset
    print("[dataset] Loading TinyStories (streaming)...")
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=args.streaming,
    )

    def text_iterator():
        count = 0
        for ex in ds:
            yield ex["text"]
            count += 1
            if count % 100000 == 0:
                print(f"[progress] processed {count} documents")
            if args.max_samples and count >= args.max_samples:
                print(f"[progress] reached max_samples={args.max_samples}")
                break

    # Create and train tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    print("[tokenizer] Training on TinyStories...")
    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    # Save the model (vocab.json + merges.txt)
    tokenizer.save_model(str(out_dir))
    print(f"[tokenizer] Saved vocab.json and merges.txt to {out_dir}")

    # Also save the full tokenizer.json for PreTrainedTokenizerFast compatibility
    tokenizer.save(str(out_dir / "tokenizer.json"))
    print(f"[tokenizer] Saved tokenizer.json to {out_dir}")

    # Create tokenizer_config.json for HuggingFace compatibility
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 512,
        "clean_up_tokenization_spaces": True,
    }
    with open(out_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"[tokenizer] Saved tokenizer_config.json to {out_dir}")

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(out_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"[tokenizer] Saved special_tokens_map.json to {out_dir}")

    # Quick test
    print("\n[test] Quick encoding test:")
    from transformers import PreTrainedTokenizerFast
    
    tok = PreTrainedTokenizerFast(tokenizer_file=str(out_dir / "tokenizer.json"))
    tok.bos_token = "<bos>"
    tok.eos_token = "<eos>"
    tok.unk_token = "<unk>"
    tok.pad_token = "<pad>"
    
    test_text = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(test_text, add_special_tokens=False)
    decoded = tok.decode(ids)
    
    print(f"  Input:   {test_text!r}")
    print(f"  Tokens:  {len(ids)} tokens")
    print(f"  IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"  Decoded: {decoded!r}")
    print(f"\n  Special token IDs:")
    print(f"    pad_token_id: {tok.pad_token_id}")
    print(f"    bos_token_id: {tok.bos_token_id}")
    print(f"    eos_token_id: {tok.eos_token_id}")
    print(f"    unk_token_id: {tok.unk_token_id}")
    
    print(f"\n[done] Tokenizer training complete!")
    print(f"[done] Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
