#!/usr/bin/env python3
"""
Sanity check for the TinyStories tokenizer.

Verifies:
  - Encoding/decoding roundtrip
  - Special token IDs
  - TiDAR mask token integration

Usage:
  python TiDAR/model/Tiny_tokenizer/check_tokenizer.py --tokenizer_path /proj/giant-data/TiDAR/Tiny_tokenizer
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Check TinyStories tokenizer")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/proj/giant-data/TiDAR/Tiny_tokenizer",
        help="Path to tokenizer directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tok_path = Path(args.tokenizer_path)

    print(f"[check] Loading tokenizer from {tok_path}")

    # Load tokenizer
    tok = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
    
    print("\n" + "=" * 60)
    print("TOKENIZER INFO")
    print("=" * 60)
    print(f"  Vocab size:     {len(tok)}")
    print(f"  pad_token:      '{tok.pad_token}' (id={tok.pad_token_id})")
    print(f"  bos_token:      '{tok.bos_token}' (id={tok.bos_token_id})")
    print(f"  eos_token:      '{tok.eos_token}' (id={tok.eos_token_id})")
    print(f"  unk_token:      '{tok.unk_token}' (id={tok.unk_token_id})")
    
    # Check special token IDs are as expected
    assert tok.pad_token_id == 0, f"Expected pad_token_id=0, got {tok.pad_token_id}"
    assert tok.bos_token_id == 1, f"Expected bos_token_id=1, got {tok.bos_token_id}"
    assert tok.eos_token_id == 2, f"Expected eos_token_id=2, got {tok.eos_token_id}"
    assert tok.unk_token_id == 3, f"Expected unk_token_id=3, got {tok.unk_token_id}"
    print("  ✓ Special token IDs verified")

    # Test encoding/decoding
    print("\n" + "=" * 60)
    print("ENCODING/DECODING TESTS")
    print("=" * 60)
    
    test_texts = [
        "Once upon a time, there was a little girl named Lily.",
        "The dog ran fast.",
        "Hello! How are you?",
        "1 + 2 = 3",
        "The cat sat on the mat.",
    ]
    
    all_passed = True
    for text in test_texts:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        match = (text == decoded)
        status = "✓" if match else "✗"
        print(f"\n  {status} Input:   {text!r}")
        print(f"    Tokens:  {len(ids)}")
        print(f"    IDs:     {ids[:10]}{'...' if len(ids) > 10 else ''}")
        print(f"    Decoded: {decoded!r}")
        if not match:
            print(f"    WARNING: Roundtrip mismatch!")
            all_passed = False
    
    # Test with special tokens
    print("\n" + "=" * 60)
    print("SPECIAL TOKEN HANDLING")
    print("=" * 60)
    
    text = "Hello world!"
    ids_with_special = tok.encode(text, add_special_tokens=True)
    ids_without_special = tok.encode(text, add_special_tokens=False)
    
    print(f"\n  Text: {text!r}")
    print(f"  With special tokens:    {ids_with_special}")
    print(f"  Without special tokens: {ids_without_special}")
    
    # Test TiDAR mask token integration
    print("\n" + "=" * 60)
    print("TIDAR MASK TOKEN INTEGRATION TEST")
    print("=" * 60)
    
    # Import the TiDAR tokenizer utility
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from TiDAR.model.tokenizer_utils import ensure_tidar_mask_token
    
    print("\n  Testing mask token addition...")
    mask_token, mask_id, added = ensure_tidar_mask_token(tok)
    print(f"  Mask token: '{mask_token}' (id={mask_id})")
    print(f"  Added:      {added} new token(s)")
    print(f"  New vocab size: {len(tok)}")
    
    # Verify mask token works
    test_with_mask = f"The {mask_token} is cute."
    ids = tok.encode(test_with_mask, add_special_tokens=False)
    decoded = tok.decode(ids)
    print(f"\n  Text with mask: {test_with_mask!r}")
    print(f"  IDs: {ids}")
    print(f"  Decoded: {decoded!r}")
    
    # Check mask ID is in the sequence
    if mask_id in ids:
        print(f"  ✓ Mask token ID {mask_id} found in encoded sequence")
    else:
        print(f"  ✗ WARNING: Mask token ID {mask_id} NOT found in encoded sequence")
        all_passed = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_passed:
        print("  ✓ All checks passed!")
    else:
        print("  ✗ Some checks failed - see warnings above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
