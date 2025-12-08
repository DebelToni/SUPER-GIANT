#!/usr/bin/env python3
import argparse
import os
from collections import Counter
from typing import Iterable, List

from transformers import AutoTokenizer


def iter_files(paths: List[str]) -> Iterable[str]:
    """
    Given a list of paths, yield paths of all text files.
    If a path is a directory, walk it recursively.
    If it's a file, yield it directly.
    """
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for name in files:
                    yield os.path.join(root, name)
        else:
            print(f"Warning: path not found or not a file/dir: {p}")


def iter_texts(paths: List[str], max_files: int | None = None) -> Iterable[str]:
    """
    Stream text content from files.

    - Opens each file as UTF-8 with errors ignored
    - Yields line by line (so memory usage stays low)
    """
    count = 0
    for file_path in iter_files(paths):
        if max_files is not None and count >= max_files:
            break
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
            count += 1
        except Exception as e:
            print(f"Could not read {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect used tokens for INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 "
            "from a given dataset (English + Bulgarian + code)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input files and/or directories containing text data."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many files to scan (for quick tests).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save used tokens as a TSV file: token_id<TAB>count<TAB>token_string",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=100,
        help="How many most frequent tokens to print to stdout (default: 100).",
    )

    args = parser.parse_args()

    print("Loading tokenizer: INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 ...")
    tokenizer = AutoTokenizer.from_pretrained("INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0")

    token_counts = Counter()

    print("Streaming text and collecting token usage...")
    for i, text in enumerate(iter_texts(args.paths, max_files=args.max_files), start=1):
        # Tokenize without adding special tokens (we care about actual data tokens)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = encoded["input_ids"]
        token_counts.update(ids)

        if i % 1000 == 0:
            print(f"Processed {i} lines...")

    print(f"\nDone. Saw {len(token_counts)} unique tokens out of vocab size {tokenizer.vocab_size}.")

    # Sort tokens by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # Convert token IDs to strings
    token_ids = [tid for tid, _ in sorted_tokens]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    # Print top-N to stdout
    top_n = min(args.show_top, len(sorted_tokens))
    print(f"\nTop {top_n} most frequent tokens:\n")
    for (tid, cnt), tstr in zip(sorted_tokens[:top_n], token_strings[:top_n]):
        print(f"{tid}\t{cnt}\t{repr(tstr)}")

    # Optionally save all used tokens
    if args.output is not None:
        print(f"\nSaving all used tokens to {args.output} ...")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("token_id\tcount\ttoken\n")
            for (tid, cnt), tstr in zip(sorted_tokens, token_strings):
                # use repr() to make control chars visible but still valid text
                f.write(f"{tid}\t{cnt}\t{repr(tstr)}\n")
        print("Saved.")


if __name__ == "__main__":
    main()

