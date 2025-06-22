"""
generate_math_dataset.py

Builds a closed‐vocabulary arithmetic dataset of all expressions
“N1 op N2 op N3 = RESULT” where RESULT ∈ [0..122], 
splits off 3 % for eval, and saves as an Arrow DatasetDict.
"""

import itertools
from datasets import Dataset, DatasetDict
import numpy as np

# 1. Configuration: numbers 0–121, ops, and formatting
MAX_NUM = 121
NUMBERS = list(range(MAX_NUM + 1))
OPS = ["+", "-", "*"]

def format_num(n: int) -> str:
    """Zero-pad to three digits, e.g.  5 → “005”."""
    return f"{n:03d}"

def compute(a: int, op: str, b: int) -> int:
    """Compute a ∘ b for op in {+, –, *}."""
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    raise ValueError(f"Unknown op {op}")

def gen_examples():
    """Yields dicts {"text": "..."} for every valid arithmetic row."""
    for n1, op1, n2, op2, n3 in itertools.product(NUMBERS, OPS, NUMBERS, OPS, NUMBERS):
        res = compute(compute(n1, op1, n2), op2, n3)
        # keep only results that fit in our [0..MAX_NUM] token range
        if 0 <= res <= MAX_NUM:
            txt = f"{format_num(n1)} {op1} {format_num(n2)} {op2} {format_num(n3)} = {format_num(res)}"
            yield {"text": txt}

def main():
    # 2. Build a Dataset via from_generator (streams to Arrow on the fly) :contentReference[oaicite:0]{index=0}
    full_ds = Dataset.from_generator(gen_examples)

    # 3. Shuffle and split: 3% eval, 97% train :contentReference[oaicite:1]{index=1}
    split = full_ds.train_test_split(test_size=0.03, seed=42)

    # 4. Package into a DatasetDict
    ds = DatasetDict({
        "train": split["train"].shuffle(seed=42),
        "validation": split["test"].shuffle(seed=42),
    })

    # 5. Persist to disk as Arrow :contentReference[oaicite:2]{index=2}
    ds.save_to_disk("math_dataset_arrow")

    print("Saved math_dataset_arrow/ with splits:", {k: len(ds[k]) for k in ds})

if __name__ == "__main__":
    main()

