# math_env.py
"""Tiny helper for RL-fine-tuning GiantGPT on arithmetic.

Functions
---------
sample_problem()          -> (expr_str, truth_int)
sample_batch(B)           -> (list[str], list[int])
encode_batch(tok, exprs, ctx_len) -> list[list[int]]
"""

import random
from typing import List, Tuple
import jax.numpy as jnp

# ---------------------------------------------------------------------------

_OPS = ["+", "-", "*"]


def _draw_numbers(op: str) -> Tuple[int, int]:
    """Pick two numbers so that the result ∈ [0, 121]."""
    while True:
        a = random.randint(0, 121)
        b = random.randint(0, 121)
        if op == "+" and a + b <= 121:
            return a, b
        if op == "-" and a - b >= 0:
            return a, b
        if op == "*":
            # keep products small enough; try a few times
            if a * b <= 121:
                return a, b

# math_env.py  –  excerpt
MAX_TERMS = 3          # ≤ Config.max_expression_terms

def sample_problem(max_terms: int = MAX_TERMS) -> Tuple[str, int]:
    """Random chain of 2…max_terms numbers and operators."""
    n_terms = random.randint(2, max_terms)

    nums = [random.randint(0, 121) for _ in range(n_terms)]
    ops  = [random.choice(_OPS)     for _ in range(n_terms - 1)]

    # Evaluate left-to-right (like a hand calculator)
    val = nums[0]
    for op, b in zip(ops, nums[1:]):
        if op == "+": val += b
        elif op == "-": val -= b
        else:          val *= b

    # Reject results that fall outside the vocabulary; resample
    if not (0 <= val <= 121):
        return sample_problem(max_terms)

    expr_parts = [f"{nums[0]:03d}"]
    for op, num in zip(ops, nums[1:]):
        expr_parts.append(op)
        expr_parts.append(f"{num:03d}")
    expr = " ".join(expr_parts) + " ="
    return expr, val


def sample_batch(batch_size: int, max_terms: int = MAX_TERMS):
    exprs, truths = zip(*(sample_problem(max_terms) for _ in range(batch_size)))
    return list(exprs), list(truths)

# def sample_problem() -> Tuple[str, int]:
#     """Return one valid expression and its ground-truth answer.
#
#     Example:
#         "047 + 015 =" , 62
#     """
#     op = random.choice(_OPS)
#     a, b = _draw_numbers(op)
#
#     if op == "+":
#         truth = a + b
#     elif op == "-":
#         truth = a - b
#     else:  # "*"
#         truth = a * b
#
#     expr = f"{a:03d} {op} {b:03d} ="
#     return expr, truth
#
#
# def sample_batch(batch_size: int) -> Tuple[List[str], List[int]]:
#     """Vectorised wrapper around *sample_problem()*."""
#     exprs, truths = zip(*(sample_problem() for _ in range(batch_size)))
#     return list(exprs), list(truths)


# ---------------------------------------------------------------------------
# Token-helper --------------------------------------------------------------
# ---------------------------------------------------------------------------
def encode_batch(tokenizer, exprs, ctx_len):
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id

    tokens   = []
    lengths  = []

    for expr in exprs:
        ids = tokenizer.encode(expr, add_special_tokens=False)

        # Strip the trailing EOS that our math tokenizer always appends
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]

        lengths.append(len(ids))          # position of '='  →  lengths[i]-1
        if len(ids) > ctx_len:
            ids = ids[:ctx_len]           # just in case
            lengths[-1] = ctx_len

        tokens.append(ids + [pad_id] * (ctx_len - len(ids)))

    return tokens, jnp.array(lengths, dtype=jnp.int32)
