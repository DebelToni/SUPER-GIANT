# math_env.py
"""Tiny environment for on‑the‑fly generation of simple arithmetic problems
suitable for RL fine‑tuning.

The vocabulary in *math_tokenizer.py* can represent integers 0‒121 as three‑
character zero‑padded tokens ("000" … "121") plus the operator tokens
"+", "‑" and "*", and the equals sign "=".  To keep the answer always
representable we restrict operands accordingly (for multiplication we cap
factors at 11 because 11×11 = 121).

Exports
-------
sample_problem() → (expr: str, answer: int)
    Returns a single expression such as "047 + 015 =" and its ground‑truth
    result (62).

sample_batch(batch_size: int) → (list[str], list[int])
    Convenience wrapper that calls *sample_problem* `batch_size` times and
    returns parallel lists of prompts and answers.
"""
from __future__ import annotations

import operator
import random
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Supported binary operators — must align with the tokenizer’s symbols
# ---------------------------------------------------------------------------
OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}

# Precompute things that speed up sampling a bit
_BINARY_OPS = list(OPS.keys())
_MAX_SAFE_FACTOR = 11        # 11 * 11 <= 121 so product always valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_number(n: int) -> str:
    """Return *n* as a zero‑padded string of width 3 (000‒999)."""
    return f"{n:03d}"


def sample_problem() -> Tuple[str, int]:
    """Generate one valid arithmetic expression and its result.

    The function keeps drawing random operands/operators until the computed
    answer is within the 0‒121 range inclusive.
    """
    while True:
        op = random.choice(_BINARY_OPS)

        if op == "*":            # keep products in range
            a = random.randint(0, _MAX_SAFE_FACTOR)
            b = random.randint(0, _MAX_SAFE_FACTOR)
        else:                    # + or − can use the full range
            a = random.randint(0, 121)
            b = random.randint(0, 121)

        result = OPS[op](a, b)
        if 0 <= result <= 121:
            expr = f"{_format_number(a)} {op} {_format_number(b)} ="
            return expr, result
        # else loop again – very cheap given tiny ranges


def sample_batch(batch_size: int) -> Tuple[List[str], List[int]]:
    """Vectorised wrapper around *sample_problem*.

    Parameters
    ----------
    batch_size : int
        Number of independent expressions to draw.

    Returns
    -------
    exprs : list[str]
        The prompts ready for tokenisation.
    truths : list[int]
        The ground‑truth answers aligned with *exprs*.
    """
    exprs, truths = zip(*(sample_problem() for _ in range(batch_size)))
    return list(exprs), list(truths)


__all__ = ["sample_problem", "sample_batch"]

