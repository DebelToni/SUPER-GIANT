# math_env.py
"""Environment helpers for RL fine‑tuning on toy arithmetic.

The grammar is limited to two operands so that the ground‑truth result is
always representable in the 0‑121 range covered by *math_tokenizer.py*.
Feel free to extend this once you enlarge the vocabulary.
"""
from __future__ import annotations

import operator
import random
from typing import Tuple

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


def sample_problem() -> Tuple[str, int]:
    """Return a tuple *(expression_string, correct_answer_int).*"""
    while True:
        a = random.randint(0, 121)
        b = random.randint(0, 121)
        op = random.choice(list(OPS))
        result = OPS[op](a, b)
        # Restrict result so that it is still encodable with the existing vocab
        if 0 <= result <= 121:
            expr = f"{a:03d} {op} {b:03d} ="
            return expr, result

