"""Token sequence deduplication helpers."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Tuple

import numpy as np


class Deduplicator:
    """Track hashed token windows to filter duplicates."""

    def __init__(self, *, hash_bits: int = 64) -> None:
        if hash_bits < 8:
            raise ValueError("hash_bits must be >= 8")
        self._mask = (1 << hash_bits) - 1 if hash_bits < 64 else None
        self._seen: set[int] = set()

    def _hash_tokens(self, tokens: Iterable[int]) -> int:
        arr = np.asarray(list(tokens), dtype=np.int32)
        digest = hashlib.blake2b(arr.tobytes(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        if self._mask is not None:
            value &= self._mask
        return value

    def check_and_add(self, tokens: Iterable[int]) -> Tuple[bool, int]:
        """Return (is_duplicate, key) for the provided tokens."""
        key = self._hash_tokens(tokens)
        is_dup = key in self._seen
        if not is_dup:
            self._seen.add(key)
        return is_dup, key

    def reset(self) -> None:
        self._seen.clear()


__all__ = ["Deduplicator"]
