from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


SIDE = 9
BOX = 3
N = SIDE * SIDE
ALL = (1 << SIDE) - 1  # 9 bits

ROW_OF = np.array([i // SIDE for i in range(N)], dtype=np.int32)
COL_OF = np.array([i % SIDE for i in range(N)], dtype=np.int32)
BOX_OF = np.array([(r // BOX) * BOX + (c // BOX) for r in range(SIDE) for c in range(SIDE)], dtype=np.int32)


def _pattern(r: int, c: int) -> int:
    return (BOX * (r % BOX) + (r // BOX) + c) % SIDE


def random_solved_grid(rng: np.random.Generator) -> np.ndarray:
    """Fast solved Sudoku via group permutations (no backtracking). Returns (81,) int32 digits 1..9."""
    def _shuffled(seq):
        return rng.permutation(np.array(seq, dtype=np.int32))

    rows = [g * BOX + r for g in _shuffled(range(BOX)) for r in _shuffled(range(BOX))]
    cols = [g * BOX + c for g in _shuffled(range(BOX)) for c in _shuffled(range(BOX))]
    nums = _shuffled(range(1, SIDE + 1))

    grid = np.empty((SIDE, SIDE), dtype=np.int32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            grid[i, j] = nums[_pattern(r, c)]
    return grid.reshape(-1)


def _count_solutions_internal(grid: np.ndarray, limit: int) -> int:
    row_mask = [0] * SIDE
    col_mask = [0] * SIDE
    box_mask = [0] * SIDE
    empties = []

    for i in range(N):
        v = int(grid[i])
        if v == 0:
            empties.append(i)
            continue
        bit = 1 << (v - 1)
        r = int(ROW_OF[i])
        c = int(COL_OF[i])
        b = int(BOX_OF[i])
        if (row_mask[r] & bit) or (col_mask[c] & bit) or (box_mask[b] & bit):
            return 0
        row_mask[r] |= bit
        col_mask[c] |= bit
        box_mask[b] |= bit

    def rec(empty_positions, remaining_limit: int) -> int:
        if not empty_positions:
            return 1

        best_i = -1
        best_pos = -1
        best_mask = 0
        best_bits = 10

        for idx, pos in enumerate(empty_positions):
            r = int(ROW_OF[pos])
            c = int(COL_OF[pos])
            b = int(BOX_OF[pos])
            used = row_mask[r] | col_mask[c] | box_mask[b]
            mask = ALL & (~used)
            bits = mask.bit_count()
            if bits == 0:
                return 0
            if bits < best_bits:
                best_bits = bits
                best_i = idx
                best_pos = pos
                best_mask = mask
                if bits == 1:
                    break

        pos = best_pos
        r = int(ROW_OF[pos])
        c = int(COL_OF[pos])
        b = int(BOX_OF[pos])
        rest = empty_positions[:best_i] + empty_positions[best_i + 1 :]

        count = 0
        mask = best_mask
        while mask and count < remaining_limit:
            bit = mask & -mask
            digit = int(bit.bit_length())  # 1..9

            grid[pos] = digit
            row_mask[r] |= bit
            col_mask[c] |= bit
            box_mask[b] |= bit

            count += rec(rest, remaining_limit - count)

            row_mask[r] ^= bit
            col_mask[c] ^= bit
            box_mask[b] ^= bit
            grid[pos] = 0

            mask ^= bit

        return count

    return rec(empties, limit)


def count_solutions(grid: np.ndarray, limit: int = 2) -> int:
    """Count solutions up to `limit` (early exit). Input is (81,) with 0 for blanks."""
    grid = np.array(grid, dtype=np.int32, copy=True)
    return _count_solutions_internal(grid, int(limit))


def make_puzzle(
    solution: np.ndarray,
    rng: np.random.Generator,
    *,
    min_clues: int = 24,
    max_remove_attempts: int = 200,
) -> np.ndarray:
    """Remove digits while keeping a unique solution. Returns (81,) int32 with 0 blanks."""
    if solution.shape != (N,):
        raise ValueError("solution must be shape (81,)")
    puzzle = np.array(solution, dtype=np.int32, copy=True)

    filled = int(np.count_nonzero(puzzle))
    if min_clues < 1 or min_clues > filled:
        raise ValueError(f"min_clues must be in [1,{filled}]")

    indices = rng.permutation(N).tolist()
    attempts = 0
    for idx in indices:
        if filled <= min_clues:
            break
        if attempts >= max_remove_attempts:
            break
        attempts += 1

        backup = int(puzzle[idx])
        if backup == 0:
            continue
        puzzle[idx] = 0
        if count_solutions(puzzle, limit=2) != 1:
            puzzle[idx] = backup
        else:
            filled -= 1

    return puzzle


def format_grid(tokens: np.ndarray) -> str:
    g = np.array(tokens, dtype=np.int32).reshape(9, 9)
    lines = []
    for r in range(9):
        row = []
        for c in range(9):
            v = int(g[r, c])
            row.append("." if v == 0 else str(v))
        lines.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
        if r in (2, 5):
            lines.append("-" * 21)
    return "\n".join(lines)


@dataclass(frozen=True)
class SudokuDataset:
    train_puzzle: np.ndarray  # (N_train, 81) int32
    train_solution: np.ndarray  # (N_train, 81) int32
    val_puzzle: np.ndarray  # (N_val, 81) int32
    val_solution: np.ndarray  # (N_val, 81) int32


def generate_dataset(
    *,
    train_samples: int,
    val_samples: int,
    min_clues: int,
    seed: int = 0,
    max_remove_attempts: int = 200,
    show_progress: bool = True,
) -> SudokuDataset:
    rng = np.random.default_rng(seed)
    total = int(train_samples) + int(val_samples)

    puzzles = np.zeros((total, N), dtype=np.int32)
    solutions = np.zeros((total, N), dtype=np.int32)

    it = range(total)
    if show_progress:
        it = tqdm(it, desc=f"Generating Sudoku (min_clues={min_clues})")

    for i in it:
        sol = random_solved_grid(rng)
        puz = make_puzzle(sol, rng, min_clues=min_clues, max_remove_attempts=max_remove_attempts)
        puzzles[i] = puz
        solutions[i] = sol

    train_puzzle = puzzles[:train_samples]
    train_solution = solutions[:train_samples]
    val_puzzle = puzzles[train_samples:]
    val_solution = solutions[train_samples:]
    return SudokuDataset(
        train_puzzle=train_puzzle,
        train_solution=train_solution,
        val_puzzle=val_puzzle,
        val_solution=val_solution,
    )


def save_dataset(path: str | Path, ds: SudokuDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_puzzle=ds.train_puzzle,
        train_solution=ds.train_solution,
        val_puzzle=ds.val_puzzle,
        val_solution=ds.val_solution,
        meta_train_samples=np.array([ds.train_puzzle.shape[0]], dtype=np.int32),
        meta_val_samples=np.array([ds.val_puzzle.shape[0]], dtype=np.int32),
    )


def load_dataset(path: str | Path) -> SudokuDataset:
    data = np.load(str(path))
    return SudokuDataset(
        train_puzzle=data["train_puzzle"].astype(np.int32),
        train_solution=data["train_solution"].astype(np.int32),
        val_puzzle=data["val_puzzle"].astype(np.int32),
        val_solution=data["val_solution"].astype(np.int32),
    )


def load_or_generate_dataset(
    *,
    cache_path: str | Path,
    train_samples: int,
    val_samples: int,
    min_clues: int,
    seed: int,
    regen: bool = False,
    max_remove_attempts: int = 200,
) -> SudokuDataset:
    cache_path = Path(cache_path)
    if cache_path.exists() and not regen:
        data = np.load(str(cache_path))
        try:
            cached_train = int(np.array(data.get("meta_train_samples", [-1]))[0])
            cached_val = int(np.array(data.get("meta_val_samples", [-1]))[0])
            cached_min_clues = int(np.array(data.get("meta_min_clues", [-1]))[0])
            cached_seed = int(np.array(data.get("meta_seed", [-1]))[0])
        except Exception:
            cached_train = cached_val = cached_min_clues = cached_seed = -1

        if (
            "train_puzzle" in data
            and "train_solution" in data
            and "val_puzzle" in data
            and "val_solution" in data
            and data["train_puzzle"].shape == (train_samples, N)
            and data["train_solution"].shape == (train_samples, N)
            and data["val_puzzle"].shape == (val_samples, N)
            and data["val_solution"].shape == (val_samples, N)
            and cached_train == train_samples
            and cached_val == val_samples
            and cached_min_clues == min_clues
            and cached_seed == seed
        ):
            return SudokuDataset(
                train_puzzle=data["train_puzzle"].astype(np.int32),
                train_solution=data["train_solution"].astype(np.int32),
                val_puzzle=data["val_puzzle"].astype(np.int32),
                val_solution=data["val_solution"].astype(np.int32),
            )

    ds = generate_dataset(
        train_samples=train_samples,
        val_samples=val_samples,
        min_clues=min_clues,
        seed=seed,
        max_remove_attempts=max_remove_attempts,
        show_progress=True,
    )
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_puzzle=ds.train_puzzle,
        train_solution=ds.train_solution,
        val_puzzle=ds.val_puzzle,
        val_solution=ds.val_solution,
        meta_train_samples=np.array([train_samples], dtype=np.int32),
        meta_val_samples=np.array([val_samples], dtype=np.int32),
        meta_min_clues=np.array([min_clues], dtype=np.int32),
        meta_seed=np.array([seed], dtype=np.int32),
    )
    return ds
