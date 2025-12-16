from __future__ import annotations

import argparse

import numpy as np

from sudoku_dataset import format_grid, make_puzzle, random_solved_grid


def parse_args():
    p = argparse.ArgumentParser("Generate sample (puzzle, solution) Sudoku pairs")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--min_clues", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_remove_attempts", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        sol = random_solved_grid(rng)
        puz = make_puzzle(sol, rng, min_clues=args.min_clues, max_remove_attempts=args.max_remove_attempts)

        print("=" * 60)
        print(f"pair {i+1}/{args.n}  clues={int(np.count_nonzero(puz))}")
        print("\n[puzzle]")
        print(format_grid(puz))
        print("\n[solution]")
        print(format_grid(sol))


if __name__ == "__main__":
    main()

