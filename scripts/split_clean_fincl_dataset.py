#!/usr/bin/env python3
"""Split the clean FinCL dataset into memory/training and test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("materials/FinCL-eval-subset-clean.csv")
DEFAULT_MEMORY_OUTPUT = Path("materials/FinCL-eval-subset-clean-memory.csv")
DEFAULT_TEST_OUTPUT = Path("materials/FinCL-eval-subset-clean-test.csv")
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 300


def split_dataset(
    input_csv: Path,
    memory_output_csv: Path,
    test_output_csv: Path,
    test_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    if test_size <= 0 or test_size >= len(df):
        raise ValueError(f"test_size must be between 1 and {len(df) - 1}; got {test_size}")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:test_size].reset_index(drop=True)
    memory_df = shuffled.iloc[test_size:].reset_index(drop=True)

    memory_output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_output_csv.parent.mkdir(parents=True, exist_ok=True)
    memory_df.to_csv(memory_output_csv, index=False)
    test_df.to_csv(test_output_csv, index=False)
    return memory_df, test_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Clean FinCL CSV.")
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=DEFAULT_MEMORY_OUTPUT,
        help="Output CSV for memory build/training rows.",
    )
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT, help="Output CSV for test rows.")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE, help="Number of rows in the test set.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_df, test_df = split_dataset(
        args.input,
        args.memory_output,
        args.test_output,
        args.test_size,
        args.seed,
    )
    print(f"Wrote {len(memory_df)} memory/training rows to {args.memory_output}")
    print(f"Wrote {len(test_df)} test rows to {args.test_output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
