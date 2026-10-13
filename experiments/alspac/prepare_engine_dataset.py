"""Rewrite ``ALSPAC_READY.csv`` so the existing experiments engine can ingest it.

The engine in ``experiments/engine.py`` infers longitudinal feature groups from
``_w<n>$`` suffixes and identifies non-longitudinal columns by the
``non_long__`` prefix. ALSPAC uses bare column names for its non-longitudinal
features, so we add the prefix here and keep wave columns untouched. The
target column (``class``) is left as-is.

Run with:

    uv run python -m experiments.alspac.prepare_engine_dataset
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rich.console import Console

DEFAULT_INPUT_PATH = Path("data/ALSPAC/ALSPAC_READY.csv")
DEFAULT_OUTPUT_PATH = Path("data/ALSPAC/ALSPAC_READY_engine.csv")
DEFAULT_TARGET_COLUMN = "class"
WAVE_SUFFIX_PATTERN = re.compile(r"_w\d+$", flags=re.IGNORECASE)
NON_LONG_PREFIX = "non_long__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_COLUMN)
    return parser.parse_args()


def add_non_long_prefix(
    *, frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rename_map: dict[str, str] = {}
    longitudinal: list[str] = []
    non_longitudinal: list[str] = []
    for column in frame.columns:
        if column == target_column:
            continue
        if column.startswith(NON_LONG_PREFIX):
            non_longitudinal.append(column)
            continue
        if WAVE_SUFFIX_PATTERN.search(column) is not None:
            longitudinal.append(column)
            continue
        new_name = f"{NON_LONG_PREFIX}{column}"
        rename_map[column] = new_name
        non_longitudinal.append(new_name)
    return frame.rename(columns=rename_map), longitudinal, non_longitudinal


def main() -> None:
    args = parse_args()
    console = Console()
    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    raw = pd.read_csv(args.input)
    if args.target not in raw.columns:
        raise KeyError(f"Target column `{args.target}` missing from {args.input}.")

    renamed, longitudinal, non_longitudinal = add_non_long_prefix(
        frame=raw, target_column=args.target
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renamed.to_csv(args.output, index=False)
    console.print(
        f"[bold green]Wrote[/bold green] {renamed.shape[0]} rows × "
        f"{renamed.shape[1]} columns to {args.output}\n"
        f"  longitudinal columns: {len(longitudinal)}\n"
        f"  non_long__ columns:   {len(non_longitudinal)}\n"
        f"  target:               {args.target}"
    )


if __name__ == "__main__":
    main()
