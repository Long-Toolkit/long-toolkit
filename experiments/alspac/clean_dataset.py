"""Clean ALSPAC for ML by dropping the ID column and mean-imputing NaNs.

This is the **deliberately minimal** preprocessing step requested for the
ALSPAC sweep. Refine it as findings from `inspect_dataset.py` accumulate
(e.g. one-hot encoding of categoricals, dropping high-NaN/constant columns,
removing sentinel codes).

Run with:

    uv run python -m experiments.alspac.clean_dataset \\
        --input data/ALSPAC/alspac.csv \\
        --output data/ALSPAC/alspac_ml_ready.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

DEFAULT_INPUT_PATH = Path("data/ALSPAC/alspac.csv")
DEFAULT_OUTPUT_PATH = Path("data/ALSPAC/ALSPAC_READY.csv")
DEFAULT_DROP_COLUMNS: tuple[str, ...] = ("subject",)
DEFAULT_TARGET_COLUMN = "class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--drop-columns",
        type=str,
        nargs="*",
        default=list(DEFAULT_DROP_COLUMNS),
        help="Columns to drop unconditionally before imputation.",
    )
    parser.add_argument(
        "--impute-target",
        action="store_true",
        help="Also mean-impute the target column (off by default — rows with "
        "missing target are dropped instead).",
    )
    return parser.parse_args()


def clean_dataset(
    *,
    frame: pd.DataFrame,
    drop_columns: list[str],
    target_column: str,
    impute_target: bool,
) -> pd.DataFrame:
    cleaned = frame.copy()
    existing_drops = [column for column in drop_columns if column in cleaned.columns]
    cleaned = cleaned.drop(columns=existing_drops)

    if target_column not in cleaned.columns:
        raise KeyError(
            f"Target column `{target_column}` is missing from the dataset."
        )

    if not impute_target:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=[target_column]).reset_index(drop=True)
        dropped = before - len(cleaned)
        if dropped:
            print(
                f"Dropped {dropped} row(s) with missing target `{target_column}`."
            )

    feature_columns = [
        column for column in cleaned.columns if column != target_column
    ] if not impute_target else list(cleaned.columns)

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    non_numeric_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    if non_numeric_columns:
        raise TypeError(
            "Mean imputation only supports numeric columns; non-numeric columns "
            f"present: {non_numeric_columns}. Refine this script (e.g. add "
            "one-hot encoding) before re-running."
        )

    cleaned.loc[:, numeric_columns] = cleaned[numeric_columns].fillna(
        cleaned[numeric_columns].mean()
    )
    return cleaned


def main() -> None:
    args = parse_args()
    console = Console()
    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    raw = pd.read_csv(args.input)
    console.print(
        f"[bold]Loaded[/bold] {raw.shape[0]} rows × {raw.shape[1]} columns "
        f"from {args.input}"
    )

    cleaned = clean_dataset(
        frame=raw,
        drop_columns=list(args.drop_columns),
        target_column=args.target,
        impute_target=args.impute_target,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)

    remaining_nans = int(cleaned.isna().sum().sum())
    console.print(
        f"[bold green]Wrote[/bold green] {cleaned.shape[0]} rows × "
        f"{cleaned.shape[1]} columns to {args.output} "
        f"(remaining NaNs: {remaining_nans})"
    )


if __name__ == "__main__":
    main()
