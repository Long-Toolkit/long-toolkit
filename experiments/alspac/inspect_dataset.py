"""ML-readiness inspector for the ALSPAC dataset.

Run with:

    uv run python -m experiments.alspac.inspect_dataset \\
        --input data/ALSPAC/alspac.csv \\
        --target class
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

DEFAULT_DATASET_PATH = Path("data/ALSPAC/alspac.csv")
DEFAULT_TARGET_COLUMN = "class"
DEFAULT_ID_COLUMNS = ("subject",)
HIGH_NULL_THRESHOLD = 0.90
LOW_VARIANCE_THRESHOLD = 0.90
SUSPICIOUS_SENTINELS = (-9, -99, -999, -9999, 999, 9999)
WAVE_SUFFIX_PATTERN = re.compile(r"_w\d+$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--id-columns",
        type=str,
        nargs="*",
        default=list(DEFAULT_ID_COLUMNS),
        help="Columns to ignore from feature analysis (always dropped before ML).",
    )
    parser.add_argument(
        "--max-unique-categorical",
        type=int,
        default=15,
        help="Numeric integer columns with at most this many unique values are flagged as candidate categoricals.",
    )
    return parser.parse_args()


def _load_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    return frame


def _summarise_overview(console: Console, frame: pd.DataFrame, target: str) -> None:
    table = Table(title="Dataset overview", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Property")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{len(frame):,}")
    table.add_row("Columns", f"{frame.shape[1]:,}")
    table.add_row(
        "Total cells", f"{frame.shape[0] * frame.shape[1]:,}"
    )
    table.add_row(
        "NaN cells (overall %)",
        f"{frame.isna().sum().sum():,} ({frame.isna().mean().mean() * 100:.2f}%)",
    )
    table.add_row("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    table.add_row(
        "Memory (MB)", f"{frame.memory_usage(deep=True).sum() / 1e6:.2f}"
    )
    table.add_row("Target column", target)
    if target in frame.columns:
        target_distribution = frame[target].value_counts(dropna=False).sort_index()
        table.add_row(
            "Target classes",
            ", ".join(f"{label}={count}" for label, count in target_distribution.items()),
        )
    console.print(table)


def _dtype_breakdown(console: Console, frame: pd.DataFrame) -> None:
    table = Table(title="Pandas dtype breakdown", box=box.SIMPLE_HEAVY)
    table.add_column("Dtype")
    table.add_column("# columns", justify="right")
    table.add_column("Examples")
    for dtype, columns in frame.dtypes.groupby(frame.dtypes).groups.items():
        sample = ", ".join(list(columns)[:5])
        table.add_row(str(dtype), str(len(columns)), sample)
    console.print(table)


def _high_null_or_constant(
    console: Console,
    frame: pd.DataFrame,
    *,
    null_threshold: float,
    constant_threshold: float,
) -> None:
    null_share = frame.isna().mean()
    high_null = null_share[null_share >= null_threshold].sort_values(ascending=False)

    constant_share = pd.Series(
        {
            column: (
                frame[column].value_counts(normalize=True, dropna=False).iloc[0]
                if not frame[column].dropna().empty or frame[column].isna().all()
                else 0.0
            )
            for column in frame.columns
        }
    )
    high_constant = constant_share[constant_share >= constant_threshold].sort_values(
        ascending=False
    )

    table = Table(
        title=(
            f"Columns with ≥{int(null_threshold * 100)}% NaN "
            f"or ≥{int(constant_threshold * 100)}% single-value share"
        ),
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Column")
    table.add_column("NaN share", justify="right")
    table.add_column("Top-value share", justify="right")
    table.add_column("Top value")
    flagged = sorted(set(high_null.index) | set(high_constant.index))
    if not flagged:
        console.print(
            "[green]No columns exceed the NaN / single-value thresholds.[/green]"
        )
        return
    for column in flagged:
        top_value = (
            frame[column].value_counts(dropna=False).index[0]
            if frame[column].notna().any() or frame[column].isna().all()
            else "—"
        )
        table.add_row(
            column,
            f"{null_share[column] * 100:.1f}%",
            f"{constant_share[column] * 100:.1f}%",
            str(top_value)[:30],
        )
    console.print(table)


def _candidate_categoricals(
    console: Console,
    frame: pd.DataFrame,
    *,
    target: str,
    id_columns: list[str],
    max_unique: int,
) -> None:
    table = Table(
        title=(
            "Candidate categorical columns "
            f"(integer-coded, ≤{max_unique} unique non-NaN values)"
        ),
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Column")
    table.add_column("Dtype")
    table.add_column("# unique", justify="right")
    table.add_column("NaN %", justify="right")
    table.add_column("Sample values")
    table.add_column("Suggested encoding")

    skip = {target, *id_columns}
    flagged_any = False
    for column in frame.columns:
        if column in skip:
            continue
        series = frame[column]
        non_na = series.dropna()
        if non_na.empty:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_object = series.dtype == object
        unique_values = non_na.unique()
        n_unique = len(unique_values)

        if is_object:
            encoding = "one-hot" if n_unique <= max_unique else "target / hash"
        elif is_numeric:
            looks_integer = bool(np.all(np.equal(np.mod(non_na.astype(float), 1.0), 0)))
            if not looks_integer or n_unique > max_unique:
                continue
            if n_unique == 2 and set(map(int, unique_values)) <= {0, 1}:
                encoding = "binary (keep as-is)"
            else:
                encoding = "one-hot"
        else:
            continue

        flagged_any = True
        table.add_row(
            column,
            str(series.dtype),
            str(n_unique),
            f"{series.isna().mean() * 100:.1f}%",
            ", ".join(map(str, sorted(unique_values, key=str)[:8])),
            encoding,
        )

    if not flagged_any:
        console.print("[yellow]No categorical-looking columns found.[/yellow]")
        return
    console.print(table)


def _suspicious_values(console: Console, frame: pd.DataFrame) -> None:
    rows: list[tuple[str, str, int, float]] = []
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_na = series.dropna()
        if non_na.empty:
            continue
        for sentinel in SUSPICIOUS_SENTINELS:
            count = int((non_na == sentinel).sum())
            if count > 0:
                rows.append(
                    (column, str(sentinel), count, count / len(non_na) * 100)
                )
    if not rows:
        return
    table = Table(
        title="Suspicious sentinel values (likely missing-value codes)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Column")
    table.add_column("Sentinel")
    table.add_column("Count", justify="right")
    table.add_column("Share %", justify="right")
    for row in rows:
        table.add_row(row[0], row[1], f"{row[2]:,}", f"{row[3]:.2f}%")
    console.print(table)


def _wave_layout(console: Console, frame: pd.DataFrame, *, target: str) -> None:
    grouped: dict[str, list[str]] = {}
    non_long: list[str] = []
    for column in frame.columns:
        if column == target:
            continue
        match = WAVE_SUFFIX_PATTERN.search(column)
        if match is None:
            non_long.append(column)
            continue
        base = column[: match.start()]
        grouped.setdefault(base, []).append(column)
    table = Table(
        title="Inferred longitudinal feature groups (suffix `_w<n>`)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Base feature")
    table.add_column("# waves", justify="right")
    table.add_column("Wave columns")
    for base, columns in sorted(grouped.items()):
        table.add_row(
            base,
            str(len(columns)),
            ", ".join(sorted(columns, key=lambda name: int(WAVE_SUFFIX_PATTERN.search(name).group()[2:]))),
        )
    console.print(table)
    console.print(
        f"[bold]Non-longitudinal columns ({len(non_long)}):[/bold] "
        + ", ".join(non_long)
    )


def _try_sklong_loader(
    console: Console, frame: pd.DataFrame, *, target: str, id_columns: list[str]
) -> None:
    console.rule("[bold]scikit-longitudinal LongitudinalDataset preview[/bold]")
    try:
        from scikit_longitudinal.data_preparation import LongitudinalDataset
    except Exception as exc:  # pragma: no cover - dependency present in this repo
        console.print(f"[red]scikit-longitudinal import failed: {exc}[/red]")
        return

    cleaned = frame.drop(columns=[col for col in id_columns if col in frame.columns])
    if target not in cleaned.columns:
        console.print(f"[red]Target `{target}` not found.[/red]")
        return
    dataset = LongitudinalDataset(file_path="<in-memory>", data_frame=cleaned)
    try:
        dataset.load_data()
        dataset.load_target(target_column=target)
    except TypeError:
        # API variant where load_target signature differs
        dataset.load_target(target)
    try:
        dataset.setup_features_group("elsa")
    except Exception as exc:
        console.print(f"[red]setup_features_group('elsa') failed: {exc}[/red]")
        return

    indices = dataset.feature_groups()
    names = dataset.feature_groups(names=True)
    non_long = list(dataset.non_longitudinal_features() or [])

    table = Table(
        title="ELSA preset → feature groups (names)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("# group", justify="right")
    table.add_column("# waves", justify="right")
    table.add_column("Names (oldest → newest, -1 = missing wave)")
    for idx, group in enumerate(names):
        table.add_row(str(idx), str(len(group)), ", ".join(map(str, group)))
    console.print(table)

    console.print(
        f"[bold]Total longitudinal groups:[/bold] {len(indices)}    "
        f"[bold]Non-longitudinal features ({len(non_long)}):[/bold] "
        + (", ".join(map(str, non_long)) if non_long else "[i]none[/i]")
    )


def main() -> None:
    args = parse_args()
    console = Console()
    frame = _load_csv_strict(args.input)
    console.rule(f"[bold]ALSPAC ML-readiness inspector — {args.input}[/bold]")
    _summarise_overview(console, frame, args.target)
    _dtype_breakdown(console, frame)
    _high_null_or_constant(
        console,
        frame,
        null_threshold=HIGH_NULL_THRESHOLD,
        constant_threshold=LOW_VARIANCE_THRESHOLD,
    )
    _candidate_categoricals(
        console,
        frame,
        target=args.target,
        id_columns=args.id_columns,
        max_unique=args.max_unique_categorical,
    )
    _suspicious_values(console, frame)
    _wave_layout(console, frame, target=args.target)
    _try_sklong_loader(
        console,
        frame,
        target=args.target,
        id_columns=list(args.id_columns),
    )


if __name__ == "__main__":
    main()
