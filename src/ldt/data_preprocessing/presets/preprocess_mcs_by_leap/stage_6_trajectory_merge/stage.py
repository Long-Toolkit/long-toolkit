from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "policy.yaml"


@beartype
@dataclass(frozen=True)
class Stage6Config:
    """Runtime configuration for the optional trajectory merge stage."""

    enabled: bool
    trajectory_input_path: Path | None
    original_id_column: str
    trajectory_id_column: str
    trajectory_class_column: str
    output_class_column: str
    drop_target_column_after_merge: bool
    target_column_to_drop: str


@beartype
@dataclass(frozen=True)
class Stage6Summary:
    """Compact stage 6 summary metrics."""

    enabled: bool
    applied: bool
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    trajectory_rows_read: int
    trajectory_rows_after_id_filter: int
    trajectory_unique_ids: int
    duplicate_trajectory_ids_detected: int
    matched_rows: int
    unmatched_rows: int
    original_id_column_resolved: str
    trajectory_id_column_resolved: str
    trajectory_class_column_resolved: str
    output_class_column: str
    target_column_to_drop_requested: str
    target_column_to_drop_resolved: str
    target_column_dropped: bool


@beartype
@dataclass(frozen=True)
class Stage6Result:
    """Output payload for stage 6."""

    data: pd.DataFrame
    summary: Stage6Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_6_config(
    *,
    config_path: Path | None = None,
    trajectory_input_path: Path | str | None = None,
    original_id_column: str = "CHID",
    trajectory_id_column: str = "CHID",
    trajectory_class_column: str = "class",
) -> Stage6Config:
    """Resolve runtime settings for the optional trajectory merge stage."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    path: Path | None = None
    if trajectory_input_path not in (None, ""):
        path = Path(str(trajectory_input_path)).expanduser()

    return Stage6Config(
        enabled=path is not None,
        trajectory_input_path=path,
        original_id_column=_as_string(original_id_column, "original_id_column"),
        trajectory_id_column=_as_string(trajectory_id_column, "trajectory_id_column"),
        trajectory_class_column=_as_string(
            trajectory_class_column,
            "trajectory_class_column",
        ),
        output_class_column=_as_string(
            raw.get("output_class_column", "class"),
            "output_class_column",
        ),
        drop_target_column_after_merge=bool(
            raw.get("drop_target_column_after_merge", True)
        ),
        target_column_to_drop=_as_string(
            raw.get("target_column_to_drop", "EMOTION_w7"),
            "target_column_to_drop",
        ),
    )


@beartype
def apply_stage_6(*, data: pd.DataFrame, config: Stage6Config) -> Stage6Result:
    """Optionally merge externally inferred trajectory classes into the preset output."""

    input_rows, input_columns = int(data.shape[0]), int(data.shape[1])
    duplicate_ids_table = _empty_duplicate_ids_table()
    unmatched_ids_table = _empty_unmatched_ids_table()

    if not config.enabled:
        summary = Stage6Summary(
            enabled=False,
            applied=False,
            input_rows=input_rows,
            output_rows=input_rows,
            input_columns=input_columns,
            output_columns=input_columns,
            trajectory_rows_read=0,
            trajectory_rows_after_id_filter=0,
            trajectory_unique_ids=0,
            duplicate_trajectory_ids_detected=0,
            matched_rows=0,
            unmatched_rows=0,
            original_id_column_resolved="",
            trajectory_id_column_resolved="",
            trajectory_class_column_resolved="",
            output_class_column=config.output_class_column,
            target_column_to_drop_requested=config.target_column_to_drop,
            target_column_to_drop_resolved="",
            target_column_dropped=False,
        )
        return Stage6Result(
            data=data.copy(),
            summary=summary,
            tables={
                "stage6_trajectory_merge_policy": _build_policy_table(
                    config=config,
                    summary=summary,
                ),
                "stage6_trajectory_merge_duplicate_ids": duplicate_ids_table,
                "stage6_trajectory_merge_unmatched_ids": unmatched_ids_table,
            },
        )

    trajectory_input_path = config.trajectory_input_path
    if trajectory_input_path is None:
        raise InputValidationError(
            "Stage 6 trajectory merge is enabled but no trajectory_input_path was provided."
        )
    if not trajectory_input_path.exists() or not trajectory_input_path.is_file():
        raise InputValidationError(
            f"Stage 6 trajectory dataset does not exist: {trajectory_input_path}"
        )
    if trajectory_input_path.suffix.lower() != ".csv":
        raise InputValidationError(
            "Stage 6 trajectory dataset must point to a .csv file."
        )

    original_id_column = _resolve_column_name(
        columns=data.columns,
        requested=config.original_id_column,
        context="stage 6 preset dataset ID column",
    )
    _ensure_output_column_is_available(
        columns=data.columns,
        requested=config.output_class_column,
    )

    trajectory_data = pd.read_csv(trajectory_input_path)
    if trajectory_data.empty:
        raise InputValidationError(
            f"Stage 6 trajectory dataset is empty: {trajectory_input_path.resolve()}"
        )

    trajectory_id_column = _resolve_column_name(
        columns=trajectory_data.columns,
        requested=config.trajectory_id_column,
        context="stage 6 trajectory dataset ID column",
    )
    trajectory_class_column = _resolve_column_name(
        columns=trajectory_data.columns,
        requested=config.trajectory_class_column,
        context="stage 6 trajectory dataset class column",
    )

    subset = trajectory_data[[trajectory_id_column, trajectory_class_column]].copy()
    trajectory_rows_read = int(subset.shape[0])
    subset = subset.dropna(subset=[trajectory_id_column])
    trajectory_rows_after_id_filter = int(subset.shape[0])
    if subset.empty:
        raise InputValidationError(
            "Stage 6 trajectory dataset has no non-null IDs after filtering."
        )

    duplicate_ids_table = _build_duplicate_ids_table(
        data=subset,
        trajectory_id_column=trajectory_id_column,
        trajectory_class_column=trajectory_class_column,
    )
    conflicting_duplicates = duplicate_ids_table[
        duplicate_ids_table["distinct_class_count"] > 1
    ]
    if not conflicting_duplicates.empty:
        sample_ids = ", ".join(
            str(value)
            for value in conflicting_duplicates["trajectory_id"].head(5).tolist()
        )
        raise InputValidationError(
            "Stage 6 trajectory dataset contains duplicate IDs with conflicting "
            f"class assignments. Sample IDs: {sample_ids}"
        )

    collapsed = subset.sort_values(by=[trajectory_id_column]).drop_duplicates(
        subset=[trajectory_id_column],
        keep="first",
    )
    trajectory_unique_ids = int(collapsed.shape[0])

    merge_source = collapsed.rename(
        columns={
            trajectory_id_column: "_stage6_trajectory_id",
            trajectory_class_column: config.output_class_column,
        }
    )
    combined = data.merge(
        merge_source,
        left_on=original_id_column,
        right_on="_stage6_trajectory_id",
        how="left",
        indicator="_stage6_merge_status",
    )

    matched_rows = int((combined["_stage6_merge_status"] == "both").sum())
    unmatched_rows = int((combined["_stage6_merge_status"] == "left_only").sum())
    unmatched_ids_table = (
        combined.loc[combined["_stage6_merge_status"] == "left_only", [original_id_column]]
        .drop_duplicates()
        .rename(columns={original_id_column: "original_id"})
        .sort_values(by=["original_id"])
        .reset_index(drop=True)
    )

    updated = combined.drop(
        columns=["_stage6_trajectory_id", "_stage6_merge_status"],
    )
    target_column_to_drop_resolved = ""
    target_column_dropped = False
    if config.drop_target_column_after_merge:
        target_column_to_drop_resolved = _resolve_column_name(
            columns=updated.columns,
            requested=config.target_column_to_drop,
            context="stage 6 target column to drop",
        )
        updated = updated.drop(columns=[target_column_to_drop_resolved])
        target_column_dropped = True

    summary = Stage6Summary(
        enabled=True,
        applied=True,
        input_rows=input_rows,
        output_rows=int(updated.shape[0]),
        input_columns=input_columns,
        output_columns=int(updated.shape[1]),
        trajectory_rows_read=trajectory_rows_read,
        trajectory_rows_after_id_filter=trajectory_rows_after_id_filter,
        trajectory_unique_ids=trajectory_unique_ids,
        duplicate_trajectory_ids_detected=int(duplicate_ids_table.shape[0]),
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        original_id_column_resolved=original_id_column,
        trajectory_id_column_resolved=trajectory_id_column,
        trajectory_class_column_resolved=trajectory_class_column,
        output_class_column=config.output_class_column,
        target_column_to_drop_requested=config.target_column_to_drop,
        target_column_to_drop_resolved=target_column_to_drop_resolved,
        target_column_dropped=target_column_dropped,
    )

    return Stage6Result(
        data=updated,
        summary=summary,
        tables={
            "stage6_trajectory_merge_policy": _build_policy_table(
                config=config,
                summary=summary,
            ),
            "stage6_trajectory_merge_duplicate_ids": duplicate_ids_table,
            "stage6_trajectory_merge_unmatched_ids": unmatched_ids_table,
        },
    )


@beartype
def _build_policy_table(
    *,
    config: Stage6Config,
    summary: Stage6Summary,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "enabled": "yes" if summary.enabled else "no",
                "applied": "yes" if summary.applied else "no",
                "trajectory_input_path": (
                    str(config.trajectory_input_path.resolve())
                    if config.trajectory_input_path is not None
                    else ""
                ),
                "original_id_column_requested": config.original_id_column,
                "original_id_column_resolved": summary.original_id_column_resolved,
                "trajectory_id_column_requested": config.trajectory_id_column,
                "trajectory_id_column_resolved": summary.trajectory_id_column_resolved,
                "trajectory_class_column_requested": config.trajectory_class_column,
                "trajectory_class_column_resolved": summary.trajectory_class_column_resolved,
                "output_class_column": summary.output_class_column,
                "drop_target_column_after_merge": (
                    "yes" if config.drop_target_column_after_merge else "no"
                ),
                "target_column_to_drop_requested": summary.target_column_to_drop_requested,
                "target_column_to_drop_resolved": summary.target_column_to_drop_resolved,
                "target_column_dropped": "yes" if summary.target_column_dropped else "no",
                "trajectory_rows_read": summary.trajectory_rows_read,
                "trajectory_rows_after_id_filter": summary.trajectory_rows_after_id_filter,
                "trajectory_unique_ids": summary.trajectory_unique_ids,
                "duplicate_trajectory_ids_detected": summary.duplicate_trajectory_ids_detected,
                "matched_rows": summary.matched_rows,
                "unmatched_rows": summary.unmatched_rows,
            }
        ]
    )


@beartype
def _build_duplicate_ids_table(
    *,
    data: pd.DataFrame,
    trajectory_id_column: str,
    trajectory_class_column: str,
) -> pd.DataFrame:
    grouped = (
        data.groupby(trajectory_id_column, dropna=False)[trajectory_class_column]
        .agg(
            row_count="size",
            distinct_class_count=_distinct_class_count,
        )
        .reset_index()
        .rename(columns={trajectory_id_column: "trajectory_id"})
    )
    duplicates = grouped[grouped["row_count"] > 1].copy()
    if duplicates.empty:
        return _empty_duplicate_ids_table()
    return duplicates.sort_values(by=["trajectory_id"]).reset_index(drop=True)


@beartype
def _distinct_class_count(series: pd.Series) -> int:
    normalised = series.astype("string").fillna("<NA>")
    return int(normalised.nunique(dropna=False))


@beartype
def _resolve_column_name(
    *,
    columns: pd.Index,
    requested: str,
    context: str,
) -> str:
    available = [str(column) for column in columns]
    if requested in available:
        return requested

    requested_folded = requested.casefold()
    matches = [column for column in available if column.casefold() == requested_folded]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(sorted(matches))
        raise InputValidationError(
            f"Ambiguous {context}: `{requested}` matches multiple columns: {joined}"
        )

    joined = ", ".join(sorted(available))
    raise InputValidationError(
        f"Missing {context}: `{requested}`. Available columns: {joined}"
    )


@beartype
def _ensure_output_column_is_available(*, columns: pd.Index, requested: str) -> None:
    for column in columns:
        if str(column).casefold() == requested.casefold():
            raise InputValidationError(
                "Stage 6 output class column already exists in the preset dataset: "
                f"{column}"
            )


@beartype
def _empty_duplicate_ids_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["trajectory_id", "row_count", "distinct_class_count"]
    )


@beartype
def _empty_unmatched_ids_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["original_id"])


@beartype
def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"`{context}` must be a non-empty string.")
    return value.strip()


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-6 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-6 config root must be a mapping.")
    return raw
