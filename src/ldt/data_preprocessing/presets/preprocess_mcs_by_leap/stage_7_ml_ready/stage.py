from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.data_preprocessing.tools.missing_imputation.imputers.common import (
    count_missing_values,
    default_constant_fill_value,
    numeric_columns,
)
from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "policy.yaml"
_ALLOWED_DROP_MODES = {"auto", "manual"}
_ALLOWED_IMPUTATION_STRATEGIES = {"mean", "median", "most_frequent", "constant"}


@beartype
@dataclass(frozen=True)
class Stage7Config:
    """Runtime configuration for the optional machine-learning-ready stage."""

    enabled: bool
    drop_columns_mode: str
    default_drop_columns: tuple[str, ...]
    manual_drop_columns: tuple[str, ...]
    imputation_enabled: bool
    imputation_strategy: str


@beartype
@dataclass(frozen=True)
class Stage7Summary:
    """Compact stage 7 summary metrics."""

    enabled: bool
    applied: bool
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    drop_columns_mode: str
    requested_drop_columns_count: int
    dropped_columns_count: int
    imputation_enabled: bool
    imputation_strategy: str
    missing_before: int
    missing_after: int
    imputed_columns_count: int
    skipped_imputation_columns_count: int


@beartype
@dataclass(frozen=True)
class Stage7Result:
    """Output payload for stage 7."""

    data: pd.DataFrame
    summary: Stage7Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_7_config(
    *,
    config_path: Path | None = None,
    enabled: bool = False,
    drop_columns_mode: str = "auto",
    manual_drop_columns: tuple[str, ...] | list[str] | None = None,
    imputation_enabled: bool | None = None,
    imputation_strategy: str | None = None,
) -> Stage7Config:
    """Resolve runtime settings for the optional machine-learning-ready stage."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    drop_columns = _as_mapping(raw.get("drop_columns"), "drop_columns")
    default_drop_columns = tuple(
        _as_string_list(
            drop_columns.get("default_columns", ["CHID"]),
            "drop_columns.default_columns",
        )
    )
    if not default_drop_columns:
        raise InputValidationError(
            "`drop_columns.default_columns` must contain at least one column."
        )

    imputation = _as_mapping(raw.get("imputation"), "imputation")
    resolved_drop_mode = _as_choice(
        drop_columns_mode,
        choices=_ALLOWED_DROP_MODES,
        context="drop_columns_mode",
    )
    resolved_manual_drop_columns = tuple(_as_optional_string_list(manual_drop_columns))
    if enabled and resolved_drop_mode == "manual" and not resolved_manual_drop_columns:
        raise InputValidationError(
            "Stage 7 manual drop mode requires at least one column name."
        )
    resolved_imputation_enabled = (
        bool(imputation.get("default_enabled", False))
        if imputation_enabled is None
        else bool(imputation_enabled)
    )
    resolved_imputation_strategy = _as_choice(
        imputation_strategy or str(imputation.get("default_strategy", "mean")),
        choices=_ALLOWED_IMPUTATION_STRATEGIES,
        context="imputation.default_strategy",
    )

    return Stage7Config(
        enabled=bool(enabled),
        drop_columns_mode=resolved_drop_mode,
        default_drop_columns=default_drop_columns,
        manual_drop_columns=resolved_manual_drop_columns,
        imputation_enabled=resolved_imputation_enabled,
        imputation_strategy=resolved_imputation_strategy,
    )


@beartype
def apply_stage_7(*, data: pd.DataFrame, config: Stage7Config) -> Stage7Result:
    """Optionally remove configured columns and apply simple imputation."""

    input_rows, input_columns = int(data.shape[0]), int(data.shape[1])
    if not config.enabled:
        summary = Stage7Summary(
            enabled=False,
            applied=False,
            input_rows=input_rows,
            output_rows=input_rows,
            input_columns=input_columns,
            output_columns=input_columns,
            drop_columns_mode=config.drop_columns_mode,
            requested_drop_columns_count=0,
            dropped_columns_count=0,
            imputation_enabled=config.imputation_enabled,
            imputation_strategy=config.imputation_strategy,
            missing_before=0,
            missing_after=0,
            imputed_columns_count=0,
            skipped_imputation_columns_count=0,
        )
        return Stage7Result(
            data=data.copy(),
            summary=summary,
            tables={
                "stage7_ml_ready_policy": _build_policy_table(
                    config=config, summary=summary
                ),
                "stage7_ml_ready_dropped_columns": _empty_dropped_columns_table(),
                "stage7_ml_ready_imputation_columns": _empty_imputation_columns_table(),
            },
        )

    updated = data.copy()
    requested_drop_columns = (
        config.default_drop_columns
        if config.drop_columns_mode == "auto"
        else config.manual_drop_columns
    )
    resolved_drop_columns = _resolve_requested_columns(
        columns=updated.columns,
        requested=requested_drop_columns,
        context="stage 7 drop column",
    )
    dropped_columns_table = pd.DataFrame(
        [
            {
                "requested_column": requested,
                "resolved_column": resolved,
                "source": config.drop_columns_mode,
            }
            for requested, resolved in resolved_drop_columns
        ]
    )
    if dropped_columns_table.empty:
        dropped_columns_table = _empty_dropped_columns_table()

    if resolved_drop_columns:
        updated = updated.drop(
            columns=[resolved for _, resolved in resolved_drop_columns]
        )

    missing_before = int(count_missing_values(updated))
    imputation_columns_table = _empty_imputation_columns_table()
    imputed_columns_count = 0
    skipped_imputation_columns_count = 0
    if config.imputation_enabled:
        (
            updated,
            imputation_columns_table,
            imputed_columns_count,
            skipped_imputation_columns_count,
        ) = _apply_imputation(data=updated, strategy=config.imputation_strategy)

    summary = Stage7Summary(
        enabled=True,
        applied=True,
        input_rows=input_rows,
        output_rows=int(updated.shape[0]),
        input_columns=input_columns,
        output_columns=int(updated.shape[1]),
        drop_columns_mode=config.drop_columns_mode,
        requested_drop_columns_count=len(requested_drop_columns),
        dropped_columns_count=int(dropped_columns_table.shape[0]),
        imputation_enabled=config.imputation_enabled,
        imputation_strategy=config.imputation_strategy,
        missing_before=missing_before,
        missing_after=int(count_missing_values(updated)),
        imputed_columns_count=imputed_columns_count,
        skipped_imputation_columns_count=skipped_imputation_columns_count,
    )

    return Stage7Result(
        data=updated,
        summary=summary,
        tables={
            "stage7_ml_ready_policy": _build_policy_table(config=config, summary=summary),
            "stage7_ml_ready_dropped_columns": dropped_columns_table,
            "stage7_ml_ready_imputation_columns": imputation_columns_table,
        },
    )


@beartype
def _apply_imputation(
    *,
    data: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    updated = data.copy()
    rows: list[dict[str, object]] = []
    missing_columns = tuple(
        str(column) for column in updated.columns if updated[column].isna().any()
    )
    if not missing_columns:
        return updated, _empty_imputation_columns_table(), 0, 0

    eligible_numeric = set(numeric_columns(updated))

    for column in missing_columns:
        series = updated[column]

        if strategy == "mean":
            if column not in eligible_numeric or series.dropna().empty:
                rows.append(
                    _imputation_row(column=column, status="skipped", strategy=strategy)
                )
                continue
            updated[column] = series.fillna(series.mean(skipna=True))
            rows.append(
                _imputation_row(column=column, status="imputed", strategy=strategy)
            )
            continue

        if strategy == "median":
            if column not in eligible_numeric or series.dropna().empty:
                rows.append(
                    _imputation_row(column=column, status="skipped", strategy=strategy)
                )
                continue
            updated[column] = series.fillna(series.median(skipna=True))
            rows.append(
                _imputation_row(column=column, status="imputed", strategy=strategy)
            )
            continue

        if strategy == "most_frequent":
            if series.dropna().empty:
                rows.append(
                    _imputation_row(column=column, status="skipped", strategy=strategy)
                )
                continue
            mode = series.mode(dropna=True)
            if mode.empty:
                rows.append(
                    _imputation_row(column=column, status="skipped", strategy=strategy)
                )
                continue
            updated[column] = series.fillna(mode.iloc[0])
            rows.append(
                _imputation_row(column=column, status="imputed", strategy=strategy)
            )
            continue

        if strategy == "constant":
            updated[column] = series.fillna(default_constant_fill_value(series))
            rows.append(
                _imputation_row(column=column, status="imputed", strategy=strategy)
            )
            continue

        raise InputValidationError(f"Unsupported stage 7 imputation strategy: {strategy}")

    if not rows:
        return updated, _empty_imputation_columns_table(), 0, 0

    table = pd.DataFrame(rows).sort_values(by=["status", "column"]).reset_index(
        drop=True
    )
    imputed = int((table["status"] == "imputed").sum())
    skipped = int((table["status"] == "skipped").sum())
    return updated, table, imputed, skipped


@beartype
def _resolve_requested_columns(
    *,
    columns: pd.Index,
    requested: tuple[str, ...],
    context: str,
) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in requested:
        resolved_name = _resolve_column_name(
            columns=columns,
            requested=item,
            context=context,
        )
        folded = resolved_name.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        resolved.append((item, resolved_name))
    return tuple(resolved)


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
def _imputation_row(*, column: str, status: str, strategy: str) -> dict[str, object]:
    return {
        "column": column,
        "status": status,
        "strategy": strategy,
    }


@beartype
def _build_policy_table(*, config: Stage7Config, summary: Stage7Summary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "enabled": "yes" if summary.enabled else "no",
                "applied": "yes" if summary.applied else "no",
                "drop_columns_mode": summary.drop_columns_mode,
                "default_drop_columns": ", ".join(config.default_drop_columns),
                "manual_drop_columns": ", ".join(config.manual_drop_columns),
                "requested_drop_columns_count": summary.requested_drop_columns_count,
                "dropped_columns_count": summary.dropped_columns_count,
                "imputation_enabled": "yes" if summary.imputation_enabled else "no",
                "imputation_strategy": summary.imputation_strategy,
                "missing_before": summary.missing_before,
                "missing_after": summary.missing_after,
                "imputed_columns_count": summary.imputed_columns_count,
                "skipped_imputation_columns_count": summary.skipped_imputation_columns_count,
            }
        ]
    )


@beartype
def _empty_dropped_columns_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["requested_column", "resolved_column", "source"])


@beartype
def _empty_imputation_columns_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["column", "status", "strategy"])


@beartype
def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{context}` must be a mapping.")
    return value


@beartype
def _as_choice(value: object, *, choices: set[str], context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"`{context}` must be a non-empty string.")
    candidate = value.strip().lower()
    if candidate not in choices:
        allowed = ", ".join(sorted(choices))
        raise InputValidationError(f"`{context}` must be one of: {allowed}.")
    return candidate


@beartype
def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"`{context}` must be a non-empty string.")
    return value.strip()


@beartype
def _as_string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise InputValidationError(f"`{context}` must be a list of strings.")
    resolved: list[str] = []
    for item in value:
        resolved.append(_as_string(item, context))
    return resolved


@beartype
def _as_optional_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        resolved: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise InputValidationError("Manual drop columns must be strings.")
            stripped = item.strip()
            if stripped:
                resolved.append(stripped)
        return resolved
    raise InputValidationError(
        "Manual drop columns must be a CSV string or string list."
    )


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-7 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-7 config root must be a mapping.")
    return raw
