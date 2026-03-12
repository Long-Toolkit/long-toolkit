from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ldt.data_preprocessing.support.inputs import (
    as_bool,
    as_choice,
    as_optional_int,
    as_optional_object,
    as_optional_string,
    as_required_int,
    as_required_string,
    normalise_key,
    run_with_validation,
)
from ldt.machine_learning.tools.metrics import resolve_metric_definition
from ldt.utils.errors import InputValidationError


def load_input_dataset(
    *,
    input_path: Path,
) -> pd.DataFrame:
    """Load one CSV dataset path with validation.

    Args:
        input_path (Path): Filesystem path used by the workflow.

    Returns:
        pd.DataFrame: Transformed dataset as a pandas DataFrame.
    """

    if not input_path.exists() or not input_path.is_file():
        raise InputValidationError(f"Input path does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a CSV file.")
    return pd.read_csv(input_path)


def resolve_target_column(
    *,
    dataset: pd.DataFrame,
    requested: str,
) -> str:
    """Resolve and validate one target column.

    Args:
        dataset (pd.DataFrame): Input dataset.
        requested (str): Number of requested items.

    Returns:
        str: Parsed string value.
    """

    target_column = requested.strip()
    if not target_column:
        raise InputValidationError("Target column is required.")
    if target_column not in dataset.columns:
        raise InputValidationError(f"Unknown target column: {target_column}")
    return target_column


def resolve_feature_columns(
    *,
    dataset: pd.DataFrame,
    target_column: str,
    raw_feature_columns: str,
) -> list[str]:
    """Resolve and validate feature columns from CSV or auto mode.

    Args:
        dataset (pd.DataFrame): Input dataset.
        target_column (str): Column name for target column.
        raw_feature_columns (str): Column names used by this workflow.

    Returns:
        list[str]: List of parsed values.
    """

    raw = raw_feature_columns.strip()
    if raw.lower() == "auto":
        selected = [column for column in dataset.columns if column != target_column]
    else:
        selected = [column.strip() for column in raw.split(",") if column.strip()]

    if not selected:
        raise InputValidationError("At least one feature column is required.")
    missing = [column for column in selected if column not in dataset.columns]
    if missing:
        raise InputValidationError(
            f"Missing feature columns in dataset: {', '.join(missing)}"
        )
    if target_column in selected:
        raise InputValidationError("Target column cannot also be a feature column.")
    return selected


def parse_metric_keys(raw_metric_keys: str) -> tuple[str, ...]:
    """Parse one or more metric keys to canonical keys.

    Args:
        raw_metric_keys (str): Raw metric keys.

    Returns:
        tuple[str, ...]: Tuple of resolved values.
    """

    parsed_keys = tuple(
        token.strip() for token in raw_metric_keys.split(",") if token.strip()
    )
    if not parsed_keys:
        raise InputValidationError("At least one metric key is required.")

    resolved_metric_keys: list[str] = []
    seen_metric_keys: set[str] = set()
    for raw_key in parsed_keys:
        try:
            metric_definition = resolve_metric_definition(raw_key)
        except KeyError as exc:
            raise InputValidationError(f"Unknown metric: {raw_key}") from exc
        if metric_definition.key in seen_metric_keys:
            continue
        seen_metric_keys.add(metric_definition.key)
        resolved_metric_keys.append(metric_definition.key)
    return tuple(resolved_metric_keys)


@dataclass(frozen=True)
class EvaluationConfig:
    """Canonical evaluation settings shared across ML workflows.

    Args:
        evaluation_strategy (str): Public evaluation strategy key.
        n_splits (int): Number of folds or repeated holdout splits.
        holdout_split (float | None): Holdout ratio, when applicable.
    """

    evaluation_strategy: str
    n_splits: int
    holdout_split: float | None


def parse_holdout_split(raw_value: str) -> float:
    """Parse a holdout split string.

    Args:
        raw_value (str): Raw value.

    Returns:
        float: Parsed floating-point value.
    """

    raw = raw_value.strip()
    if not raw:
        raise InputValidationError(
            "holdout_split is required when evaluation_strategy is `holdout`."
        )

    if "/" in raw:
        parts = [part.strip() for part in raw.split("/", maxsplit=1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise InputValidationError("Split ratio must look like `70/30` or `80/20`.")
        train_share = _parse_split_number(parts[0])
        validation_share = _parse_split_number(parts[1])
        if train_share <= 0 or validation_share <= 0:
            raise InputValidationError("Split ratio values must be positive.")
        return validation_share / (train_share + validation_share)

    try:
        parsed = float(raw)
    except ValueError as exc:
        raise InputValidationError(
            "holdout_split must be a ratio like `70/30` "
            "or a float in (0, 1)."
        ) from exc
    if parsed > 1.0:
        parsed = parsed / 100.0
    if not 0.0 < parsed < 1.0:
        raise InputValidationError(
            "Validation split must be a float strictly between 0 and 1."
        )
    return parsed


def resolve_evaluation_config(
    *,
    evaluation_strategy_raw: str,
    n_splits: int,
    holdout_split_raw: str | None,
) -> EvaluationConfig:
    """Resolve public evaluation inputs into a canonical configuration.

    Args:
        evaluation_strategy_raw (str): Raw strategy token.
        n_splits (int): Number of folds or repeated holdout splits.
        holdout_split_raw (str | None): Raw holdout split string.

    Returns:
        EvaluationConfig: Canonical evaluation settings.
    """

    strategy = evaluation_strategy_raw.strip().lower().replace("-", "_")
    if strategy not in {"cross_validation", "holdout"}:
        raise InputValidationError(
            "evaluation_strategy must be `cross_validation` or `holdout`."
        )
    if n_splits < 2:
        raise InputValidationError("n_splits must be >= 2.")

    if strategy == "cross_validation":
        return EvaluationConfig(
            evaluation_strategy="cross_validation",
            n_splits=n_splits,
            holdout_split=None,
        )

    return EvaluationConfig(
        evaluation_strategy="holdout",
        n_splits=n_splits,
        holdout_split=parse_holdout_split(holdout_split_raw or ""),
    )


def parse_excluded_estimators(raw_value: str) -> set[str]:
    """Parse excluded-estimator tokens from CSV string.

    Args:
        raw_value (str): Raw value.

    Returns:
        set[str]: Set of parsed values.
    """

    token = raw_value.strip().lower()
    if token in {"", "none", "no", "n"}:
        return set()
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def parse_class_index(raw_value: str) -> int | None:
    """Parse class-index input for SHAP analysis.

    Args:
        raw_value (str): Raw value.

    Returns:
        int | None: Parsed integer value.
    """

    token = raw_value.strip().lower()
    if token in {"", "auto", "none"}:
        return None
    try:
        parsed = int(token)
    except ValueError as exc:
        raise InputValidationError("class_index must be an integer or `auto`.") from exc
    if parsed < 0:
        raise InputValidationError("class_index must be >= 0.")
    return parsed


def resolve_notebook_path(
    *,
    generate_notebook: bool,
    requested_path: str | None,
    default_path: Path,
) -> Path | None:
    """Resolve optional notebook output path.

    Args:
        generate_notebook (bool): Boolean option that controls behaviour.
        requested_path (str | None): Filesystem path used by the workflow.
        default_path (Path): Filesystem path used by the workflow.

    Returns:
        Path | None: Resolved filesystem path.
    """

    if not generate_notebook:
        return None
    if requested_path is None or not requested_path.strip():
        return default_path
    candidate = Path(requested_path).expanduser()
    if candidate.suffix.lower() != ".ipynb":
        raise InputValidationError("Notebook output path must end with `.ipynb`.")
    return candidate


def _parse_split_number(raw_value: str) -> float:
    token = raw_value.strip()
    if token.endswith("%"):
        token = token[:-1].strip()
    try:
        return float(token)
    except ValueError as exc:
        raise InputValidationError(f"Invalid split value: {raw_value}") from exc


__all__ = [
    "as_bool",
    "as_choice",
    "as_optional_int",
    "as_optional_object",
    "as_optional_string",
    "as_required_int",
    "as_required_string",
    "load_input_dataset",
    "normalise_key",
    "parse_class_index",
    "parse_excluded_estimators",
    "parse_holdout_split",
    "parse_metric_keys",
    "resolve_evaluation_config",
    "resolve_feature_columns",
    "resolve_notebook_path",
    "resolve_target_column",
    "run_with_validation",
]
