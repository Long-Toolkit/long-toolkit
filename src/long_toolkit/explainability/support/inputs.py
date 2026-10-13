from __future__ import annotations

from long_toolkit.data_preprocessing.support.inputs import (
    as_optional_string,
    as_required_int,
    as_required_string,
    run_with_validation,
)
from long_toolkit.utils.errors import InputValidationError

__all__ = [
    "as_optional_string",
    "as_required_int",
    "as_required_string",
    "parse_class_index",
    "run_with_validation",
]


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
