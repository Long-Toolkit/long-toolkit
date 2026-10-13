from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from long_toolkit.bridge.operation_registry import OperationRegistry
from long_toolkit.utils.errors import InputValidationError

from .catalog import (
    list_data_conversion_techniques,
    list_synthetic_techniques,
)


def register_operations(registry: OperationRegistry) -> None:
    """Register data-preparation operation handlers on a registry.

    Args:
        registry (OperationRegistry): Operation registry instance to configure.
    """
    registry.register(
        "data_preparation.synthetic_data_generation.catalog",
        _op_synthetic_catalog,
        description="List synthetic data-generation techniques.",
    )
    registry.register(
        "data_preparation.synthetic_data_generation.run",
        _op_synthetic_run,
        description="Run one synthetic generation technique.",
    )
    registry.register(
        "data_preparation.data_conversion.catalog",
        _op_conversion_catalog,
        description="List data conversion techniques.",
    )
    registry.register(
        "data_preparation.data_conversion.run",
        _op_conversion_run,
        description="Run one data conversion technique.",
    )


def _op_synthetic_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_synthetic_techniques()}


def _op_synthetic_run(params: Mapping[str, Any]) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return _resolve_runner(
        "long_toolkit.data_preparation.tools.synthetic_data_generation.run",
        "run_synthetic_generation",
    )(technique=technique, params=raw_params)


def _op_conversion_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_data_conversion_techniques()}


def _op_conversion_run(params: Mapping[str, Any]) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return _resolve_runner(
        "long_toolkit.data_preparation.tools.data_conversion.run",
        "run_data_conversion",
    )(technique=technique, params=raw_params)


def _resolve_runner(module_path: str, function_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def _extract_technique_and_params(
    params: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    technique = _as_required_string(params, "technique")
    raw_params = _as_object(params, "params")
    return technique, raw_params


def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


def _as_object(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key, {})
    if not isinstance(value, dict):
        raise InputValidationError(f"`{key}` must be an object.")
    return value
