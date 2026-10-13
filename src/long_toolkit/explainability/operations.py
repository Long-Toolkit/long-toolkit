from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from long_toolkit.bridge.operation_registry import OperationRegistry
from long_toolkit.utils.errors import InputValidationError

from .catalog import list_shap_analysis_techniques


def register_operations(registry: OperationRegistry) -> None:
    """Register explainability operation handlers on a registry.

    Args:
        registry (OperationRegistry): Operation registry instance to configure.
    """
    registry.register(
        "explainability.shap_analysis.catalog",
        _op_shap_catalog,
        description="List SHAP-analysis techniques.",
    )
    registry.register(
        "explainability.shap_analysis.run",
        _op_shap_run,
        description="Run one SHAP-analysis technique.",
    )


def _op_shap_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_shap_analysis_techniques()}


def _op_shap_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "long_toolkit.explainability.tools.shap_analysis.run",
            "run_shap_analysis_tool",
        ),
    )


def _run_tool_operation(*, params: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    technique = _as_required_string(params, "technique")
    raw_params = _as_object(params, "params")
    return runner(technique=technique, params=raw_params)


def _resolve_runner(module_path: str, function_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


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
