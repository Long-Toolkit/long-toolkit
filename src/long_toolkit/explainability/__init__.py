from __future__ import annotations

import importlib
from typing import Any

from long_toolkit.utils.templates.tools.explainability import (
    ExplainabilityTool,
    ToolParameterDefinition,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SHAPAnalysis": (
        "long_toolkit.explainability.tools.shap_analysis.run",
        "SHAPAnalysis",
    ),
}

__all__ = [
    "ExplainabilityTool",
    "ToolParameterDefinition",
    "SHAPAnalysis",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, symbol_name = target
    module = importlib.import_module(module_path)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
