from __future__ import annotations

import importlib
from typing import Any

from long_toolkit.utils.templates.tools.machine_learning import (
    MachineLearningTool,
    ToolParameterDefinition,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BenchStandardAndLongitudinalML": (
        "long_toolkit.machine_learning.tools.bench_standard_and_longitudinal_ml.run",
        "BenchStandardAndLongitudinalML",
    ),
    "BenchmarkLongitudinalML": (
        "long_toolkit.machine_learning.tools.benchmark_longitudinal_ml.run",
        "BenchmarkLongitudinalML",
    ),
    "BenchmarkStandardML": (
        "long_toolkit.machine_learning.tools.benchmark_standard_ml.run",
        "BenchmarkStandardML",
    ),
    "LongitudinalMachineLearning": (
        "long_toolkit.machine_learning.tools.longitudinal_machine_learning.run",
        "LongitudinalMachineLearning",
    ),
    "StandardMachineLearning": (
        "long_toolkit.machine_learning.tools.standard_machine_learning.run",
        "StandardMachineLearning",
    ),
}

__all__ = [
    "MachineLearningTool",
    "ToolParameterDefinition",
    "BenchStandardAndLongitudinalML",
    "BenchmarkLongitudinalML",
    "BenchmarkStandardML",
    "LongitudinalMachineLearning",
    "StandardMachineLearning",
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
