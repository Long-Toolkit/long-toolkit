from __future__ import annotations

import importlib
from typing import Any

from long_toolkit.utils.templates.tools.data_preprocessing import (
    DataPreprocessingTool,
    ToolParameterDefinition,
)

from .catalog import (
    list_aggregate_long_to_cross_sectional_techniques,
    list_clean_dataset_techniques,
    list_harmonise_categories_techniques,
    list_missing_imputation_techniques,
    list_pivot_long_to_wide_techniques,
    list_remove_columns_techniques,
    list_rename_feature_techniques,
    list_show_table_techniques,
    list_tool_techniques,
    resolve_technique_with_defaults,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AggregateLongToCrossSectional": (
        "long_toolkit.data_preprocessing.tools.aggregate_long_to_cross_sectional.run",
        "AggregateLongToCrossSectional",
    ),
    "AggregateLongToCrossSectionalResult": (
        "long_toolkit.data_preprocessing.tools.aggregate_long_to_cross_sectional.run",
        "AggregateLongToCrossSectionalResult",
    ),
    "CleanDataset": (
        "long_toolkit.data_preprocessing.tools.clean_dataset.run",
        "CleanDataset",
    ),
    "CleanDatasetResult": (
        "long_toolkit.data_preprocessing.tools.clean_dataset.run",
        "CleanDatasetResult",
    ),
    "HarmoniseCategories": (
        "long_toolkit.data_preprocessing.tools.harmonise_categories.run",
        "HarmoniseCategories",
    ),
    "HarmoniseCategoriesResult": (
        "long_toolkit.data_preprocessing.tools.harmonise_categories.run",
        "HarmoniseCategoriesResult",
    ),
    "ConstantImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "ConstantImputer",
    ),
    "MICEImputationResult": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MICEImputationResult",
    ),
    "MICEImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MICEImputer",
    ),
    "MeanImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MeanImputer",
    ),
    "MedianImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MedianImputer",
    ),
    "MissingImputation": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.run",
        "MissingImputation",
    ),
    "MissingImputationResult": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MissingImputationResult",
    ),
    "MissingImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MissingImputer",
    ),
    "MostFrequentImputer": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "MostFrequentImputer",
    ),
    "PivotLongToWide": (
        "long_toolkit.data_preprocessing.tools.pivot_long_to_wide.run",
        "PivotLongToWide",
    ),
    "PivotLongToWideResult": (
        "long_toolkit.data_preprocessing.tools.pivot_long_to_wide.run",
        "PivotLongToWideResult",
    ),
    "RemoveColumns": (
        "long_toolkit.data_preprocessing.tools.remove_columns.run",
        "RemoveColumns",
    ),
    "RemoveColumnsResult": (
        "long_toolkit.data_preprocessing.tools.remove_columns.run",
        "RemoveColumnsResult",
    ),
    "RenameFeature": (
        "long_toolkit.data_preprocessing.tools.rename_feature.run",
        "RenameFeature",
    ),
    "RenameFeatureResult": (
        "long_toolkit.data_preprocessing.tools.rename_feature.run",
        "RenameFeatureResult",
    ),
    "ShowTable": (
        "long_toolkit.data_preprocessing.tools.show_table.run",
        "ShowTable",
    ),
    "ShowTableResult": (
        "long_toolkit.data_preprocessing.tools.show_table.run",
        "ShowTableResult",
    ),
    "parse_columns_csv": (
        "long_toolkit.data_preprocessing.tools.remove_columns.run",
        "parse_columns_csv",
    ),
    "discover_missing_imputers": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.discovery",
        "discover_missing_imputers",
    ),
    "SimpleImputationResult": (
        "long_toolkit.data_preprocessing.tools.missing_imputation.imputers",
        "SimpleImputationResult",
    ),
}

__all__ = [
    "AggregateLongToCrossSectional",
    "AggregateLongToCrossSectionalResult",
    "CleanDataset",
    "CleanDatasetResult",
    "ConstantImputer",
    "DataPreprocessingTool",
    "HarmoniseCategories",
    "HarmoniseCategoriesResult",
    "MICEImputationResult",
    "MICEImputer",
    "MeanImputer",
    "MedianImputer",
    "MissingImputation",
    "MissingImputationResult",
    "MissingImputer",
    "MostFrequentImputer",
    "PivotLongToWide",
    "PivotLongToWideResult",
    "RemoveColumns",
    "RemoveColumnsResult",
    "RenameFeature",
    "RenameFeatureResult",
    "ShowTable",
    "ShowTableResult",
    "SimpleImputationResult",
    "ToolParameterDefinition",
    "list_aggregate_long_to_cross_sectional_techniques",
    "list_clean_dataset_techniques",
    "list_harmonise_categories_techniques",
    "list_missing_imputation_techniques",
    "list_pivot_long_to_wide_techniques",
    "list_remove_columns_techniques",
    "list_rename_feature_techniques",
    "list_show_table_techniques",
    "list_tool_techniques",
    "parse_columns_csv",
    "resolve_technique_with_defaults",
    "discover_missing_imputers",
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
