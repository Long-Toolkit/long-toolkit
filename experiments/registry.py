from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from long_toolkit.machine_learning.tools.longitudinal_machine_learning.discovery import (
    discover_longitudinal_estimators,
)
from long_toolkit.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from long_toolkit.utils.metadata import resolve_component_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

DEFAULT_DATASET_RELATIVE_PATH = Path(
    "data/processed/MCS/mcs_longitudinal_wide_preprocessed.csv"
)
DEFAULT_DATASET_PATH = PROJECT_ROOT / DEFAULT_DATASET_RELATIVE_PATH
DEFAULT_SPLITS_ROOT = EXPERIMENTS_ROOT / "splits"
DEFAULT_SPLIT_MANIFEST_PATH = DEFAULT_SPLITS_ROOT / "mcs_holdout_cv_80_20_seed42.json"
DEFAULT_RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
DEFAULT_REPORTS_ROOT = EXPERIMENTS_ROOT / "reports"
DEFAULT_LAUNCHERS_ROOT = EXPERIMENTS_ROOT / "launchers"
DEFAULT_SCENARIOS_ROOT = EXPERIMENTS_ROOT / "scenarios"
DEFAULT_LOGS_ROOT = EXPERIMENTS_ROOT / "logs"

TARGET_COLUMN = "class"
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 10
TRAIN_IMPUTATION_STRATEGY = "mean"
NUMERIC_SCALER_RANGE = (0.0, 1.0)
DEFAULT_RANKING_METRIC = "auprc"
DEFAULT_TOP_K = 3
DEFAULT_SHAP_BACKGROUND_SIZE = 250
DEFAULT_SHAP_ROWS_PER_CLASS = 250
DEFAULT_SHAP_MAX_DISPLAY = 20
DEFAULT_SHAP_TARGET_CLASS = 4

BAGGING_HYPERPARAMETERS = {
    "n_estimators": 15,
    "sampling_strategy": "all",
    "replacement": True,
    "bootstrap": True,
    "max_samples": 0.7529930396701624,
}

TUNED_GB_HYPERPARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.048736757615316824,
    "n_estimators": 350,
    "subsample": 0.7923998099457771,
    "max_depth": 4,
}

BASE_OPERATIONAL_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "svm": {"probability": True},
    "logistic_regression": {"max_iter": 1000},
    "knn": {"n_jobs": 1},
    "random_forest": {"n_jobs": 1},
    "extra_trees": {"n_jobs": 1},
}

SCOPE_ORDER = {
    "standard_ml": 0,
    "longitudinal_ml": 1,
    "balanced_bagging_standard_ml": 2,
    "balanced_bagging_longitudinal_ml": 3,
}

SCOPE_SHORT_NAMES = {
    "standard_ml": "std",
    "longitudinal_ml": "long",
    "balanced_bagging_standard_ml": "bbstd",
    "balanced_bagging_longitudinal_ml": "bblong",
}


@dataclass(frozen=True)
class PrimitiveSpec:
    launcher_key: str
    scope_key: str
    estimator_domain: str
    primitive_key: str
    display_name: str
    balanced_bagging: bool
    desired_hyperparameters: dict[str, Any]

    @property
    def launcher_filename(self) -> str:
        return f"{self.launcher_key}.py"

    @property
    def fold_output_root(self) -> Path:
        return DEFAULT_RESULTS_ROOT / self.scope_key / self.primitive_key

    @property
    def final_output_dir(self) -> Path:
        return self.fold_output_root / "final_model"

    @property
    def scenario_dir(self) -> Path:
        return DEFAULT_SCENARIOS_ROOT / self.scope_key / self.primitive_key

    @property
    def logs_dir(self) -> Path:
        return DEFAULT_LOGS_ROOT / self.scope_key / self.primitive_key

    @property
    def launcher_path(self) -> Path:
        return DEFAULT_LAUNCHERS_ROOT / self.launcher_filename


def _resolve_desired_hyperparameters(primitive_key: str) -> dict[str, Any]:
    desired: dict[str, Any] = {}

    if "gradient_boosting" in primitive_key:
        desired.update(TUNED_GB_HYPERPARAMETERS)

    for suffix, overrides in BASE_OPERATIONAL_HYPERPARAMETERS.items():
        if primitive_key == suffix or primitive_key.endswith(f"__{suffix}"):
            desired.update(overrides)
    return desired


def _build_standard_specs() -> list[PrimitiveSpec]:
    estimators = discover_standard_estimators()
    specs: list[PrimitiveSpec] = []
    for primitive_key, template_cls in estimators.items():
        display_name = resolve_component_metadata(template_cls).full_name
        desired_hyperparameters = _resolve_desired_hyperparameters(primitive_key)
        specs.append(
            PrimitiveSpec(
                launcher_key=f"standard_ml__{primitive_key}",
                scope_key="standard_ml",
                estimator_domain="standard",
                primitive_key=primitive_key,
                display_name=display_name,
                balanced_bagging=False,
                desired_hyperparameters=dict(desired_hyperparameters),
            )
        )
        specs.append(
            PrimitiveSpec(
                launcher_key=f"balanced_bagging_standard_ml__{primitive_key}",
                scope_key="balanced_bagging_standard_ml",
                estimator_domain="standard",
                primitive_key=primitive_key,
                display_name=f"Balanced Bagging + {display_name}",
                balanced_bagging=True,
                desired_hyperparameters=dict(desired_hyperparameters),
            )
        )
    return specs


def _build_longitudinal_specs() -> list[PrimitiveSpec]:
    estimators = discover_longitudinal_estimators()
    specs: list[PrimitiveSpec] = []
    for primitive_key, template_cls in estimators.items():
        display_name = resolve_component_metadata(template_cls).full_name
        desired_hyperparameters = _resolve_desired_hyperparameters(primitive_key)
        specs.append(
            PrimitiveSpec(
                launcher_key=f"longitudinal_ml__{primitive_key}",
                scope_key="longitudinal_ml",
                estimator_domain="longitudinal",
                primitive_key=primitive_key,
                display_name=display_name,
                balanced_bagging=False,
                desired_hyperparameters=dict(desired_hyperparameters),
            )
        )
        specs.append(
            PrimitiveSpec(
                launcher_key=f"balanced_bagging_longitudinal_ml__{primitive_key}",
                scope_key="balanced_bagging_longitudinal_ml",
                estimator_domain="longitudinal",
                primitive_key=primitive_key,
                display_name=f"Balanced Bagging + {display_name}",
                balanced_bagging=True,
                desired_hyperparameters=dict(desired_hyperparameters),
            )
        )
    return specs


@lru_cache(maxsize=1)
def get_all_primitive_specs() -> tuple[PrimitiveSpec, ...]:
    specs = _build_standard_specs() + _build_longitudinal_specs()
    return tuple(
        sorted(
            specs,
            key=lambda spec: (
                SCOPE_ORDER[spec.scope_key],
                spec.display_name.lower(),
                spec.launcher_key,
            ),
        )
    )


@lru_cache(maxsize=1)
def get_primitive_spec_map() -> dict[str, PrimitiveSpec]:
    return {spec.launcher_key: spec for spec in get_all_primitive_specs()}


def get_primitive_spec(launcher_key: str) -> PrimitiveSpec:
    spec = get_primitive_spec_map().get(launcher_key)
    if spec is None:
        raise KeyError(f"Unknown launcher key: {launcher_key}")
    return spec


def iter_primitive_specs(scope_keys: set[str] | None = None) -> tuple[PrimitiveSpec, ...]:
    specs = get_all_primitive_specs()
    if not scope_keys:
        return specs
    return tuple(spec for spec in specs if spec.scope_key in scope_keys)
