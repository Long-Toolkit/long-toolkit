from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ldt.utils.errors import InputValidationError


def validate_one_vs_rest_evaluation_splits(
    *,
    n_splits: int,
    evaluation_strategy: str,
    holdout_split: float | None,
    scenarios: tuple[Any, ...],
) -> None:
    """Validate split configuration across one-vs-rest scenarios.

    Args:
        n_splits (int): Requested number of folds or repeated holdout splits.
        evaluation_strategy (str): Public evaluation strategy key.
        holdout_split (float | None): Optional holdout split fraction.
        scenarios (tuple[Any, ...]): One-vs-rest scenario payloads.
    """

    invalid_scenarios: list[str] = []
    for scenario in scenarios:
        try:
            validate_evaluation_split_feasibility(
                y=scenario.y_binary,
                n_splits=n_splits,
                evaluation_strategy=evaluation_strategy,
                holdout_split=holdout_split,
            )
        except InputValidationError as exc:
            invalid_scenarios.append(f"{scenario.scenario_key} ({str(exc).strip()})")
    if invalid_scenarios:
        raise InputValidationError(
            "Invalid cross-validation configuration for one-vs-rest scenarios: "
            + ", ".join(invalid_scenarios)
            + "."
        )


def validate_evaluation_split_feasibility(
    *,
    y: pd.Series,
    n_splits: int,
    evaluation_strategy: str,
    holdout_split: float | None,
) -> None:
    """Validate split feasibility for either k-fold or shuffle split.

    Args:
        y (pd.Series): Target labels to validate.
        n_splits (int): Requested number of folds or repeated holdout splits.
        evaluation_strategy (str): Public evaluation strategy key.
        holdout_split (float | None): Optional holdout split fraction.
    """

    if y.nunique(dropna=True) < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    min_class_count = int(y.value_counts(dropna=False).min())
    if evaluation_strategy == "cross_validation":
        if n_splits > min_class_count:
            raise InputValidationError(
                "Cross-validation folds exceed the smallest class count "
                f"({min_class_count}). Reduce folds or rebalance data."
            )
        return
    if evaluation_strategy != "holdout":
        raise InputValidationError(
            "evaluation_strategy must be `cross_validation` or `holdout`."
        )
    if holdout_split is None:
        raise InputValidationError(
            "holdout_split is required when evaluation_strategy is `holdout`."
        )

    class_count = int(y.nunique(dropna=True))
    sample_count = len(y)
    validation_sample_count = int(np.ceil(holdout_split * sample_count))
    train_sample_count = sample_count - validation_sample_count
    if validation_sample_count < class_count:
        raise InputValidationError(
            "Validation split is too small for stratification "
            f"(needs at least {class_count} validation samples per split)."
        )
    if train_sample_count < class_count:
        raise InputValidationError(
            "Validation split is too large for stratification "
            f"(needs at least {class_count} training samples per split)."
        )
    if min_class_count < 2:
        raise InputValidationError(
            "Holdout evaluation requires at least 2 samples in each class."
        )


def resolve_estimator_init_kwargs(
    *,
    estimator_template: Any,
    hyperparameter_overrides: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Resolve estimator constructor kwargs used by library notebook helpers.

    Args:
        estimator_template (Any): Template exposing an `estimator_cls` constructor.
        hyperparameter_overrides (Mapping[str, Any]): User-provided parameter overrides.
        random_seed (int): Random seed for reproducibility.

    Returns:
        dict[str, Any]: Constructor arguments accepted by the estimator class.
    """

    signature = inspect.signature(estimator_template.estimator_cls.__init__)
    init_kwargs: dict[str, Any] = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in hyperparameter_overrides:
            init_kwargs[parameter.name] = hyperparameter_overrides[parameter.name]
        elif parameter.name == "random_state":
            init_kwargs[parameter.name] = random_seed
    return init_kwargs


def filter_supported_hyperparameters(
    *,
    estimator_template: Any,
    hyperparameter_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter overrides down to the hyperparameters supported by a template.

    Args:
        estimator_template (Any): Template exposing `supports_hyperparameter(...)`.
        hyperparameter_overrides (Mapping[str, Any]): User-provided overrides.

    Returns:
        dict[str, Any]: Supported override mapping for this estimator only.
    """

    supports = getattr(estimator_template, "supports_hyperparameter", None)
    if not callable(supports):
        return {}

    return {
        key: value
        for key, value in hyperparameter_overrides.items()
        if supports(key)
    }


def public_estimator_module(*, estimator_template: Any) -> str:
    """Resolve a public scikit-learn import module for estimator classes.

    Args:
        estimator_template (Any): Template exposing an `estimator_cls` type.

    Returns:
        str: Public import module path.
    """

    module_name = estimator_template.estimator_cls.__module__
    if module_name.startswith("sklearn.") and "._" in module_name:
        return module_name.split("._", maxsplit=1)[0]
    return module_name


__all__ = [
    "filter_supported_hyperparameters",
    "public_estimator_module",
    "resolve_estimator_init_kwargs",
    "validate_evaluation_split_feasibility",
    "validate_one_vs_rest_evaluation_splits",
]
