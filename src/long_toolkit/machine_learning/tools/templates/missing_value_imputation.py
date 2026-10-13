from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from scikit_longitudinal.pipeline import LongitudinalPipeline
from sklearn.base import BaseEstimator, TransformerMixin

from long_toolkit.data_preprocessing.tools.missing_imputation.imputers.common import (
    default_constant_fill_value,
)
from long_toolkit.utils.errors import InputValidationError

ALLOWED_MISSING_VALUE_IMPUTATION_STRATEGIES = (
    "mean",
    "median",
    "most_frequent",
    "constant",
)
_SKIP_FILL_VALUE = object()


@beartype
@dataclass(frozen=True)
class MissingValueImputationConfig:
    """Training-time missing-value imputation settings."""

    enabled: bool
    strategy: str


@beartype
def validate_missing_value_imputation_strategy(*, strategy: str) -> str:
    """Validate and normalise a missing-value imputation strategy key."""

    candidate = strategy.strip().lower()
    if candidate not in ALLOWED_MISSING_VALUE_IMPUTATION_STRATEGIES:
        allowed = ", ".join(ALLOWED_MISSING_VALUE_IMPUTATION_STRATEGIES)
        raise InputValidationError(
            "missing_value_imputation_strategy must be one of: "
            f"{allowed}."
        )
    return candidate


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Column-wise imputer that preserves feature count and DataFrame columns."""

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy

    @beartype
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> MissingValueImputer:
        _ = y
        frame = self._coerce_to_frame(X=X)
        strategy = validate_missing_value_imputation_strategy(strategy=self.strategy)

        self.selected_features_ = np.arange(frame.shape[1], dtype=int)
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        self.fill_values_: dict[str, object] = {}
        self.skipped_columns_: tuple[str, ...] = ()

        skipped_columns: list[str] = []
        for column in self.feature_names_in_:
            series = frame[column]
            resolved = self._resolve_fill_value(series=series, strategy=strategy)
            if resolved is _SKIP_FILL_VALUE:
                skipped_columns.append(column)
                continue
            self.fill_values_[column] = resolved

        self.skipped_columns_ = tuple(skipped_columns)
        return self

    @beartype
    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        frame = self._coerce_to_frame(
            X=X,
            expected_columns=getattr(self, "feature_names_in_", None),
        )
        updated = frame.copy()
        for column, fill_value in getattr(self, "fill_values_", {}).items():
            updated[column] = updated[column].fillna(fill_value)
        return updated

    @staticmethod
    @beartype
    def _coerce_to_frame(
        *,
        X: pd.DataFrame | np.ndarray,
        expected_columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            if (
                expected_columns is not None
                and tuple(map(str, frame.columns)) != expected_columns
            ):
                raise InputValidationError(
                    "Input feature columns do not match the fitted imputation pipeline."
                )
            frame.columns = [str(column) for column in frame.columns]
            return frame

        if not isinstance(X, np.ndarray):
            raise InputValidationError(
                "Missing-value imputation expects a pandas DataFrame or numpy array."
            )
        if X.ndim != 2:
            raise InputValidationError("Input features must be two-dimensional.")

        if expected_columns is not None:
            columns = list(expected_columns)
        else:
            columns = [f"feature_{index}" for index in range(X.shape[1])]
        if len(columns) != X.shape[1]:
            raise InputValidationError(
                "Input feature count does not match the fitted imputation pipeline."
            )
        return pd.DataFrame(X, columns=columns)

    @staticmethod
    @beartype
    def _resolve_fill_value(*, series: pd.Series, strategy: str) -> object:
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
                return _SKIP_FILL_VALUE
            return float(series.mean(skipna=True))

        if strategy == "median":
            if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
                return _SKIP_FILL_VALUE
            return float(series.median(skipna=True))

        if strategy == "most_frequent":
            if series.dropna().empty:
                return _SKIP_FILL_VALUE
            mode = series.mode(dropna=True)
            if mode.empty:
                return _SKIP_FILL_VALUE
            return mode.iloc[0]

        if strategy == "constant":
            return default_constant_fill_value(series)

        raise InputValidationError(
            f"Unsupported missing-value imputation strategy: {strategy}"
        )


class ImputedLongitudinalPipeline(LongitudinalPipeline):
    """Longitudinal pipeline wrapper that applies fitted imputation at inference."""

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        features_group: list[list[int]],
        non_longitudinal_features: list[int | str] | None = None,
        update_feature_groups_callback: Any = None,
        feature_list_names: list[str] | None = None,
        missing_value_imputation_strategy: str = "mean",
    ) -> None:
        super().__init__(
            steps=steps,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            update_feature_groups_callback=update_feature_groups_callback,
            feature_list_names=feature_list_names,
        )
        self.missing_value_imputation_strategy = missing_value_imputation_strategy

    @beartype
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: dict[str, Any],
    ) -> ImputedLongitudinalPipeline:
        self.missing_value_imputer_ = MissingValueImputer(
            strategy=self.missing_value_imputation_strategy
        )
        transformed_X = self.missing_value_imputer_.fit_transform(X)
        return super().fit(transformed_X, y, **fit_params)

    @beartype
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        **predict_params: dict[str, Any],
    ) -> np.ndarray:
        return super().predict(
            self._transform_with_imputer(X=X),
            **predict_params,
        )

    @beartype
    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        **predict_params: dict[str, Any],
    ) -> np.ndarray:
        return super().predict_proba(
            self._transform_with_imputer(X=X),
            **predict_params,
        )

    @beartype
    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        **transform_params: dict[str, Any],
    ) -> np.ndarray:
        return super().transform(
            self._transform_with_imputer(X=X),
            **transform_params,
        )

    @beartype
    def decision_function(
        self,
        X: pd.DataFrame | np.ndarray,
        **params: dict[str, Any],
    ) -> np.ndarray:
        return super().decision_function(
            self._transform_with_imputer(X=X),
            **params,
        )

    @beartype
    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **score_params: dict[str, Any],
    ) -> float:
        return float(
            super().score(
                self._transform_with_imputer(X=X),
                y,
                **score_params,
            )
        )

    @beartype
    def _transform_with_imputer(
        self,
        *,
        X: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        imputer = getattr(self, "missing_value_imputer_", None)
        if imputer is None:
            return X
        return imputer.transform(X)
