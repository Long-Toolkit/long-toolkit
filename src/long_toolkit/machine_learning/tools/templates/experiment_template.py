from __future__ import annotations

import copy
import json
import pickle
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from scikit_longitudinal.pipeline import LongitudinalPipeline
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from long_toolkit.machine_learning.tools.metrics import (
    MetricCompatibilityError,
    MetricDefinition,
    resolve_metric_definition,
)
from long_toolkit.machine_learning.tools.templates.missing_value_imputation import (
    ImputedLongitudinalPipeline,
    MissingValueImputationConfig,
    MissingValueImputer,
    validate_missing_value_imputation_strategy,
)
from long_toolkit.utils.errors import InputValidationError


@beartype
@dataclass(frozen=True)
class ClassificationMetricSummary:
    """Per-metric cross-validation summary.

    Attributes:
        metric_key (str): Identifier for metric key.
        fold_scores (tuple[float, ...]): Fold scores.
        mean_score (float): Mean score.
        std_score (float): Std score.

    """

    metric_key: str
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float


@beartype
@dataclass(frozen=True)
class ClassificationExperimentArtifacts:
    """Artefact paths produced by one classification experiment.

    Attributes:
        model_path (Path): Path for model path.
        summary_path (Path): Path for summary path.
        report_path (Path): Path for report path.

    """

    model_path: Path
    summary_path: Path
    report_path: Path


@beartype
@dataclass(frozen=True)
class ClassificationExperimentResult:
    """Structured output produced by the classification experiment template.

    Attributes:
        estimator_name (str): Name for estimator.
        metric_key (str): Identifier for metric key.
        metric_keys (tuple[str, ...]): Metric keys.
        random_seed (int): Random seed for reproducibility.
        n_splits (int): Number of folds or repeated holdout splits.
        evaluation_strategy (str): Public evaluation strategy key.
        holdout_split (float | None): Holdout ratio, when applicable.
        split_strategy (str): Internal sklearn splitter key.
        silent_training_output (bool): Whether to silent training output.
        fold_scores (tuple[float, ...]): Fold scores.
        mean_score (float): Mean score.
        std_score (float): Std score.
        metric_summaries (dict[str, ClassificationMetricSummary]): Metric summaries.
        classification_report_text (str): Classification report text.
        classification_report_dict (dict[str, Any]): Classification report dict.
        fitted_estimator (BaseEstimator): Fitted estimator.
        artifacts (ClassificationExperimentArtifacts): Artefacts.

    """

    estimator_name: str
    metric_key: str
    metric_keys: tuple[str, ...]
    random_seed: int
    n_splits: int
    evaluation_strategy: str
    holdout_split: float | None
    split_strategy: str
    silent_training_output: bool
    missing_value_imputation_enabled: bool
    missing_value_imputation_strategy: str
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float
    metric_summaries: dict[str, ClassificationMetricSummary]
    classification_report_text: str
    classification_report_dict: dict[str, Any]
    fitted_estimator: BaseEstimator
    artifacts: ClassificationExperimentArtifacts


class EstimatorMetricCompatibilityError(InputValidationError):
    """Raised when an estimator cannot satisfy metric output requirements."""


@beartype
class ClassificationExperimentTemplate:
    """Reusable template for cross-validated classification experiments."""

    @beartype
    def run(
        self,
        *,
        estimator: BaseEstimator,
        estimator_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        metric_key: str | None = None,
        metric_keys: tuple[str, ...] | None = None,
        output_dir: Path,
        n_splits: int = 10,
        evaluation_strategy: str = "cross_validation",
        holdout_split: float | None = None,
        random_seed: int | None = None,
        silent_training_output: bool = False,
        missing_value_imputation_enabled: bool = False,
        missing_value_imputation_strategy: str = "mean",
        show_progress: bool = True,
    ) -> ClassificationExperimentResult:
        """Execute a classification experiment.

        Args:
            estimator (BaseEstimator): Estimator.
            estimator_name (str): Name for estimator.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            metric_key (str | None): Identifier for metric key.
            metric_keys (tuple[str, ...] | None): Metric keys.
            output_dir (Path): Filesystem location for output dir.
            n_splits (int): Number of folds or repeated holdout splits.
            evaluation_strategy (str): Public evaluation strategy key.
            holdout_split (float | None): Holdout ratio, when applicable.
            random_seed (int | None): Random seed for reproducibility.
            silent_training_output (bool): Boolean option that controls behaviour.
            missing_value_imputation_enabled (bool): Whether to fit missing-value
                imputation inside the model pipeline.
            missing_value_imputation_strategy (str): Strategy used when
                `missing_value_imputation_enabled=True`.
            show_progress (bool): Whether to render progress output.

        Returns:
            ClassificationExperimentResult: Result object for this operation.
        """

        if n_splits < 2:
            raise InputValidationError("n_splits must be >= 2.")
        if evaluation_strategy == "cross_validation" and len(y) < n_splits:
            raise InputValidationError(
                "Cross-validation folds cannot exceed the number of samples."
            )
        if y.nunique(dropna=True) < 2:
            raise InputValidationError(
                "Classification requires at least two target classes."
            )
        min_class_count = int(y.value_counts(dropna=False).min())

        seed = random_seed
        if seed is None:
            seed = int(np.random.randint(0, 2_147_483_647))

        (
            cv,
            resolved_holdout_split,
            split_strategy,
        ) = self._build_splitter(
            y=y,
            n_splits=n_splits,
            evaluation_strategy=evaluation_strategy,
            holdout_split=holdout_split,
            min_class_count=min_class_count,
            random_seed=seed,
        )
        resolved_metrics = self._resolve_metrics(
            metric_key=metric_key,
            metric_keys=metric_keys,
        )
        resolved_metric_keys = tuple(metric.key for metric in resolved_metrics)
        primary_metric_key = resolved_metric_keys[0]
        imputation_config = MissingValueImputationConfig(
            enabled=bool(missing_value_imputation_enabled),
            strategy=validate_missing_value_imputation_strategy(
                strategy=missing_value_imputation_strategy
            ),
        )
        model = self._ensure_model_pipeline(
            estimator=estimator,
            X=X,
            missing_value_imputation=imputation_config,
        )
        fold_scores_by_metric: dict[str, list[float]] = {
            metric.key: [] for metric in resolved_metrics
        }
        validation_y_true: list[np.ndarray] = []
        validation_y_pred: list[np.ndarray] = []

        def _run_with_progress(
            *,
            progress: Progress | None = None,
            task_id: object | None = None,
        ) -> ClassificationExperimentResult:
            try:
                for fold_index, (train_indices, test_indices) in enumerate(
                    cv.split(X, y),
                    start=1,
                ):
                    fold_model = self._clone_model(model=model)
                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]
                    with self._silence_training_output(enabled=silent_training_output):
                        fold_model.fit(X_train, y_train)
                    for metric in resolved_metrics:
                        try:
                            with self._silence_training_output(
                                enabled=silent_training_output
                            ):
                                fold_score = float(
                                    metric.scorer(fold_model, X_test, y_test)
                                )
                        except MetricCompatibilityError as exc:
                            raise EstimatorMetricCompatibilityError(
                                "Estimator/metric compatibility failed: "
                                f"`{estimator_name}` does not support "
                                f"`{metric.key}`. {exc}"
                            ) from exc
                        fold_scores_by_metric[metric.key].append(fold_score)
                    with self._silence_training_output(enabled=silent_training_output):
                        fold_predictions = np.asarray(fold_model.predict(X_test))
                    validation_y_true.append(np.asarray(y_test))
                    validation_y_pred.append(fold_predictions)
                    if progress is not None and task_id is not None:
                        progress.update(
                            task_id,
                            advance=1,
                            description=(
                                "Running cross-validation folds... "
                                f"({fold_index}/{n_splits})"
                            ),
                        )
            except ValueError as exc:
                raise InputValidationError(
                    "Cross-validation failed for the selected metric/estimator "
                    f"combination: {exc}"
                ) from exc

            metric_summaries = {
                metric.key: ClassificationMetricSummary(
                    metric_key=metric.key,
                    fold_scores=tuple(fold_scores_by_metric[metric.key]),
                    mean_score=float(np.nanmean(fold_scores_by_metric[metric.key])),
                    std_score=float(np.nanstd(fold_scores_by_metric[metric.key])),
                )
                for metric in resolved_metrics
            }
            primary_metric_summary = metric_summaries[primary_metric_key]
            if not validation_y_true or not validation_y_pred:
                raise InputValidationError(
                    "Cross-validation did not produce validation predictions."
                )
            y_true_evaluation = np.concatenate(validation_y_true)
            y_pred_evaluation = np.concatenate(validation_y_pred)
            report_text = classification_report(
                y_true_evaluation,
                y_pred_evaluation,
                zero_division=0,
            )
            report_dict = classification_report(
                y_true_evaluation,
                y_pred_evaluation,
                zero_division=0,
                output_dict=True,
            )
            if progress is not None and task_id is not None:
                progress.update(task_id, description="Training final model...")
            with self._silence_training_output(enabled=silent_training_output):
                fitted_estimator = self._clone_model(model=model).fit(X, y)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1, description="Writing artefacts...")
            artifacts = self._write_artifacts(
                estimator_name=estimator_name,
                metric_key=primary_metric_key,
                metric_keys=resolved_metric_keys,
                output_dir=output_dir,
                n_splits=n_splits,
                evaluation_strategy=evaluation_strategy,
                random_seed=seed,
                holdout_split=resolved_holdout_split,
                split_strategy=split_strategy,
                silent_training_output=silent_training_output,
                missing_value_imputation_enabled=imputation_config.enabled,
                missing_value_imputation_strategy=imputation_config.strategy,
                fold_scores=primary_metric_summary.fold_scores,
                metric_summaries=metric_summaries,
                report_text=report_text,
                fitted_estimator=fitted_estimator,
                class_distribution=y.value_counts(dropna=False).to_dict(),
            )
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1, description="Done")

            return ClassificationExperimentResult(
                estimator_name=estimator_name,
                metric_key=primary_metric_key,
                metric_keys=resolved_metric_keys,
                random_seed=seed,
                n_splits=n_splits,
                evaluation_strategy=evaluation_strategy,
                holdout_split=resolved_holdout_split,
                split_strategy=split_strategy,
                silent_training_output=silent_training_output,
                missing_value_imputation_enabled=imputation_config.enabled,
                missing_value_imputation_strategy=imputation_config.strategy,
                fold_scores=primary_metric_summary.fold_scores,
                mean_score=primary_metric_summary.mean_score,
                std_score=primary_metric_summary.std_score,
                metric_summaries=metric_summaries,
                classification_report_text=report_text,
                classification_report_dict=report_dict,
                fitted_estimator=fitted_estimator,
                artifacts=artifacts,
            )

        if not show_progress:
            return _run_with_progress()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Running cross-validation folds...",
                total=n_splits + 2,
            )
            return _run_with_progress(progress=progress, task_id=task_id)

    @staticmethod
    @beartype
    def _resolve_metrics(
        *,
        metric_key: str | None,
        metric_keys: tuple[str, ...] | None,
    ) -> tuple[MetricDefinition, ...]:
        """Resolve and validate one or more metric definitions."""

        if metric_key is not None and metric_keys is not None:
            raise InputValidationError(
                "Provide either `metric_key` or `metric_keys`, not both."
            )

        raw_metric_keys: tuple[str, ...]
        if metric_keys is not None:
            raw_metric_keys = tuple(metric.strip() for metric in metric_keys)
        elif metric_key is not None:
            raw_metric_keys = tuple(metric.strip() for metric in metric_key.split(","))
        else:
            raise InputValidationError(
                "At least one metric is required (metric_key or metric_keys)."
            )

        filtered_metric_keys = tuple(metric for metric in raw_metric_keys if metric)
        if not filtered_metric_keys:
            raise InputValidationError("At least one metric key is required.")

        resolved = []
        seen_metric_keys: set[str] = set()
        for raw_key in filtered_metric_keys:
            try:
                metric_definition = resolve_metric_definition(raw_key)
            except KeyError as exc:
                raise InputValidationError(f"Unknown metric: {raw_key}") from exc
            if metric_definition.key in seen_metric_keys:
                continue
            seen_metric_keys.add(metric_definition.key)
            resolved.append(metric_definition)
        return tuple(resolved)

    @staticmethod
    @beartype
    def _clone_model(*, model: BaseEstimator) -> BaseEstimator:
        """Clone estimator, with deepcopy fallback for non-cloneable models."""

        try:
            return clone(model)
        except (RuntimeError, TypeError, ValueError):
            pass

        try:
            return copy.deepcopy(model)
        except Exception as exc:
            raise InputValidationError(
                "Unable to duplicate estimator for cross-validation. "
                "This estimator is not sklearn-cloneable and deepcopy also failed: "
                f"{exc}"
            ) from exc

    @staticmethod
    @beartype
    def _build_splitter(
        *,
        y: pd.Series,
        n_splits: int,
        evaluation_strategy: str,
        holdout_split: float | None,
        min_class_count: int,
        random_seed: int,
    ) -> tuple[StratifiedKFold | StratifiedShuffleSplit, float | None, str]:
        """Build and validate the CV splitter used by the experiment run."""

        if evaluation_strategy == "cross_validation":
            if n_splits > min_class_count:
                raise InputValidationError(
                    "Cross-validation folds exceed the smallest class count "
                    f"({min_class_count})."
                )
            return (
                StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_seed,
                ),
                None,
                "stratified_kfold",
            )

        if evaluation_strategy != "holdout":
            raise InputValidationError(
                "evaluation_strategy must be `cross_validation` or `holdout`."
            )
        if holdout_split is None:
            raise InputValidationError(
                "holdout_split is required when evaluation_strategy is `holdout`."
            )
        if not 0.0 < holdout_split < 1.0:
            raise InputValidationError(
                "holdout_split must be a float strictly between 0 and 1."
            )

        class_count = int(y.nunique(dropna=True))
        sample_count = len(y)
        validation_sample_count = int(np.ceil(holdout_split * sample_count))
        train_sample_count = sample_count - validation_sample_count
        if validation_sample_count < class_count:
            raise InputValidationError(
                "Holdout split is too small for stratification: validation "
                "samples per split must be at least the number of classes "
                f"({class_count})."
            )
        if train_sample_count < class_count:
            raise InputValidationError(
                "Holdout split is too large for stratification: training "
                "samples per split must be at least the number of classes "
                f"({class_count})."
            )
        if min_class_count < 2:
            raise InputValidationError(
                "Holdout evaluation requires at least 2 samples in each class."
            )

        return (
            StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=holdout_split,
                random_state=random_seed,
            ),
            float(holdout_split),
            "stratified_shuffle_split",
        )

    @staticmethod
    @contextmanager
    @beartype
    def _silence_training_output(
        *,
        enabled: bool,
    ) -> Any:
        """Optionally silence estimator stdout/stderr during training operations."""

        if not enabled:
            yield
            return

        with StringIO() as captured_stdout, StringIO() as captured_stderr:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                yield

    @staticmethod
    @beartype
    def _ensure_model_pipeline(
        *,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        missing_value_imputation: MissingValueImputationConfig,
    ) -> BaseEstimator:
        """Wrap estimators with the appropriate preprocessing pipeline.

        Args:
            estimator: User-selected estimator.
            X: Raw feature matrix.

        Returns:
            BaseEstimator: Estimator (possibly wrapped in a pipeline).
        """

        if isinstance(estimator, LongitudinalPipeline):
            if not missing_value_imputation.enabled:
                return estimator
            if isinstance(estimator, ImputedLongitudinalPipeline):
                return estimator
            return ImputedLongitudinalPipeline(
                steps=list(estimator.steps),
                features_group=[list(group) for group in estimator.features_group],
                non_longitudinal_features=list(estimator.non_longitudinal_features),
                update_feature_groups_callback=estimator.update_feature_groups_callback,
                feature_list_names=(
                    list(estimator.feature_list_names)
                    if estimator.feature_list_names is not None
                    else None
                ),
                missing_value_imputation_strategy=missing_value_imputation.strategy,
            )

        if isinstance(estimator, Pipeline):
            if not missing_value_imputation.enabled:
                return estimator
            return Pipeline(
                steps=[
                    (
                        "missing_value_imputer",
                        MissingValueImputer(
                            strategy=missing_value_imputation.strategy
                        ),
                    ),
                    ("estimator_pipeline", estimator),
                ]
            )

        if X.empty:
            raise InputValidationError(
                "No usable feature columns were found for modelling."
            )

        numeric_columns = list(X.select_dtypes(include=["number"]).columns)
        categorical_columns = [
            column for column in X.columns if column not in numeric_columns
        ]

        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_columns:
            transformers.append(("numeric", "passthrough", numeric_columns))
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                )
            )

        if not transformers:
            raise InputValidationError(
                "No usable feature columns were found for modelling."
            )

        steps: list[tuple[str, object]] = []
        if missing_value_imputation.enabled:
            steps.append(
                (
                    "missing_value_imputer",
                    MissingValueImputer(strategy=missing_value_imputation.strategy),
                )
            )
        steps.append(("preprocessor", ColumnTransformer(transformers=transformers)))
        steps.append(("estimator", estimator))
        return Pipeline(steps=steps)

    @staticmethod
    @beartype
    def _write_artifacts(
        *,
        estimator_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        output_dir: Path,
        n_splits: int,
        evaluation_strategy: str,
        random_seed: int,
        holdout_split: float | None,
        split_strategy: str,
        silent_training_output: bool,
        missing_value_imputation_enabled: bool,
        missing_value_imputation_strategy: str,
        fold_scores: tuple[float, ...],
        metric_summaries: dict[str, ClassificationMetricSummary],
        report_text: str,
        fitted_estimator: BaseEstimator,
        class_distribution: dict[object, int],
    ) -> ClassificationExperimentArtifacts:
        """Persist experiment artefacts.

        Args:
            estimator_name: Estimator display name.
            metric_key: Primary metric key used for scoring.
            metric_keys: Ordered metric keys evaluated for scoring.
            output_dir: Artefact output directory.
            n_splits: Number of folds or repeated holdout splits.
            evaluation_strategy: Public evaluation strategy key.
            random_seed: Random seed used for experiment.
            holdout_split: Holdout ratio, when applicable.
            missing_value_imputation_enabled: Whether missing-value imputation
                was fitted inside the model pipeline.
            missing_value_imputation_strategy: Strategy used for training-time
                missing-value imputation.
            fold_scores: Per-fold primary-metric scores.
            metric_summaries: Per-metric fold/mean/std summaries.
            report_text: Classification report text.
            fitted_estimator: Retrained estimator.
            class_distribution: Observed target-class distribution.

        Returns:
            ClassificationExperimentArtifacts: Saved artefact paths.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_estimator = estimator_name.lower().replace(" ", "_")
        artifact_prefix = f"{safe_estimator}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{artifact_prefix}_model.pkl"
        summary_path = output_dir / f"{artifact_prefix}_summary.json"
        report_path = output_dir / f"{artifact_prefix}_classification_report.txt"

        with model_path.open("wb") as file:
            pickle.dump(fitted_estimator, file)

        metric_summaries_payload = {
            key: {
                "fold_scores": [float(score) for score in summary.fold_scores],
                "mean_score": float(summary.mean_score),
                "std_score": float(summary.std_score),
            }
            for key, summary in metric_summaries.items()
        }
        summary_payload = {
            "estimator": estimator_name,
            "metric_of_interest": metric_key,
            "metrics_of_interest": list(metric_keys),
            "evaluation_strategy": evaluation_strategy,
            "n_splits": n_splits,
            "random_seed": random_seed,
            "holdout_split": (
                float(holdout_split) if holdout_split is not None else None
            ),
            "split_strategy": split_strategy,
            "silent_training_output": bool(silent_training_output),
            "missing_value_imputation": {
                "enabled": bool(missing_value_imputation_enabled),
                "strategy": missing_value_imputation_strategy,
            },
            "fold_scores": [float(score) for score in fold_scores],
            "mean_score": float(np.nanmean(fold_scores)),
            "std_score": float(np.nanstd(fold_scores)),
            "metric_summaries": metric_summaries_payload,
            "class_distribution": {
                str(key): int(value) for key, value in class_distribution.items()
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        report_path.write_text(report_text)

        return ClassificationExperimentArtifacts(
            model_path=model_path,
            summary_path=summary_path,
            report_path=report_path,
        )
