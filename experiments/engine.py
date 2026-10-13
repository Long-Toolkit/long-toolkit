from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiments.registry import (
    BAGGING_HYPERPARAMETERS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SPLIT_MANIFEST_PATH,
    NUMERIC_SCALER_RANGE,
    PrimitiveSpec,
    RANDOM_SEED,
    TRAIN_IMPUTATION_STRATEGY,
    get_primitive_spec,
)
from experiments.splits import (
    SplitManifest,
    load_split_manifest,
    normalise_target_series,
    prepare_split_manifest,
    resolve_dataset_path_from_manifest,
)
from long_toolkit.machine_learning.tools.longitudinal_machine_learning.discovery import (
    discover_longitudinal_estimators,
)
from long_toolkit.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from long_toolkit.machine_learning.tools.templates.missing_value_imputation import (
    MissingValueImputer,
)

LONGITUDINAL_SUFFIX_PATTERN = re.compile(r"_w(\d+)$")


def parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pre-generated MCS experiment launcher for one fold."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to the MCS wide CSV dataset. Defaults to the manifest dataset.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Prepared holdout + CV split manifest JSON.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        required=True,
        help="Zero-based fold index to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory where fold outputs are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the fold even if outputs already exist.",
    )
    return parser.parse_args()


def run_launcher_cli(launcher_key: str) -> None:
    args = parse_launcher_args()
    run_fold_for_launcher(
        launcher_key=launcher_key,
        dataset_path=args.dataset_path,
        split_manifest_path=args.split_manifest,
        fold_index=args.fold_index,
        output_root=args.output_root,
        force=args.force,
    )


def coerce_to_frame(
    X: pd.DataFrame | np.ndarray,
    expected_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        observed_columns = tuple(map(str, frame.columns))
        if expected_columns is not None and observed_columns != expected_columns:
            raise ValueError(
                "Input feature columns do not match the fitted preprocessing pipeline."
            )
        frame.columns = list(observed_columns)
        return frame

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("Input features must be two-dimensional.")

    columns = (
        list(expected_columns)
        if expected_columns is not None
        else [f"feature_{index}" for index in range(array.shape[1])]
    )
    if len(columns) != array.shape[1]:
        raise ValueError(
            "Input feature count does not match the fitted preprocessing pipeline."
        )
    return pd.DataFrame(array, columns=columns)


class NumericMinMaxScaler(BaseEstimator, TransformerMixin):
    """Scale numeric columns only while preserving DataFrame columns."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> NumericMinMaxScaler:
        _ = y
        frame = coerce_to_frame(X)
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        self.numeric_columns_ = tuple(
            column
            for column in self.feature_names_in_
            if pd.api.types.is_numeric_dtype(frame[column])
            and not frame[column].dropna().empty
        )
        self._scaler = None
        if self.numeric_columns_:
            self._scaler = MinMaxScaler(feature_range=self.feature_range)
            self._scaler.fit(frame.loc[:, list(self.numeric_columns_)])
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = coerce_to_frame(
            X,
            expected_columns=getattr(self, "feature_names_in_", None),
        )
        updated = frame.copy()
        if getattr(self, "_scaler", None) is not None:
            updated.loc[:, list(self.numeric_columns_)] = self._scaler.transform(
                updated.loc[:, list(self.numeric_columns_)]
            )
        return updated


def infer_feature_groups(feature_columns: list[str]) -> list[list[int]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, column in enumerate(feature_columns):
        if column.startswith("non_long__"):
            continue
        match = LONGITUDINAL_SUFFIX_PATTERN.search(column)
        if match is None:
            continue
        base_name = column[: match.start()]
        wave = int(match.group(1))
        groups[base_name].append((wave, index))

    return [
        [column_index for _, column_index in sorted(group_entries)]
        for _, group_entries in sorted(groups.items())
        if len(group_entries) >= 2
    ]


def resolve_non_longitudinal_feature_indices(feature_columns: list[str]) -> list[int]:
    return [
        index for index, column in enumerate(feature_columns) if column.startswith("non_long__")
    ]


def filter_supported_hyperparameters(template_cls: type, desired: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in desired.items()
        if template_cls.supports_hyperparameter(key)
    }


def build_balanced_bagging_classifier(estimator: object) -> BalancedBaggingClassifier:
    return BalancedBaggingClassifier(
        estimator=estimator,
        n_estimators=BAGGING_HYPERPARAMETERS["n_estimators"],
        sampling_strategy=BAGGING_HYPERPARAMETERS["sampling_strategy"],
        replacement=BAGGING_HYPERPARAMETERS["replacement"],
        bootstrap=BAGGING_HYPERPARAMETERS["bootstrap"],
        max_samples=BAGGING_HYPERPARAMETERS["max_samples"],
        sampler=RandomUnderSampler(
            sampling_strategy=BAGGING_HYPERPARAMETERS["sampling_strategy"],
            replacement=BAGGING_HYPERPARAMETERS["replacement"],
            random_state=RANDOM_SEED,
        ),
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def build_model(
    *,
    spec: PrimitiveSpec,
    feature_columns: list[str],
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
) -> Pipeline:
    if spec.estimator_domain == "standard":
        template_cls = discover_standard_estimators()[spec.primitive_key]
        hyperparameters = filter_supported_hyperparameters(
            template_cls,
            spec.desired_hyperparameters,
        )
        estimator = template_cls.build_estimator(
            hyperparameters=hyperparameters,
            random_seed=RANDOM_SEED,
        )
    else:
        template_cls = discover_longitudinal_estimators()[spec.primitive_key]
        hyperparameters = filter_supported_hyperparameters(
            template_cls,
            spec.desired_hyperparameters,
        )
        estimator = template_cls.build_estimator(
            random_seed=RANDOM_SEED,
            hyperparameters=hyperparameters,
            feature_groups=tuple(tuple(group) for group in feature_groups),
            non_longitudinal_features=tuple(non_longitudinal_feature_indices),
            feature_list_names=tuple(feature_columns),
        )

    final_estimator = (
        build_balanced_bagging_classifier(estimator)
        if spec.balanced_bagging
        else estimator
    )

    return Pipeline(
        steps=[
            ("impute", MissingValueImputer(strategy=TRAIN_IMPUTATION_STRATEGY)),
            ("scale", NumericMinMaxScaler(feature_range=NUMERIC_SCALER_RANGE)),
            ("classifier", final_estimator),
        ]
    )


def load_dataset(
    dataset_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    data = pd.read_csv(dataset_path)
    y = normalise_target_series(data[target_column])
    X = data.drop(columns=[target_column]).copy()
    classes = np.sort(y.dropna().unique())
    return X, y, classes


def validate_manifest_compatibility(
    manifest: SplitManifest,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    expected_rows = int(manifest.dataset_summary["row_count"])
    expected_features = int(manifest.dataset_summary["feature_column_count"])
    if len(X) != expected_rows:
        raise ValueError(
            f"Dataset row count mismatch: manifest expects {expected_rows}, found {len(X)}."
        )
    if X.shape[1] != expected_features:
        raise ValueError(
            "Dataset feature count mismatch: "
            f"manifest expects {expected_features}, found {X.shape[1]}."
        )
    observed_distribution = {
        str(label): int(count)
        for label, count in y.value_counts(dropna=False).sort_index().items()
    }
    if observed_distribution != manifest.dataset_summary["class_distribution"]:
        raise ValueError(
            "Dataset class distribution does not match the prepared split manifest."
        )


def align_probability_matrix(
    probability_matrix: np.ndarray,
    estimator_classes: np.ndarray,
    expected_classes: np.ndarray,
) -> np.ndarray:
    aligned = pd.DataFrame(probability_matrix, columns=estimator_classes)
    return aligned.reindex(columns=expected_classes, fill_value=0.0).to_numpy()


def _softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def extract_score_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    classes: np.ndarray,
) -> tuple[np.ndarray, str]:
    classifier = model.named_steps["classifier"]
    classifier_classes = np.asarray(getattr(classifier, "classes_", classes))

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            probabilities = np.column_stack([1.0 - probabilities[:, 0], probabilities[:, 0]])
        return align_probability_matrix(probabilities, classifier_classes, classes), "predict_proba"

    if hasattr(model, "decision_function"):
        decisions = np.asarray(model.decision_function(X))
        if decisions.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-decisions))
            probabilities = np.column_stack([1.0 - positive, positive])
        else:
            probabilities = _softmax(decisions)
        return align_probability_matrix(probabilities, classifier_classes, classes), "decision_function"

    predictions = np.asarray(model.predict(X))
    one_hot = pd.get_dummies(pd.Series(predictions), dtype=float).reindex(
        columns=list(classes),
        fill_value=0.0,
    )
    return one_hot.to_numpy(), "predicted_labels"


def compute_multiclass_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    from scikit_longitudinal.metrics import auprc_score

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                y_score,
                average="weighted",
                multi_class="ovo",
                labels=classes,
            )
        ),
        "auprc": float(
            auprc_score(
                y_true,
                y_score,
                average="weighted",
                labels=classes,
            )
        ),
        "geometric_mean": float(
            geometric_mean_score(
                y_true,
                y_pred,
                labels=classes,
                average="weighted",
            )
        ),
    }


def report_dict_to_frame(report_dict: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(report_dict).transpose().reset_index()
    frame = frame.rename(columns={"index": "label"})
    return frame


def evaluate_split(
    *,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    classes: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    y_score, score_source = extract_score_matrix(model, X, classes)
    metrics = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        classes=classes,
    )

    classification_report_dict = classification_report(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )
    imbalanced_report_dict = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=classes,
        target_names=[str(label) for label in classes],
        zero_division=0,
        output_dict=True,
    )

    return {
        "metrics": metrics,
        "score_source": score_source,
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=classes,
            zero_division=0,
        ),
        "classification_report_table": report_dict_to_frame(classification_report_dict),
        "imbalanced_report_text": classification_report_imbalanced(
            y_true,
            y_pred,
            labels=classes,
            target_names=[str(label) for label in classes],
            zero_division=0,
        ),
        "imbalanced_report_table": report_dict_to_frame(imbalanced_report_dict),
        "class_distribution": {
            str(label): int(count)
            for label, count in y.value_counts(dropna=False).sort_index().items()
        },
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _fold_output_dir(output_root: Path, spec: PrimitiveSpec, fold_index: int) -> Path:
    return output_root / spec.scope_key / spec.primitive_key / f"fold_{fold_index + 1:02d}"


def _error_payload(
    *,
    spec: PrimitiveSpec,
    fold_index: int | None,
    dataset_path: Path,
    exception: BaseException,
) -> dict[str, Any]:
    return {
        "launcher_key": spec.launcher_key,
        "scope_key": spec.scope_key,
        "primitive_key": spec.primitive_key,
        "display_name": spec.display_name,
        "fold_index": fold_index,
        "dataset_path": str(dataset_path),
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc(),
    }


def run_fold_for_launcher(
    *,
    launcher_key: str,
    dataset_path: Path | None,
    split_manifest_path: Path,
    fold_index: int,
    output_root: Path = DEFAULT_RESULTS_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    spec = get_primitive_spec(launcher_key)
    if not split_manifest_path.exists():
        prepare_split_manifest(output_path=split_manifest_path)

    manifest = load_split_manifest(split_manifest_path)
    resolved_dataset_path = resolve_dataset_path_from_manifest(manifest, dataset_path)
    fold_output_dir = _fold_output_dir(output_root, spec, fold_index)
    success_path = fold_output_dir / "fold_result.json"
    failure_path = fold_output_dir / "fold_failure.json"

    if success_path.exists() and not force:
        return json.loads(success_path.read_text())

    try:
        X, y, classes = load_dataset(resolved_dataset_path, manifest.target_column)
        validate_manifest_compatibility(manifest, X, y)

        if fold_index < 0 or fold_index >= len(manifest.folds):
            raise IndexError(
                f"Fold index {fold_index} is out of range for {len(manifest.folds)} folds."
            )

        fold_split = manifest.folds[fold_index]
        feature_columns = list(X.columns)
        feature_groups = infer_feature_groups(feature_columns)
        non_longitudinal_feature_indices = resolve_non_longitudinal_feature_indices(
            feature_columns
        )

        model = build_model(
            spec=spec,
            feature_columns=feature_columns,
            feature_groups=feature_groups,
            non_longitudinal_feature_indices=non_longitudinal_feature_indices,
        )

        X_train = X.iloc[list(fold_split.train_indices)]
        y_train = y.iloc[list(fold_split.train_indices)]
        X_validation = X.iloc[list(fold_split.validation_indices)]
        y_validation = y.iloc[list(fold_split.validation_indices)]

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed_seconds = time.perf_counter() - start_time

        train_evaluation = evaluate_split(
            model=model,
            X=X_train,
            y=y_train,
            classes=classes,
        )
        validation_evaluation = evaluate_split(
            model=model,
            X=X_validation,
            y=y_validation,
            classes=classes,
        )

        _write_frame(
            fold_output_dir / "train_classification_report.csv",
            train_evaluation["classification_report_table"],
        )
        _write_frame(
            fold_output_dir / "train_imbalanced_report.csv",
            train_evaluation["imbalanced_report_table"],
        )
        _write_frame(
            fold_output_dir / "validation_classification_report.csv",
            validation_evaluation["classification_report_table"],
        )
        _write_frame(
            fold_output_dir / "validation_imbalanced_report.csv",
            validation_evaluation["imbalanced_report_table"],
        )
        _write_text(
            fold_output_dir / "train_classification_report.txt",
            train_evaluation["classification_report_text"],
        )
        _write_text(
            fold_output_dir / "train_imbalanced_report.txt",
            train_evaluation["imbalanced_report_text"],
        )
        _write_text(
            fold_output_dir / "validation_classification_report.txt",
            validation_evaluation["classification_report_text"],
        )
        _write_text(
            fold_output_dir / "validation_imbalanced_report.txt",
            validation_evaluation["imbalanced_report_text"],
        )

        payload = {
            "launcher_key": spec.launcher_key,
            "scope_key": spec.scope_key,
            "primitive_key": spec.primitive_key,
            "display_name": spec.display_name,
            "balanced_bagging": spec.balanced_bagging,
            "estimator_domain": spec.estimator_domain,
            "fold_index": fold_index,
            "train_rows": len(X_train),
            "validation_rows": len(X_validation),
            "classes": [int(label) if float(label).is_integer() else float(label) for label in classes],
            "desired_hyperparameters": spec.desired_hyperparameters,
            "elapsed_seconds": elapsed_seconds,
            "metrics": {
                "train": train_evaluation["metrics"],
                "validation": validation_evaluation["metrics"],
            },
            "score_sources": {
                "train": train_evaluation["score_source"],
                "validation": validation_evaluation["score_source"],
            },
            "class_distributions": {
                "train": train_evaluation["class_distribution"],
                "validation": validation_evaluation["class_distribution"],
            },
            "artifacts": {
                "train_classification_report_csv": str(
                    fold_output_dir / "train_classification_report.csv"
                ),
                "train_imbalanced_report_csv": str(
                    fold_output_dir / "train_imbalanced_report.csv"
                ),
                "validation_classification_report_csv": str(
                    fold_output_dir / "validation_classification_report.csv"
                ),
                "validation_imbalanced_report_csv": str(
                    fold_output_dir / "validation_imbalanced_report.csv"
                ),
            },
        }
        _write_json(success_path, payload)
        if failure_path.exists():
            failure_path.unlink()
        return payload
    except Exception as exception:
        error_payload = _error_payload(
            spec=spec,
            fold_index=fold_index,
            dataset_path=resolved_dataset_path if "resolved_dataset_path" in locals() else Path("."),
            exception=exception,
        )
        _write_json(failure_path, error_payload)
        raise


def fit_and_evaluate_full_train_test(
    *,
    spec: PrimitiveSpec,
    dataset_path: Path,
    split_manifest_path: Path,
    output_root: Path = DEFAULT_RESULTS_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest_path)
    resolved_dataset_path = resolve_dataset_path_from_manifest(manifest, dataset_path)
    final_output_dir = output_root / spec.scope_key / spec.primitive_key / "final_model"
    success_path = final_output_dir / "final_evaluation.json"
    failure_path = final_output_dir / "final_evaluation_failure.json"

    if success_path.exists() and not force:
        return json.loads(success_path.read_text())

    try:
        X, y, classes = load_dataset(resolved_dataset_path, manifest.target_column)
        validate_manifest_compatibility(manifest, X, y)

        feature_columns = list(X.columns)
        feature_groups = infer_feature_groups(feature_columns)
        non_longitudinal_feature_indices = resolve_non_longitudinal_feature_indices(
            feature_columns
        )

        model = build_model(
            spec=spec,
            feature_columns=feature_columns,
            feature_groups=feature_groups,
            non_longitudinal_feature_indices=non_longitudinal_feature_indices,
        )

        X_train = X.iloc[list(manifest.train_indices)]
        y_train = y.iloc[list(manifest.train_indices)]
        X_test = X.iloc[list(manifest.test_indices)]
        y_test = y.iloc[list(manifest.test_indices)]

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed_seconds = time.perf_counter() - start_time

        train_evaluation = evaluate_split(
            model=model,
            X=X_train,
            y=y_train,
            classes=classes,
        )
        test_evaluation = evaluate_split(
            model=model,
            X=X_test,
            y=y_test,
            classes=classes,
        )

        final_output_dir.mkdir(parents=True, exist_ok=True)
        with (final_output_dir / "model.pkl").open("wb") as file:
            pickle.dump(model, file)

        _write_frame(
            final_output_dir / "train_classification_report.csv",
            train_evaluation["classification_report_table"],
        )
        _write_frame(
            final_output_dir / "train_imbalanced_report.csv",
            train_evaluation["imbalanced_report_table"],
        )
        _write_frame(
            final_output_dir / "test_classification_report.csv",
            test_evaluation["classification_report_table"],
        )
        _write_frame(
            final_output_dir / "test_imbalanced_report.csv",
            test_evaluation["imbalanced_report_table"],
        )
        _write_text(
            final_output_dir / "train_classification_report.txt",
            train_evaluation["classification_report_text"],
        )
        _write_text(
            final_output_dir / "train_imbalanced_report.txt",
            train_evaluation["imbalanced_report_text"],
        )
        _write_text(
            final_output_dir / "test_classification_report.txt",
            test_evaluation["classification_report_text"],
        )
        _write_text(
            final_output_dir / "test_imbalanced_report.txt",
            test_evaluation["imbalanced_report_text"],
        )

        payload = {
            "launcher_key": spec.launcher_key,
            "scope_key": spec.scope_key,
            "primitive_key": spec.primitive_key,
            "display_name": spec.display_name,
            "balanced_bagging": spec.balanced_bagging,
            "estimator_domain": spec.estimator_domain,
            "dataset_path": str(resolved_dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "classes": [int(label) if float(label).is_integer() else float(label) for label in classes],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "elapsed_seconds": elapsed_seconds,
            "metrics": {
                "train": train_evaluation["metrics"],
                "test": test_evaluation["metrics"],
            },
            "score_sources": {
                "train": train_evaluation["score_source"],
                "test": test_evaluation["score_source"],
            },
            "class_distributions": {
                "train": train_evaluation["class_distribution"],
                "test": test_evaluation["class_distribution"],
            },
            "feature_columns": feature_columns,
            "artifacts": {
                "model_path": str(final_output_dir / "model.pkl"),
                "train_classification_report_csv": str(
                    final_output_dir / "train_classification_report.csv"
                ),
                "train_imbalanced_report_csv": str(
                    final_output_dir / "train_imbalanced_report.csv"
                ),
                "test_classification_report_csv": str(
                    final_output_dir / "test_classification_report.csv"
                ),
                "test_imbalanced_report_csv": str(
                    final_output_dir / "test_imbalanced_report.csv"
                ),
            },
        }
        _write_json(success_path, payload)
        if failure_path.exists():
            failure_path.unlink()
        return payload
    except Exception as exception:
        error_payload = _error_payload(
            spec=spec,
            fold_index=None,
            dataset_path=resolved_dataset_path if "resolved_dataset_path" in locals() else Path("."),
            exception=exception,
        )
        _write_json(failure_path, error_payload)
        raise
