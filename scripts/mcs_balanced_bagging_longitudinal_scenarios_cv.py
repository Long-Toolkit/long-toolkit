from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if SRC_ROOT.exists():
    resolved_src_root = SRC_ROOT.resolve()
    if str(resolved_src_root) not in sys.path:
        sys.path.insert(0, str(resolved_src_root))


DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "MCS"
    / "mcs_longitudinal_wide_preprocessed.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "mcs_balanced_bagging_longitudinal_scenarios_cv"
)

TARGET_COLUMN = "class"
DEFAULT_EXCLUDE_CLASS_1 = False
RANDOM_SEED = 42
DEFAULT_N_SPLITS = 10
NUMERIC_IMPUTATION_STRATEGY = "mean"
NUMERIC_SCALER_RANGE = (0.0, 1.0)
DEFAULT_MODEL_N_JOBS = 1
DEFAULT_FOLD_N_JOBS = 2

BAGGING_HYPERPARAMETERS = {
    "n_estimators": 15,
    "sampling_strategy": "all",
    "replacement": True,
    "bootstrap": True,
    "max_samples": 0.7529930396701624,
}

GB_HYPERPARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.048736757615316824,
    "n_estimators": 350,
    "subsample": 0.7923998099457771,
    "max_depth": 4,
}

LEXICO_THRESHOLD_GAIN = 0.018160795832556916

LONGITUDINAL_SUFFIX_PATTERN = re.compile(r"_w(\d+)$")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
np.random.seed(RANDOM_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the balanced bagging longitudinal scenarios CV benchmark from the "
            "notebook as a terminal-friendly Python script."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the preprocessed wide MCS CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV, TXT, JSON, and PNG outputs will be written.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--fold-jobs",
        type=int,
        default=DEFAULT_FOLD_N_JOBS,
        help="Number of parallel worker processes used across CV folds.",
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=DEFAULT_MODEL_N_JOBS,
        help=(
            "Number of parallel jobs used inside each model. Keep this at 1 when "
            "running folds in parallel to avoid CPU oversubscription."
        ),
    )
    parser.add_argument(
        "--exclude-class-1",
        action="store_true",
        help="Drop class 1 rows before benchmarking.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip saving PNG plots.",
    )
    return parser.parse_args()


def build_candidate_specs() -> list[dict[str, Any]]:
    from long_toolkit.machine_learning.tools.longitudinal_machine_learning import (
        estimators as longitudinal_estimators,
    )

    # Toggle scenarios here the same way as in the notebook: comment or uncomment
    # the spec dicts you want to include in the run.
    return [
        # {
        #     "display_name": "Balanced Bagging + Gradient Boosting (Current best)",
        #     "category": "Baseline",
        #     "kind": "baseline",
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        # {
        #     "display_name": (
        #         "Balanced Bagging + MerWavTimeMinus + Gradient Boosting"
        #     ),
        #     "category": "Data Transformation",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         MerWavTimeMinusGradientBoostingLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        # {
        #     "display_name": "Balanced Bagging + AggrFunc (Mean) + Gradient Boosting",
        #     "category": "Data Transformation",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         AggrFuncMeanGradientBoostingLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        # {
        #     "display_name": (
        #         "Balanced Bagging + AggrFunc (Median) + Gradient Boosting"
        #     ),
        #     "category": "Data Transformation",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         AggrFuncMedianGradientBoostingLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        # {
        #     "display_name": "Balanced Bagging + SepWav (Voting) + Gradient Boosting",
        #     "category": "Data Transformation",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         SepWavVotingGradientBoostingLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        {
            "display_name": (
                "Balanced Bagging + SepWav (Stacking + Logistic Regression) + "
                "Gradient Boosting"
            ),
            "category": "Data Transformation",
            "kind": "template",
            "template_cls": (
                longitudinal_estimators.SepWavStackingLRGradientBoostingLongitudinalEstimatorTemplate
            ),
            "hyperparameters": dict(GB_HYPERPARAMETERS),
        },
        # {
        #     "display_name": (
        #         "Balanced Bagging + SepWav (Stacking + Decision Tree) + "
        #         "Gradient Boosting"
        #     ),
        #     "category": "Data Transformation",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         SepWavStackingDTGradientBoostingLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": dict(GB_HYPERPARAMETERS),
        # },
        # {
        #     "display_name": (
        #         "Balanced Bagging + MerWavTimePlus + Lexicographical Decision "
        #         "Tree"
        #     ),
        #     "category": "Longitudinal-native Estimator",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         MerWavTimePlusLexicoDecisionTreeLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": {
        #         "threshold_gain": LEXICO_THRESHOLD_GAIN,
        #         "max_depth": 4,
        #     },
        # },
        # {
        #     "display_name": (
        #         "Balanced Bagging + MerWavTimePlus + Lexicographical Random "
        #         "Forest"
        #     ),
        #     "category": "Longitudinal-native Estimator",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         MerWavTimePlusLexicoRandomForestLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": {
        #         "threshold_gain": LEXICO_THRESHOLD_GAIN,
        #         "n_estimators": 100,
        #         "max_depth": 4,
        #     },
        # },
        # {
        #     "display_name": (
        #         "Balanced Bagging + MerWavTimePlus + Lexicographical Deep "
        #         "Forest"
        #     ),
        #     "category": "Longitudinal-native Estimator",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         MerWavTimePlusLexicoDeepForestLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": {
        #         "max_layers": 3,
        #     },
        # },
        # {
        #     "display_name": "Balanced Bagging + MerWavTimePlus + Nested Trees",
        #     "category": "Longitudinal-native Estimator",
        #     "kind": "template",
        #     "template_cls": (
        #         longitudinal_estimators.
        #         MerWavTimePlusNestedTreesLongitudinalEstimatorTemplate
        #     ),
        #     "hyperparameters": {
        #         "max_outer_depth": 3,
        #         "max_inner_depth": 2,
        #     },
        # },
    ]


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
            X, expected_columns=getattr(self, "feature_names_in_", None)
        )
        updated = frame.copy()
        if getattr(self, "_scaler", None) is not None:
            updated.loc[:, list(self.numeric_columns_)] = self._scaler.transform(
                updated.loc[:, list(self.numeric_columns_)]
            )
        return updated


def align_probability_matrix(
    probability_matrix: np.ndarray,
    estimator_classes: np.ndarray,
    expected_classes: np.ndarray,
) -> np.ndarray:
    aligned = pd.DataFrame(probability_matrix, columns=estimator_classes)
    return aligned.reindex(columns=expected_classes, fill_value=0.0).to_numpy()


def compute_multiclass_metrics(
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


def build_balanced_bagging_classifier(
    estimator: object,
    *,
    model_n_jobs: int,
) -> BalancedBaggingClassifier:
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
        n_jobs=model_n_jobs,
    )


def build_longitudinal_estimator(
    *,
    template_cls: type,
    hyperparameters: dict[str, object],
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
    feature_columns: list[str],
) -> object:
    return template_cls.build_estimator(
        random_seed=RANDOM_SEED,
        hyperparameters=hyperparameters,
        feature_groups=tuple(tuple(group) for group in feature_groups),
        non_longitudinal_features=tuple(non_longitudinal_feature_indices),
        feature_list_names=tuple(feature_columns),
    )


def build_candidate_model(
    *,
    spec: dict[str, Any],
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
    feature_columns: list[str],
    model_n_jobs: int,
) -> Pipeline:
    from long_toolkit.machine_learning.tools.templates.missing_value_imputation import (
        MissingValueImputer,
    )

    if spec["kind"] == "baseline":
        estimator = GradientBoostingClassifier(
            **spec["hyperparameters"],
            random_state=RANDOM_SEED,
        )
    else:
        estimator = build_longitudinal_estimator(
            template_cls=spec["template_cls"],
            hyperparameters=dict(spec["hyperparameters"]),
            feature_groups=feature_groups,
            non_longitudinal_feature_indices=non_longitudinal_feature_indices,
            feature_columns=feature_columns,
        )

    return Pipeline(
        steps=[
            ("impute", MissingValueImputer(strategy=NUMERIC_IMPUTATION_STRATEGY)),
            ("scale", NumericMinMaxScaler(feature_range=NUMERIC_SCALER_RANGE)),
            (
                "classifier",
                build_balanced_bagging_classifier(
                    estimator,
                    model_n_jobs=model_n_jobs,
                ),
            ),
        ]
    )


def evaluate_candidate_fold(
    *,
    fold_index: int,
    n_splits: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    classes: np.ndarray,
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
    feature_columns: list[str],
    model_n_jobs: int,
) -> dict[str, Any]:
    model_name = spec["display_name"]
    print(f"Running {model_name} | fold {fold_index}/{n_splits}...", flush=True)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = build_candidate_model(
        spec=spec,
        feature_groups=feature_groups,
        non_longitudinal_feature_indices=non_longitudinal_feature_indices,
        feature_columns=feature_columns,
        model_n_jobs=model_n_jobs,
    )

    started_at = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test))
    y_score = align_probability_matrix(
        probability_matrix=np.asarray(model.predict_proba(X_test)),
        estimator_classes=np.asarray(model.named_steps["classifier"].classes_),
        expected_classes=classes,
    )
    metric_values = compute_multiclass_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
        classes=classes,
    )
    elapsed_seconds = time.perf_counter() - started_at
    print(
        (
            f"Finished {model_name} | fold {fold_index}/{n_splits} | "
            f"time={elapsed_seconds:.2f}s | "
            f"AUROC={metric_values['auroc']:.4f} | "
            f"AUPRC={metric_values['auprc']:.4f} | "
            f"G-Mean={metric_values['geometric_mean']:.4f}"
        ),
        flush=True,
    )

    return {
        "fold": fold_index,
        "test_index": np.asarray(test_index),
        "y_pred": y_pred,
        "y_score": y_score,
        "fit_predict_seconds": elapsed_seconds,
        **metric_values,
    }


def evaluate_candidate_cv(
    *,
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    classes: np.ndarray,
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
    feature_columns: list[str],
    n_splits: int,
    fold_n_jobs: int,
    model_n_jobs: int,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = list(enumerate(cv.split(X, y), start=1))
    model_name = spec["display_name"]

    print(
        f"Evaluating {model_name} with {n_splits} folds using "
        f"{fold_n_jobs} fold worker(s)...",
        flush=True,
    )

    fold_results = Parallel(
        n_jobs=fold_n_jobs,
        backend="loky",
        batch_size=1,
        verbose=10 if fold_n_jobs > 1 else 0,
    )(
        delayed(evaluate_candidate_fold)(
            fold_index=fold_index,
            n_splits=n_splits,
            train_index=train_index,
            test_index=test_index,
            spec=spec,
            X=X,
            y=y,
            classes=classes,
            feature_groups=feature_groups,
            non_longitudinal_feature_indices=non_longitudinal_feature_indices,
            feature_columns=feature_columns,
            model_n_jobs=model_n_jobs,
        )
        for fold_index, (train_index, test_index) in splits
    )
    fold_results.sort(key=lambda item: item["fold"])

    fold_records: list[dict[str, float | int | str]] = []
    y_true_oof = y.to_numpy()
    y_pred_oof = np.empty_like(y_true_oof)
    y_score_oof = np.zeros((len(y_true_oof), len(classes)), dtype=float)
    assigned_rows = np.zeros(len(y_true_oof), dtype=bool)

    for fold_result in fold_results:
        fold_records.append(
            {
                "fold": int(fold_result["fold"]),
                "model": model_name,
                "category": spec["category"],
                "fit_predict_seconds": float(fold_result["fit_predict_seconds"]),
                "auroc": float(fold_result["auroc"]),
                "auprc": float(fold_result["auprc"]),
                "geometric_mean": float(fold_result["geometric_mean"]),
            }
        )
        test_index = np.asarray(fold_result["test_index"])
        y_pred_oof[test_index] = np.asarray(fold_result["y_pred"])
        y_score_oof[test_index, :] = np.asarray(fold_result["y_score"])
        assigned_rows[test_index] = True

    if not assigned_rows.all():
        missing_predictions = int((~assigned_rows).sum())
        raise RuntimeError(
            f"Cross-validation did not emit predictions for {missing_predictions} rows."
        )

    classification_report_text = classification_report(
        y_true_oof,
        y_pred_oof,
        labels=classes,
        digits=4,
        zero_division=0,
    )
    classification_report_table = pd.DataFrame(
        classification_report(
            y_true_oof,
            y_pred_oof,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).T
    imbalanced_report_text = classification_report_imbalanced(
        y_true_oof,
        y_pred_oof,
        labels=classes,
        target_names=[str(label) for label in classes],
        digits=4,
        zero_division=0,
    )
    imbalanced_report_table = pd.DataFrame(
        classification_report_imbalanced(
            y_true_oof,
            y_pred_oof,
            labels=classes,
            target_names=[str(label) for label in classes],
            output_dict=True,
            zero_division=0,
        )
    ).T

    return {
        "results": pd.DataFrame(fold_records),
        "classification_report_text": classification_report_text,
        "classification_report_table": classification_report_table,
        "imbalanced_report_text": imbalanced_report_text,
        "imbalanced_report_table": imbalanced_report_table,
        "oof_y_true": y_true_oof,
        "oof_y_pred": y_pred_oof,
        "oof_y_score": y_score_oof,
    }


def evaluate_candidates_cv(
    *,
    candidate_specs: list[dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    classes: np.ndarray,
    feature_groups: list[list[int]],
    non_longitudinal_feature_indices: list[int],
    feature_columns: list[str],
    n_splits: int,
    fold_n_jobs: int,
    model_n_jobs: int,
) -> dict[str, Any]:
    all_results: list[pd.DataFrame] = []
    classification_report_text: dict[str, str] = {}
    classification_report_tables: dict[str, pd.DataFrame] = {}
    imbalanced_report_text: dict[str, str] = {}
    imbalanced_report_tables: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []

    for spec in candidate_specs:
        try:
            evaluation = evaluate_candidate_cv(
                spec=spec,
                X=X,
                y=y,
                classes=classes,
                feature_groups=feature_groups,
                non_longitudinal_feature_indices=non_longitudinal_feature_indices,
                feature_columns=feature_columns,
                n_splits=n_splits,
                fold_n_jobs=fold_n_jobs,
                model_n_jobs=model_n_jobs,
            )
            all_results.append(evaluation["results"])
            classification_report_text[spec["display_name"]] = evaluation[
                "classification_report_text"
            ]
            classification_report_tables[spec["display_name"]] = evaluation[
                "classification_report_table"
            ]
            imbalanced_report_text[spec["display_name"]] = evaluation[
                "imbalanced_report_text"
            ]
            imbalanced_report_tables[spec["display_name"]] = evaluation[
                "imbalanced_report_table"
            ]
        except Exception as exc:
            failures.append(
                {
                    "model": spec["display_name"],
                    "category": spec["category"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback_tail": traceback.format_exc(limit=2),
                }
            )
            print(
                f"{spec['display_name']} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    results = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame(
            columns=[
                "fold",
                "model",
                "category",
                "fit_predict_seconds",
                "auroc",
                "auprc",
                "geometric_mean",
            ]
        )
    )

    if results.empty:
        summary = pd.DataFrame(
            columns=[
                "model",
                "category",
                "fit_predict_seconds_mean",
                "fit_predict_seconds_std",
                "auroc_mean",
                "auroc_std",
                "auprc_mean",
                "auprc_std",
                "geometric_mean_mean",
                "geometric_mean_std",
            ]
        )
    else:
        summary = (
            results.groupby(["model", "category"])[
                ["fit_predict_seconds", "auroc", "auprc", "geometric_mean"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(part for part in column if part)
            for column in summary.columns.to_flat_index()
        ]
        summary = summary.rename(
            columns={
                "model_": "model",
                "category_": "category",
            }
        )

    return {
        "results": results,
        "summary": summary,
        "classification_report_text": classification_report_text,
        "classification_report_tables": classification_report_tables,
        "imbalanced_report_text": imbalanced_report_text,
        "imbalanced_report_tables": imbalanced_report_tables,
        "failures": pd.DataFrame(failures),
    }


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def print_heading(title: str) -> None:
    line = "=" * 120
    print(line)
    print(title)
    print(line)


def print_frame(
    title: str, frame: pd.DataFrame, *, round_digits: int | None = None
) -> None:
    print_heading(title)
    if frame.empty:
        print("<empty>")
        return

    printable = frame.round(round_digits) if round_digits is not None else frame
    print(printable.to_string(index=False))


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_class_distribution_plot(
    *,
    class_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        class_distribution["class"].astype(str),
        class_distribution["row_count"],
        color="#1f77b4",
        alpha=0.9,
    )
    max_count = int(class_distribution["row_count"].max())
    for bar, value in zip(bars, class_distribution["row_count"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.01,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows")
    ax.set_title("Dataset class distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_plot(
    *,
    summary: pd.DataFrame,
    baseline_name: str,
    output_path: Path,
    n_splits: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    if summary.empty:
        return

    plot_data = summary.copy()
    plot_data["color"] = np.where(
        plot_data["model"] == baseline_name,
        "#2ca02c",
        np.where(
            plot_data["category"] == "Longitudinal-native Estimator",
            "#ff7f0e",
            "#1f77b4",
        ),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, max(6, 0.45 * len(plot_data))),
        sharey=True,
    )
    metric_columns = ["auroc_mean", "auprc_mean", "geometric_mean_mean"]
    metric_titles = ["AUROC", "AUPRC", "Geometric Mean"]

    for ax, metric, title in zip(axes, metric_columns, metric_titles, strict=True):
        ax.barh(
            plot_data["model"],
            plot_data[metric],
            color=plot_data["color"],
            alpha=0.9,
        )
        for value, model_name in zip(
            plot_data[metric], plot_data["model"], strict=True
        ):
            ax.text(value + 0.005, model_name, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_ylabel("Model")
    fig.suptitle(
        f"{n_splits}-fold CV benchmark across bagged longitudinal scenarios",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_run_outputs(
    *,
    output_dir: Path,
    dataset_summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    classification_report_text: dict[str, str],
    classification_report_tables: dict[str, pd.DataFrame],
    imbalanced_report_text: dict[str, str],
    imbalanced_report_tables: dict[str, pd.DataFrame],
    benchmark_seconds: float,
    n_splits: int,
    fold_n_jobs: int,
    model_n_jobs: int,
    data_path: Path,
    skip_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(dataset_summary, output_dir / "dataset_summary.csv")
    save_dataframe(class_distribution, output_dir / "class_distribution.csv")
    save_dataframe(preprocessing_summary, output_dir / "preprocessing_summary.csv")
    save_dataframe(candidate_summary, output_dir / "candidate_summary.csv")
    save_dataframe(results, output_dir / "fold_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    save_dataframe(failures, output_dir / "failures.csv")

    metadata = {
        "project_root": str(PROJECT_ROOT),
        "data_path": str(data_path),
        "target_column": TARGET_COLUMN,
        "random_seed": RANDOM_SEED,
        "cv_folds": n_splits,
        "fold_jobs": fold_n_jobs,
        "model_n_jobs": model_n_jobs,
        "benchmark_seconds": benchmark_seconds,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    reports_dir = output_dir / "reports"
    tables_dir = output_dir / "report_tables"
    for model_name, report_text in classification_report_text.items():
        model_slug = slugify(model_name)
        save_text(report_text, reports_dir / f"{model_slug}_classification_report.txt")
        save_dataframe(
            classification_report_tables[model_name].reset_index(names="label"),
            tables_dir / f"{model_slug}_classification_report.csv",
        )
        save_text(
            imbalanced_report_text[model_name],
            reports_dir / f"{model_slug}_imbalanced_classification_report.txt",
        )
        save_dataframe(
            imbalanced_report_tables[model_name].reset_index(names="label"),
            tables_dir / f"{model_slug}_imbalanced_classification_report.csv",
        )

    if not skip_plots:
        save_class_distribution_plot(
            class_distribution=class_distribution,
            output_path=output_dir / "class_distribution.png",
        )
        save_benchmark_plot(
            summary=summary,
            baseline_name="Balanced Bagging + Gradient Boosting (Current best)",
            output_path=output_dir / "benchmark_metrics.png",
            n_splits=n_splits,
        )


def main() -> None:
    args = parse_args()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.fold_jobs < 1:
        raise ValueError("--fold-jobs must be at least 1.")
    if args.model_n_jobs < 1:
        raise ValueError("--model-n-jobs must be at least 1.")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    candidate_specs = build_candidate_specs()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Exclude class 1: {args.exclude_class_1}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"CV folds: {args.n_splits}")
    print(f"Fold workers: {args.fold_jobs}")
    print(f"Model n_jobs: {args.model_n_jobs}")
    print(f"Numeric imputation strategy: {NUMERIC_IMPUTATION_STRATEGY}")
    print(f"Numeric scaler range: {NUMERIC_SCALER_RANGE}")
    print(f"Bagging hyperparameters: {BAGGING_HYPERPARAMETERS}")
    print(f"Gradient boosting hyperparameters: {GB_HYPERPARAMETERS}")

    data = pd.read_csv(data_path)
    if args.exclude_class_1:
        data = data.loc[data[TARGET_COLUMN] != 1].copy()

    analysis_scope = (
        "classes 2, 3, 4 only" if args.exclude_class_1 else "classes 1, 2, 3, 4"
    )
    X = data.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    y = data[TARGET_COLUMN].astype(int).reset_index(drop=True)
    classes = np.sort(y.unique())
    feature_columns = X.columns.tolist()
    feature_groups = infer_feature_groups(feature_columns)
    non_longitudinal_columns = [
        column for column in feature_columns if column.startswith("non_long__")
    ]
    non_longitudinal_feature_indices = [
        index
        for index, column in enumerate(feature_columns)
        if column.startswith("non_long__")
    ]
    numeric_feature_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(X[column])
    ]

    candidate_summary = pd.DataFrame(
        [
            {
                "model": spec["display_name"],
                "category": spec["category"],
                "hyperparameters": spec["hyperparameters"],
            }
            for spec in candidate_specs
        ]
    )
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset_path": str(data_path),
                "analysis_scope": analysis_scope,
                "rows": len(data),
                "feature_columns": len(feature_columns),
                "scaled_numeric_columns": len(numeric_feature_columns),
                "non_longitudinal_columns": len(non_longitudinal_columns),
                "longitudinal_feature_groups": len(feature_groups),
                "classes": ", ".join(str(label) for label in classes),
                "candidate_models": len(candidate_specs),
            }
        ]
    )
    class_distribution = (
        y.value_counts().sort_index().rename_axis("class").reset_index(name="row_count")
    )
    class_distribution["dataset_share"] = class_distribution["row_count"] / len(y)
    preprocessing_summary = pd.DataFrame(
        [
            {
                "exclude_class_1": args.exclude_class_1,
                "imputation_strategy": NUMERIC_IMPUTATION_STRATEGY,
                "scaler_min": NUMERIC_SCALER_RANGE[0],
                "scaler_max": NUMERIC_SCALER_RANGE[1],
                "cv_folds": args.n_splits,
                "fold_jobs": args.fold_jobs,
                "model_n_jobs": args.model_n_jobs,
                "bagging_hyperparameters": BAGGING_HYPERPARAMETERS,
                "gb_hyperparameters": GB_HYPERPARAMETERS,
            }
        ]
    )

    print_frame("Dataset Summary", dataset_summary)
    print_frame("Class Distribution", class_distribution, round_digits=4)
    print_frame("Preprocessing Summary", preprocessing_summary)
    print_frame("Candidate Summary", candidate_summary)

    benchmark_started_at = time.perf_counter()
    benchmark = evaluate_candidates_cv(
        candidate_specs=candidate_specs,
        X=X,
        y=y,
        classes=classes,
        feature_groups=feature_groups,
        non_longitudinal_feature_indices=non_longitudinal_feature_indices,
        feature_columns=feature_columns,
        n_splits=args.n_splits,
        fold_n_jobs=args.fold_jobs,
        model_n_jobs=args.model_n_jobs,
    )
    benchmark_seconds = time.perf_counter() - benchmark_started_at

    results = benchmark["results"]
    summary = benchmark["summary"]
    classification_report_text = benchmark["classification_report_text"]
    classification_report_tables = benchmark["classification_report_tables"]
    imbalanced_classification_report_text = benchmark["imbalanced_report_text"]
    imbalanced_classification_report_tables = benchmark["imbalanced_report_tables"]
    failures = benchmark["failures"]

    print(f"Total benchmark runtime: {benchmark_seconds:.2f} seconds")
    if not results.empty:
        print_frame("Fold Results (Head)", results.head(), round_digits=4)
    if not failures.empty:
        print_frame("Failures", failures)
    if summary.empty:
        raise RuntimeError("No successful candidate runs were produced.")

    summary = summary.sort_values("geometric_mean_mean", ascending=False).reset_index(
        drop=True
    )
    baseline_name = "Balanced Bagging + Gradient Boosting (Current best)"
    baseline_mask = summary["model"].eq(baseline_name)
    if baseline_mask.any():
        baseline_geometric_mean = float(
            summary.loc[baseline_mask, "geometric_mean_mean"].iloc[0]
        )
        summary["delta_vs_baseline_geometric_mean"] = (
            summary["geometric_mean_mean"] - baseline_geometric_mean
        )
    else:
        print(
            f'Baseline "{baseline_name}" is not in the successful summary. '
            "Setting delta to NaN."
        )
        summary["delta_vs_baseline_geometric_mean"] = np.nan

    summary["winner_by_geometric_mean"] = (
        summary["geometric_mean_mean"] == summary["geometric_mean_mean"].max()
    )
    ranking_view = summary[
        [
            "model",
            "category",
            "fit_predict_seconds_mean",
            "auroc_mean",
            "auprc_mean",
            "geometric_mean_mean",
            "delta_vs_baseline_geometric_mean",
        ]
    ].copy()

    print_frame("Summary", summary, round_digits=4)
    print_frame("Ranking View", ranking_view, round_digits=4)

    reported_models = summary["model"].tolist()
    for model_name in reported_models:
        print_heading(f"Standard out-of-fold classification report: {model_name}")
        print(classification_report_text[model_name])
        print_frame(
            f"Standard report table: {model_name}",
            classification_report_tables[model_name],
            round_digits=4,
        )
        print_heading(f"Imbalanced out-of-fold classification report: {model_name}")
        print(imbalanced_classification_report_text[model_name])
        print_frame(
            f"Imbalanced report table: {model_name}",
            imbalanced_classification_report_tables[model_name],
            round_digits=4,
        )

    save_run_outputs(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        preprocessing_summary=preprocessing_summary,
        candidate_summary=candidate_summary,
        results=results,
        summary=summary,
        failures=failures,
        classification_report_text=classification_report_text,
        classification_report_tables=classification_report_tables,
        imbalanced_report_text=imbalanced_classification_report_text,
        imbalanced_report_tables=imbalanced_classification_report_tables,
        benchmark_seconds=benchmark_seconds,
        n_splits=args.n_splits,
        fold_n_jobs=args.fold_jobs,
        model_n_jobs=args.model_n_jobs,
        data_path=data_path,
        skip_plots=args.skip_plots,
    )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
