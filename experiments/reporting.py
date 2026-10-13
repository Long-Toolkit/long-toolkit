from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.engine import fit_and_evaluate_full_train_test, load_dataset
from experiments.registry import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RANKING_METRIC,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SHAP_BACKGROUND_SIZE,
    DEFAULT_SHAP_MAX_DISPLAY,
    DEFAULT_SHAP_ROWS_PER_CLASS,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TOP_K,
    N_FOLDS,
    PrimitiveSpec,
    iter_primitive_specs,
)
from experiments.splits import load_split_manifest, resolve_dataset_path_from_manifest

matplotlib.use("Agg")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float(value)


def collect_fold_status(
    *,
    spec: PrimitiveSpec,
    results_root: Path,
) -> dict[str, Any]:
    primitive_root = results_root / spec.scope_key / spec.primitive_key
    completed_payloads: list[dict[str, Any]] = []
    failed_paths: list[str] = []

    for fold_number in range(1, N_FOLDS + 1):
        fold_dir = primitive_root / f"fold_{fold_number:02d}"
        success_path = fold_dir / "fold_result.json"
        failure_path = fold_dir / "fold_failure.json"
        if success_path.exists():
            completed_payloads.append(_read_json(success_path))
        elif failure_path.exists():
            failed_paths.append(str(failure_path))

    record: dict[str, Any] = {
        "launcher_key": spec.launcher_key,
        "scope_key": spec.scope_key,
        "primitive_key": spec.primitive_key,
        "display_name": spec.display_name,
        "balanced_bagging": spec.balanced_bagging,
        "estimator_domain": spec.estimator_domain,
        "completed_folds": len(completed_payloads),
        "failed_folds": len(failed_paths),
        "is_complete": len(completed_payloads) == N_FOLDS,
        "failure_paths": ";".join(failed_paths),
    }

    for stage in ("train", "validation"):
        for metric_key in ("auroc", "auprc", "geometric_mean"):
            scores = [
                payload["metrics"][stage][metric_key]
                for payload in completed_payloads
                if metric_key in payload["metrics"][stage]
            ]
            record[f"cv_{stage}_{metric_key}_mean"] = _safe_float(
                np.mean(scores) if scores else np.nan
            )
            record[f"cv_{stage}_{metric_key}_std"] = _safe_float(
                np.std(scores) if scores else np.nan
            )
    return record


def _collect_report_frames(
    *,
    spec: PrimitiveSpec,
    results_root: Path,
    stage: str,
    report_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold_number in range(1, N_FOLDS + 1):
        report_path = (
            results_root
            / spec.scope_key
            / spec.primitive_key
            / f"fold_{fold_number:02d}"
            / f"{stage}_{report_name}.csv"
        )
        if not report_path.exists():
            continue
        frame = pd.read_csv(report_path)
        frame["fold_index"] = fold_number - 1
        frame["launcher_key"] = spec.launcher_key
        frame["scope_key"] = spec.scope_key
        frame["primitive_key"] = spec.primitive_key
        frame["display_name"] = spec.display_name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        column
        for column in combined.columns
        if column
        not in {
            "label",
            "fold_index",
            "launcher_key",
            "scope_key",
            "primitive_key",
            "display_name",
        }
    ]
    combined.loc[:, numeric_columns] = combined.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return combined


def aggregate_cv_reports(
    *,
    spec: PrimitiveSpec,
    results_root: Path,
    report_root: Path,
) -> None:
    for stage in ("train", "validation"):
        for report_name in ("classification_report", "imbalanced_report"):
            combined = _collect_report_frames(
                spec=spec,
                results_root=results_root,
                stage=stage,
                report_name=report_name,
            )
            if combined.empty:
                continue
            numeric_columns = [
                column
                for column in combined.columns
                if column
                not in {
                    "label",
                    "fold_index",
                    "launcher_key",
                    "scope_key",
                    "primitive_key",
                    "display_name",
                }
            ]
            mean_frame = (
                combined.groupby("label", dropna=False)[numeric_columns]
                .mean()
                .reset_index()
            )
            std_frame = (
                combined.groupby("label", dropna=False)[numeric_columns]
                .std()
                .reset_index()
            )
            primitive_report_dir = report_root / "per_primitive" / spec.scope_key / spec.primitive_key
            primitive_report_dir.mkdir(parents=True, exist_ok=True)
            mean_frame.to_csv(
                primitive_report_dir / f"cv_{stage}_{report_name}_mean.csv",
                index=False,
            )
            std_frame.to_csv(
                primitive_report_dir / f"cv_{stage}_{report_name}_std.csv",
                index=False,
            )


def build_final_evaluation_table(
    *,
    specs: tuple[PrimitiveSpec, ...],
    results_root: Path,
    dataset_path: Path,
    split_manifest_path: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        try:
            payload = fit_and_evaluate_full_train_test(
                spec=spec,
                dataset_path=dataset_path,
                split_manifest_path=split_manifest_path,
                output_root=results_root,
                force=force,
            )
            rows.append(
                {
                    "launcher_key": spec.launcher_key,
                    "scope_key": spec.scope_key,
                    "primitive_key": spec.primitive_key,
                    "display_name": spec.display_name,
                    "balanced_bagging": spec.balanced_bagging,
                    "estimator_domain": spec.estimator_domain,
                    "train_rows": payload["train_rows"],
                    "test_rows": payload["test_rows"],
                    "train_auroc": payload["metrics"]["train"]["auroc"],
                    "train_auprc": payload["metrics"]["train"]["auprc"],
                    "train_geometric_mean": payload["metrics"]["train"]["geometric_mean"],
                    "test_auroc": payload["metrics"]["test"]["auroc"],
                    "test_auprc": payload["metrics"]["test"]["auprc"],
                    "test_geometric_mean": payload["metrics"]["test"]["geometric_mean"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "model_path": payload["artifacts"]["model_path"],
                    "test_imbalanced_report_csv": payload["artifacts"]["test_imbalanced_report_csv"],
                    "train_imbalanced_report_csv": payload["artifacts"]["train_imbalanced_report_csv"],
                    "test_classification_report_csv": payload["artifacts"]["test_classification_report_csv"],
                    "train_classification_report_csv": payload["artifacts"]["train_classification_report_csv"],
                }
            )
        except Exception as exception:
            failures.append(
                {
                    "launcher_key": spec.launcher_key,
                    "scope_key": spec.scope_key,
                    "primitive_key": spec.primitive_key,
                    "display_name": spec.display_name,
                    "error_type": type(exception).__name__,
                    "error_message": str(exception),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def select_specs_for_final_evaluation(
    *,
    specs: tuple[PrimitiveSpec, ...],
    cv_table: pd.DataFrame,
    ranking_metric: str,
    final_top_k: int | None,
) -> tuple[tuple[PrimitiveSpec, ...], pd.DataFrame]:
    if final_top_k is None:
        return specs, cv_table.copy()

    if final_top_k <= 0 or cv_table.empty:
        return tuple(), cv_table.iloc[0:0].copy()

    ranked = cv_table.sort_values(
        by=[
            f"cv_validation_{ranking_metric}_mean",
            "cv_validation_auroc_mean",
            "cv_validation_geometric_mean_mean",
        ],
        ascending=False,
        na_position="last",
    ).head(final_top_k)
    spec_by_launcher_key = {spec.launcher_key: spec for spec in specs}
    selected_specs = tuple(
        spec_by_launcher_key[launcher_key]
        for launcher_key in ranked["launcher_key"].tolist()
        if launcher_key in spec_by_launcher_key
    )
    return selected_specs, ranked.copy()


def write_stage_report_tables(
    *,
    final_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    for split_name in ("train", "test"):
        imbalanced_frames: list[pd.DataFrame] = []
        classification_frames: list[pd.DataFrame] = []
        for _, row in final_table.iterrows():
            imbalanced_path = Path(row[f"{split_name}_imbalanced_report_csv"])
            classification_path = Path(row[f"{split_name}_classification_report_csv"])
            if imbalanced_path.exists():
                frame = pd.read_csv(imbalanced_path)
                frame["launcher_key"] = row["launcher_key"]
                frame["scope_key"] = row["scope_key"]
                frame["primitive_key"] = row["primitive_key"]
                frame["display_name"] = row["display_name"]
                imbalanced_frames.append(frame)
            if classification_path.exists():
                frame = pd.read_csv(classification_path)
                frame["launcher_key"] = row["launcher_key"]
                frame["scope_key"] = row["scope_key"]
                frame["primitive_key"] = row["primitive_key"]
                frame["display_name"] = row["display_name"]
                classification_frames.append(frame)

        if imbalanced_frames:
            pd.concat(imbalanced_frames, ignore_index=True).to_csv(
                output_dir / f"{split_name}_imbalanced_reports.csv",
                index=False,
            )
        if classification_frames:
            pd.concat(classification_frames, ignore_index=True).to_csv(
                output_dir / f"{split_name}_classification_reports.csv",
                index=False,
            )


def plot_metric_bars(
    *,
    table: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    if table.empty or metric_column not in table.columns:
        return
    sorted_table = table.sort_values(metric_column, ascending=False).head(top_n)
    plt.figure(figsize=(12, max(6, 0.4 * len(sorted_table))))
    plt.barh(sorted_table["display_name"], sorted_table[metric_column], color="#2b6cb0")
    plt.gca().invert_yaxis()
    plt.xlabel(metric_column.replace("_", " ").title())
    plt.ylabel("Primitive")
    plt.title(f"Top {len(sorted_table)} models by {metric_column.replace('_', ' ')}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _balanced_sample_per_class(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    rows_per_class: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    sampled_indices: list[int] = []
    for class_label in sorted(y.unique()):
        class_index = y[y == class_label].index
        if len(class_index) == 0:
            continue
        sample_size = min(rows_per_class, len(class_index))
        sampled_indices.extend(
            pd.Series(class_index).sample(
                n=sample_size,
                random_state=random_state,
                replace=False,
            ).tolist()
        )
    sampled_indices = sorted(sampled_indices)
    return X.loc[sampled_indices].copy(), y.loc[sampled_indices].copy()


def _stratified_total_sample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows >= len(X):
        return X.copy(), y.copy()

    indices = np.arange(len(X), dtype=int)
    sampled_indices, _ = train_test_split(
        indices,
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    sampled_indices = np.sort(sampled_indices)
    return X.iloc[sampled_indices].copy(), y.iloc[sampled_indices].copy()


def _score_estimator_matrix(
    estimator: Any,
    X: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    estimator_classes = np.asarray(getattr(estimator, "classes_", classes))

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X))
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            probabilities = np.column_stack(
                [1.0 - probabilities[:, 0], probabilities[:, 0]]
            )
        aligned = pd.DataFrame(probabilities, columns=estimator_classes)
        return aligned.reindex(columns=classes, fill_value=0.0).to_numpy()

    if hasattr(estimator, "decision_function"):
        decisions = np.asarray(estimator.decision_function(X))
        if decisions.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-decisions))
            probabilities = np.column_stack([1.0 - positive, positive])
        else:
            shifted = decisions - np.max(decisions, axis=1, keepdims=True)
            exponent = np.exp(shifted)
            probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
        aligned = pd.DataFrame(probabilities, columns=estimator_classes)
        return aligned.reindex(columns=classes, fill_value=0.0).to_numpy()

    predictions = np.asarray(estimator.predict(X))
    one_hot = pd.get_dummies(pd.Series(predictions), dtype=float).reindex(
        columns=list(classes),
        fill_value=0.0,
    )
    return one_hot.to_numpy()


def run_shap_for_top_models(
    *,
    final_table: pd.DataFrame,
    dataset_path: Path,
    split_manifest_path: Path,
    results_root: Path,
    report_root: Path,
    top_k: int,
    ranking_metric: str,
    background_size: int,
    explain_rows_per_class: int,
    max_display: int,
) -> pd.DataFrame:
    if final_table.empty:
        return pd.DataFrame()

    ranked = final_table.sort_values(
        by=[f"test_{ranking_metric}", "test_auroc", "test_geometric_mean"],
        ascending=False,
    ).head(top_k)
    manifest = load_split_manifest(split_manifest_path)
    resolved_dataset_path = resolve_dataset_path_from_manifest(manifest, dataset_path)
    X, y, classes = load_dataset(resolved_dataset_path, manifest.target_column)
    X_train = X.iloc[list(manifest.train_indices)]
    y_train = y.iloc[list(manifest.train_indices)]
    X_test = X.iloc[list(manifest.test_indices)]
    y_test = y.iloc[list(manifest.test_indices)]

    rows: list[dict[str, Any]] = []

    import shap

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        model_path = Path(row["model_path"])
        if not model_path.exists():
            continue
        with model_path.open("rb") as file:
            model = pickle.load(file)  # type: ignore[name-defined]

        background_frame, background_labels = _stratified_total_sample(
            X_train,
            y_train,
            max_rows=background_size,
            random_state=42,
        )
        explain_frame, explain_labels = _balanced_sample_per_class(
            X_test,
            y_test,
            rows_per_class=explain_rows_per_class,
            random_state=43,
        )

        shap_dir = report_root / "shap" / row["scope_key"] / row["primitive_key"]
        shap_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "class": background_labels.value_counts().sort_index().index,
                "background_rows": background_labels.value_counts().sort_index().values,
            }
        ).to_csv(shap_dir / "background_class_counts.csv", index=False)
        pd.DataFrame(
            {
                "class": explain_labels.value_counts().sort_index().index,
                "explained_rows": explain_labels.value_counts().sort_index().values,
            }
        ).to_csv(shap_dir / "explained_class_counts.csv", index=False)

        feature_columns = list(explain_frame.columns)
        background_transformed = model.named_steps["scale"].transform(
            model.named_steps["impute"].transform(background_frame)
        )
        explain_transformed = model.named_steps["scale"].transform(
            model.named_steps["impute"].transform(explain_frame)
        )
        classifier = model.named_steps["classifier"]

        def predict_fn(data: np.ndarray | pd.DataFrame) -> np.ndarray:
            frame = (
                pd.DataFrame(data, columns=feature_columns)
                if not isinstance(data, pd.DataFrame)
                else data.copy()
            )
            return _score_estimator_matrix(classifier, frame, classes)

        explainer = shap.Explainer(
            predict_fn,
            background_transformed,
            algorithm="partition",
            feature_names=feature_columns,
            output_names=[str(label) for label in classes],
            seed=42,
        )
        shap_values = explainer(explain_transformed)

        per_class_rows: list[pd.DataFrame] = []
        for class_idx, class_label in enumerate(classes):
            per_class_rows.append(
                pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "class": f"class_{class_label}",
                        "mean_abs_shap": np.abs(shap_values.values[:, :, class_idx]).mean(axis=0),
                        "mean_shap": shap_values.values[:, :, class_idx].mean(axis=0),
                    }
                )
            )
        multiclass_frame = pd.concat(per_class_rows, ignore_index=True)
        multiclass_frame.to_csv(shap_dir / "multiclass_mean_shap_long.csv", index=False)

        top_features = (
            multiclass_frame.groupby("feature")["mean_abs_shap"]
            .max()
            .sort_values(ascending=False)
            .head(max_display)
            .index
        )
        multiclass_mean_abs_pivot = (
            multiclass_frame[multiclass_frame["feature"].isin(top_features)]
            .pivot(index="feature", columns="class", values="mean_abs_shap")
            .fillna(0.0)
            .loc[list(top_features)[::-1]]
        )
        multiclass_mean_signed_pivot = (
            multiclass_frame[multiclass_frame["feature"].isin(top_features)]
            .pivot(index="feature", columns="class", values="mean_shap")
            .fillna(0.0)
            .loc[list(top_features)[::-1]]
        )
        multiclass_mean_abs_pivot.to_csv(
            shap_dir / "multiclass_mean_abs_shap_pivot.csv"
        )
        multiclass_mean_signed_pivot.to_csv(
            shap_dir / "multiclass_mean_signed_shap_pivot.csv"
        )
        ax = multiclass_mean_abs_pivot.plot(
            kind="barh",
            figsize=(12, max(6, 0.4 * len(multiclass_mean_abs_pivot))),
            width=0.85,
        )
        ax.set_title(f"{row['display_name']} mean(|SHAP|) by feature and class")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(shap_dir / "multiclass_mean_abs_shap.png", dpi=220, bbox_inches="tight")
        plt.close()

        rows.append(
            {
                "rank": rank,
                "launcher_key": row["launcher_key"],
                "display_name": row["display_name"],
                "scope_key": row["scope_key"],
                "primitive_key": row["primitive_key"],
                "test_rank_metric": row[f"test_{ranking_metric}"],
                "explainer_algorithm": "partition",
                "background_rows": len(background_frame),
                "explained_rows": len(explain_frame),
                "background_source": "stratified_train_split",
                "explained_source": "balanced_test_split",
                "output_dir": str(shap_dir),
            }
        )

    return pd.DataFrame(rows)


def run_reporting_workflow(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    report_root: Path = DEFAULT_REPORTS_ROOT,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
    scope_keys: set[str] | None = None,
    ranking_metric: str = DEFAULT_RANKING_METRIC,
    top_k: int = DEFAULT_TOP_K,
    skip_shap: bool = False,
    force_final_evaluation: bool = False,
    final_top_k: int | None = None,
    shap_background_size: int = DEFAULT_SHAP_BACKGROUND_SIZE,
    shap_explain_rows_per_class: int = DEFAULT_SHAP_ROWS_PER_CLASS,
    shap_max_display: int = DEFAULT_SHAP_MAX_DISPLAY,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    specs = iter_primitive_specs(scope_keys)

    status_records = [
        collect_fold_status(spec=spec, results_root=results_root) for spec in specs
    ]
    status_table = pd.DataFrame(status_records).sort_values(
        ["scope_key", "display_name"],
    )
    status_table.to_csv(report_root / "fold_status.csv", index=False)

    complete_specs = tuple(
        spec
        for spec in specs
        if status_table.loc[
            status_table["launcher_key"] == spec.launcher_key, "is_complete"
        ].any()
    )

    if status_table.empty:
        return

    cv_table = status_table.copy()
    cv_table.to_csv(report_root / "cv_metric_summary.csv", index=False)

    for spec in complete_specs:
        aggregate_cv_reports(spec=spec, results_root=results_root, report_root=report_root)

    if not complete_specs:
        return

    final_specs, final_selection = select_specs_for_final_evaluation(
        specs=complete_specs,
        cv_table=cv_table[cv_table["launcher_key"].isin([spec.launcher_key for spec in complete_specs])],
        ranking_metric=ranking_metric,
        final_top_k=final_top_k,
    )
    if final_top_k is not None:
        final_selection.to_csv(report_root / "final_evaluation_selection.csv", index=False)

    if not final_specs:
        return

    final_table, final_failures = build_final_evaluation_table(
        specs=final_specs,
        results_root=results_root,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        force=force_final_evaluation,
    )
    if not final_failures.empty:
        final_failures.to_csv(report_root / "final_evaluation_failures.csv", index=False)
    if final_table.empty:
        return
    merged_table = cv_table.merge(
        final_table,
        on=[
            "launcher_key",
            "scope_key",
            "primitive_key",
            "display_name",
            "balanced_bagging",
            "estimator_domain",
        ],
        how="inner",
    )
    merged_table = merged_table.sort_values(
        [f"test_{ranking_metric}", "test_auroc", "test_geometric_mean"],
        ascending=False,
    )
    merged_table.to_csv(report_root / "final_model_comparison.csv", index=False)
    write_stage_report_tables(final_table=final_table, output_dir=report_root)

    for metric_column in ("test_auprc", "test_auroc", "test_geometric_mean"):
        plot_metric_bars(
            table=merged_table,
            metric_column=metric_column,
            output_path=report_root / "plots" / f"{metric_column}.png",
        )

    if skip_shap:
        return

    shap_summary = run_shap_for_top_models(
        final_table=merged_table,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        results_root=results_root,
        report_root=report_root,
        top_k=top_k,
        ranking_metric=ranking_metric,
        background_size=shap_background_size,
        explain_rows_per_class=shap_explain_rows_per_class,
        max_display=shap_max_display,
    )
    if not shap_summary.empty:
        shap_summary.to_csv(report_root / "shap_top_models.csv", index=False)
