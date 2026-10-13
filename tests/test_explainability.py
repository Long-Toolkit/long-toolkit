from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import long_toolkit.machine_learning as machine_learning
from long_toolkit.bridge.operations import execute_operation
from long_toolkit.explainability import ExplainabilityTool, SHAPAnalysis
from long_toolkit.explainability.support.inputs import parse_class_index
from long_toolkit.utils.errors import InputValidationError, OperationNotFoundError

SHAP_ARTEFACTS = {
    "importance_csv",
    "output_dir",
    "report_path",
    "summary_bar_plot",
    "summary_dot_plot",
}


@pytest.fixture
def wide_cohort(tmp_path):
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "subject_id": np.arange(60),
            "mood_w1": rng.normal(size=60),
            "mood_w2": rng.normal(size=60),
            "sleep_w1": rng.normal(size=60),
            "sleep_w2": rng.normal(size=60),
        }
    )
    data["outcome_class"] = (data["mood_w1"] + data["mood_w2"] > 0).astype(int)
    path = tmp_path / "cohort_wide.csv"
    data.to_csv(path, index=False)
    return path


def test_shap_analysis_lives_in_the_explainability_toolkit() -> None:
    assert issubclass(SHAPAnalysis, ExplainabilityTool)
    for verb in ("fit", "explain", "fit_explain"):
        assert callable(getattr(SHAPAnalysis, verb))
    for verb in ("predict", "predict_proba", "fit_predict"):
        assert not hasattr(SHAPAnalysis, verb)
    assert "SHAPAnalysis" not in machine_learning.__all__
    with pytest.raises(AttributeError):
        machine_learning.SHAPAnalysis  # noqa: B018


def test_bridge_exposes_explainability_operations_only() -> None:
    operations = execute_operation("core.catalog", {})["operations"]
    assert "explainability.shap_analysis.catalog" in operations
    assert "explainability.shap_analysis.run" in operations
    assert not any(name.startswith("machine_learning.shap") for name in operations)
    with pytest.raises(OperationNotFoundError):
        execute_operation("machine_learning.shap_analysis.run", {})

    techniques = execute_operation("explainability.shap_analysis.catalog", {})[
        "techniques"
    ]
    assert [technique["id"] for technique in techniques] == ["run_analysis"]
    keys = {parameter["key"] for parameter in techniques[0]["parameters"]}
    assert {"model_path", "input_path", "feature_columns", "class_index"} <= keys


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("auto", None), ("", None), ("none", None), ("0", 0), (" 2 ", 2)],
)
def test_parse_class_index(raw: str, expected: int | None) -> None:
    assert parse_class_index(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "first"])
def test_parse_class_index_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(InputValidationError):
        parse_class_index(raw)


def test_fit_explain_writes_artefacts_for_a_saved_model(tmp_path, wide_cohort) -> None:
    data = pd.read_csv(wide_cohort)
    features = ["mood_w1", "mood_w2", "sleep_w1", "sleep_w2"]
    model = LogisticRegression().fit(data[features], data["outcome_class"])
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(model))

    result = SHAPAnalysis().fit_explain(
        model_path=str(model_path),
        input_path=str(wide_cohort),
        target_column="outcome_class",
        feature_columns=",".join(features),
        max_rows_to_explain=20,
        background_rows=10,
        output_dir=str(tmp_path / "shap"),
    )

    assert set(result) == SHAP_ARTEFACTS
    importance = pd.read_csv(result["importance_csv"])
    assert set(importance["feature"]) == set(features)


def test_explain_requires_a_configuration() -> None:
    with pytest.raises(InputValidationError, match="No configuration provided"):
        SHAPAnalysis().explain()


def test_explainability_follows_a_benchmark(tmp_path, wide_cohort) -> None:
    from long_toolkit.machine_learning import BenchmarkStandardML

    features = "mood_w1,mood_w2,sleep_w1,sleep_w2"
    benchmark = BenchmarkStandardML()
    roster = benchmark.fit_predict(technique="list_estimators")
    excluded = [
        estimator["key"]
        for estimator in roster["estimators"]
        if estimator["key"] != "logistic_regression"
    ]
    result = benchmark.fit_predict(
        technique="run_benchmark",
        input_path=str(wide_cohort),
        target_column="outcome_class",
        feature_columns=features,
        excluded_estimators=",".join(excluded),
        metric_keys="accuracy",
        n_splits=2,
        generate_notebook=False,
        output_root_dir=str(tmp_path / "benchmark"),
    )
    assert result["mode"] == "multiclass"
    best = result["ranked_estimators"][0]
    assert best["estimator_key"] == "logistic_regression"

    explanation = SHAPAnalysis().fit_explain(
        model_path=best["model_artifact"],
        input_path=str(wide_cohort),
        target_column="outcome_class",
        feature_columns=features,
        max_rows_to_explain=20,
        background_rows=10,
        output_dir=str(tmp_path / "shap"),
    )
    assert set(explanation) == SHAP_ARTEFACTS
