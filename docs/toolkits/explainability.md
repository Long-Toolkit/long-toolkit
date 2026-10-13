---
icon: lucide/scan-search
---

# Explainability Toolkit

<div class="page-lead" markdown>

A ranking tells which model predicts best, but not why. The Explainability Toolkit (Stage D of the hub) explains the predictions of a saved model, i.e., which features, and which waves of each feature, drive them. The model can be the best estimator of a benchmark, or any model trained by the [Benchmarking and Modelling Toolkit](benchmarking-and-modelling.md).

</div>

## The Tool

| Tool | What it does |
| --- | --- |
| `SHAPAnalysis` | Loads a saved model and computes SHAP values (the contribution of each feature to each prediction), written as a feature-importance table, summary plots and a text report |

`SHAPAnalysis` reads any model saved by `Long-Toolkit`: every benchmark saves one fitted model per ranked estimator (the `model_artifact` of each entry in `ranked_estimators`), and each modelling run returns the path of its model (`model_path`). The feature columns must be those the model was trained on.

## Explaining the Best Model of a Benchmark

=== "CLI"

    ```bash
    long-toolkit
    # Navigate: Explainability -> SHAP Analysis
    # Or open the command palette with ":" and type: shap analysis
    ```

    The interface prompts the path of the saved model, the dataset and the explanation options, then reports where its artefacts were written.

=== "Python"

    The example below explains the estimator ranked first by the [First Benchmark](../tutorials/first-benchmark.md) tutorial, from its `benchmark_summary.json`.

    ```python
    import json
    from pathlib import Path

    import pandas as pd

    from long_toolkit.explainability import SHAPAnalysis

    summary_path = sorted(Path("outputs/first_benchmark").rglob("benchmark_summary.json"))[-1]
    best = json.loads(summary_path.read_text())["ranked_estimators"][0]
    features = ",".join(
        column
        for column in pd.read_csv("data/cohort_wide.csv").columns
        if column not in ("subject_id", "outcome_class")
    )

    result = SHAPAnalysis().fit_explain(
        model_path=best["model_artifact"],
        input_path="data/cohort_wide.csv",
        target_column="outcome_class",
        feature_columns=features,
        max_rows_to_explain=100,
        background_rows=50,
        output_dir=f"outputs/shap_{best['estimator_key']}",
    )
    print(best["estimator_key"], result["report_path"])
    ```

    In Python, `BenchStandardAndLongitudinalML` and `BenchmarkLongitudinalML` also return this ranking, with the same `model_artifact` paths, in `result["runs"][0]["ranked_estimators"]` (and `BenchmarkStandardML` in `result["ranked_estimators"]`, unless `multiclass_mode="one_vs_rest"`).

## What an Explanation Produces

| Artefact | Content |
| --- | --- |
| `shap_feature_importance.csv` | One row per feature, ranked by mean absolute SHAP value |
| `shap_summary_bar.png` | The mean absolute contribution of each feature |
| `shap_summary_dot.png` | The distribution of signed contributions, one dot per explained row and feature |
| `shap_analysis_report.txt` | The settings of the run, the paths of the artefacts and the top features |

## Reading an Explanation

SHAP values explain what a model relies on, not what causes the outcome. Hence, two precautions apply. First, with correlated waves (e.g., `mood_score_w2` and `mood_score_w3`), the importance can be shared between them in ways that do not reflect any causal ordering. Second, explaining several estimators of the same benchmark, rather than only the first one, shows whether a feature matters to the models in general or only to one of them.

## API Reference

The full parameter reference is in the [Explainability](../api/explainability.md) page of the application programming interface (API) reference.
