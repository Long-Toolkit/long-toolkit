---
icon: lucide/braces
---

# API Tutorial

This tutorial covers the Python application programming interface (API) of `Long-Toolkit`, for workflows that live in scripts or Jupyter notebooks. It chains one tool of each toolkit on a synthetic cohort, so that every block below can be run in order from an empty working directory.

The remainder of this tutorial is structured as follows. Section A shows the imports. Section B generates synthetic cohorts, Section C preprocesses one of them, Sections D and E train a standard and a longitudinal model, and Section F explains the longitudinal model. Finally, Section G points to the benchmark, which compares many models at once.

## A) Importing the Toolkits

Each toolkit is a subpackage, and every tool is importable from it directly. Imports are lazy (i.e., loaded on first use), so that heavy dependencies such as SHAP or XGBoost are only imported when a tool needs them, and `import long_toolkit` stays fast.

```python
from long_toolkit import data_preparation, data_preprocessing, explainability, machine_learning

from long_toolkit.data_preparation import TrendPatterns
from long_toolkit.data_preprocessing import PivotLongToWide
from long_toolkit.explainability import SHAPAnalysis
from long_toolkit.machine_learning import BenchStandardAndLongitudinalML, LongitudinalMachineLearning
```

## B) Generating Synthetic Cohorts

API references: [`TrendPatterns`, `EventShockRecovery`, `MissingDataScenarios`](../api/data-preparation.md)

```python
from pathlib import Path

from long_toolkit.data_preparation import EventShockRecovery, MissingDataScenarios, TrendPatterns

Path("data").mkdir(exist_ok=True)

TrendPatterns(n_samples=400, n_waves=5, random_state=7).prepare().to_csv(
    "data/synthetic_trend_patterns.csv", index=False
)

EventShockRecovery().prepare(
    n_samples=400,
    n_waves=5,
    random_state=7,
    feature_cols=["mood_score"],
    shock_wave=3,
    shock_mean=3.5,
    recovery_rate=0.9,
    noise_sd=0.8,
).to_csv("data/synthetic_event_shock.csv", index=False)

MissingDataScenarios().prepare(
    n_samples=400,
    n_waves=5,
    random_state=7,
    feature_cols=["mood_score", "sleep_score", "anxiety_score"],
    mechanism="mixed",
    missing_rate=0.20,
    dropout_rate=0.10,
    mar_strength=1.0,
).to_csv("data/synthetic_with_missing.csv", index=False)
```

## C) Preprocessing a Cohort

API references: [`ShowTable`, `CleanDataset`, `MissingImputation`, `PivotLongToWide`, `AggregateLongToCrossSectional`](../api/data-preprocessing.md)

The block below inspects the cohort with missing values, cleans and imputes it, removes the two bookkeeping columns written by the generator (`missingness_mechanism` and `missing_feature_count`), then derives two shapes of the same data: a wide table for longitudinal models, and a cross-sectional table (one summarised row per participant) for standard models.

```python
from long_toolkit.data_preprocessing import (
    AggregateLongToCrossSectional,
    CleanDataset,
    MissingImputation,
    PivotLongToWide,
    RemoveColumns,
    ShowTable,
)

ShowTable().fit_preprocess(
    input_path="data/synthetic_with_missing.csv",
    output_html="data/synthetic_with_missing_report.html",
    open_browser=False,
)
CleanDataset().fit_preprocess(
    input_path="data/synthetic_with_missing.csv",
    output_path="data/long_clean.csv",
)
MissingImputation().fit_preprocess(
    technique="median_imputation",
    input_path="data/long_clean.csv",
    output_path="data/long_imputed.csv",
)
RemoveColumns().fit_preprocess(
    input_path="data/long_imputed.csv",
    output_path="data/long_imputed_tidy.csv",
    columns=["missingness_mechanism", "missing_feature_count"],
)
PivotLongToWide().fit_preprocess(
    input_path="data/long_imputed_tidy.csv",
    output_path="data/wide.csv",
    id_cols=["subject_id"],
    time_col="wave",
    longitudinal_columns=["mood_score", "sleep_score", "anxiety_score"],
    time_label="w",
    separator="_",
)
AggregateLongToCrossSectional().fit_preprocess(
    input_path="data/long_imputed_tidy.csv",
    output_path="data/cross_sectional.csv",
    subject_id_col="subject_id",
    numeric_columns=["mood_score", "sleep_score", "anxiety_score"],
    numeric_agg="mean",
)
```

Note that the imputation above is fitted on the whole cohort, which is convenient for exploration; for model evaluation, the training-time imputation of the modelling tools avoids leaking validation data into the imputation (see [Data Preprocessing Toolkit](../toolkits/data-preprocessing.md#imputation-here-or-inside-the-models)).

## D) Training a Standard Model

API reference: [`StandardMachineLearning`](../api/standard-machine-learning.md)

A synthetic cohort has no outcome of its own, so the block below defines one (an above-median mean anxiety score) and predicts it from the mean mood and sleep scores.

```python
import pandas as pd

from long_toolkit.machine_learning import StandardMachineLearning

table = pd.read_csv("data/cross_sectional.csv")
table["outcome_class"] = (table["anxiety_score"] > table["anxiety_score"].median()).astype(int)
table.drop(columns=["anxiety_score"]).to_csv("data/cross_sectional_with_outcome.csv", index=False)

result = StandardMachineLearning().fit_predict(
    technique="run_experiment",
    input_path="data/cross_sectional_with_outcome.csv",
    target_column="outcome_class",
    feature_columns="mood_score,sleep_score",
    estimator_key="random_forest",
    metric_keys="f1_macro,accuracy",
    evaluation_strategy="cross_validation",
    n_splits=5,
    random_seed=42,
    output_dir="outputs/standard_ml",
)
print(result["mean_score"], result["model_path"])
```

## E) Training a Longitudinal Model

API reference: [`LongitudinalMachineLearning`](../api/longitudinal-machine-learning.md)

The longitudinal model works on the wide table, where each measure keeps one column per wave. Feature groups are inferred from the `_w` suffix, and every remaining numeric feature (here, `age_baseline`) is treated as non-longitudinal.

```python
import pandas as pd

from long_toolkit.machine_learning import LongitudinalMachineLearning

wide = pd.read_csv("data/wide.csv")
wide["outcome_class"] = (wide["anxiety_score_w5"] > wide["anxiety_score_w5"].median()).astype(int)
wide = wide.drop(columns=["anxiety_score_w5"])
wide.to_csv("data/wide_with_outcome.csv", index=False)
features = [
    column
    for column in wide.select_dtypes("number").columns
    if column not in ("subject_id", "outcome_class")
]

result = LongitudinalMachineLearning().fit_predict(
    technique="run_experiment",
    input_path="data/wide_with_outcome.csv",
    target_column="outcome_class",
    feature_columns=",".join(features),
    feature_groups_mode="suffix",
    feature_groups_suffix="_w",
    non_longitudinal_mode="auto",
    estimator_key="merwav_time_plus__lexico_random_forest",
    metric_keys="f1_macro,accuracy",
    evaluation_strategy="cross_validation",
    n_splits=5,
    random_seed=42,
    output_dir="outputs/longitudinal_ml",
)
print(result["mean_score"], result["model_path"])
```

Manual and preset feature groups are also supported, as follows: `feature_groups_mode="manual"` with `feature_groups="mood_score_w1,mood_score_w2;sleep_score_w1,sleep_score_w2"` (or with column indices, such as `"[[0,1],[2,3]]"`), and `feature_groups_mode="preset"` with a Scikit-Longitudinal preset such as `feature_groups_preset="elsa"`.

## F) Explaining a Model

API reference: [`SHAPAnalysis`](../api/explainability.md)

The longitudinal model of Section E is saved under `outputs/longitudinal_ml/` (its path is also returned as `result["model_path"]`). `SHAPAnalysis` explains its predictions with SHAP values, on the feature columns it was trained on.

```python
from pathlib import Path

import pandas as pd

from long_toolkit.explainability import SHAPAnalysis

model_path = sorted(Path("outputs/longitudinal_ml").glob("*_model.pkl"))[-1]
features = [
    column
    for column in pd.read_csv("data/wide_with_outcome.csv").select_dtypes("number").columns
    if column not in ("subject_id", "outcome_class")
]

explanation = SHAPAnalysis().fit_explain(
    model_path=str(model_path),
    input_path="data/wide_with_outcome.csv",
    target_column="outcome_class",
    feature_columns=",".join(features),
    max_rows_to_explain=100,
    background_rows=50,
    output_dir="outputs/shap_longitudinal_ml",
)
print(explanation["importance_csv"])
```

## G) Benchmarking Instead of Choosing

Sections D and E each evaluate one model chosen in advance. The [Benchmarking and Modelling Toolkit](../toolkits/benchmarking-and-modelling.md) evaluates dozens of standard and longitudinal estimators under one protocol instead, and the [First Benchmark](first-benchmark.md) shows a complete run in about thirty lines.
