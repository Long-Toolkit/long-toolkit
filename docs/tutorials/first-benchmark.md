---
icon: lucide/flag
---

# First Benchmark

This tutorial runs the main use case of `Long-Toolkit` end to end on a synthetic cohort: it generates repeated measures, reshapes them into one row per participant, defines an outcome, and ranks standard (non-longitudinal) and longitudinal machine learning side by side. The whole script runs in a few seconds, so that the reader can focus on what each step produces rather than on waiting for it.

The remainder of this page is structured as follows. The first section lists the script, step by step. The second section describes the artefacts the benchmark writes, and the third section explains how to read the resulting ranking without over-interpreting it.

## The Script, Step by Step

!!! note "Before running"
    Install the Python package first (see [Installation](../getting-started/index.md#installation)), then run the script from any empty working directory: it creates `data/` and `outputs/` next to itself.

```python
from pathlib import Path

import numpy as np
import pandas as pd

from long_toolkit.data_preparation import TrendFeatureSpec, TrendPatterns
from long_toolkit.data_preprocessing import PivotLongToWide
from long_toolkit.machine_learning import BenchStandardAndLongitudinalML

Path("data").mkdir(exist_ok=True)

# 1. A synthetic cohort in long format: 300 participants, 4 waves, 2 repeated measures.
generator = TrendPatterns(n_samples=300, n_waves=4, random_state=7)
classes = generator.default_class_specs()
cohort = generator.prepare(
    feature_specs=[
        TrendFeatureSpec(name="mood_score", class_specs=classes),
        TrendFeatureSpec(name="sleep_score", class_specs=classes),
    ]
)
cohort.to_csv("data/cohort_long.csv", index=False)

# 2. Long to wide: one row per participant, one column per measure and wave.
PivotLongToWide().fit_preprocess(
    input_path="data/cohort_long.csv",
    output_path="data/cohort_wide.csv",
    id_cols=["subject_id"],
    time_col="wave",
    longitudinal_columns=["mood_score", "sleep_score"],
    time_label="w",
    separator="_",
)

# 3. Outcome: an above-median mood score at the last wave, predicted from earlier waves.
wide = pd.read_csv("data/cohort_wide.csv")
wide["outcome_class"] = (wide["mood_score_w4"] > wide["mood_score_w4"].median()).astype(int)
wide["sex"] = np.random.default_rng(7).integers(0, 2, size=len(wide))
wide = wide.drop(columns=["mood_score_w4"])
wide.to_csv("data/cohort_wide.csv", index=False)
features = [column for column in wide.columns if column not in ("subject_id", "outcome_class")]

# 4. Benchmark a few standard and longitudinal estimators under one protocol.
benchmark = BenchStandardAndLongitudinalML()
roster = benchmark.fit_predict(technique="list_estimators")
kept = {
    "logistic_regression",
    "random_forest",
    "aggrfunc_mean__logistic_regression",
    "merwav_time_plus__lexico_random_forest",
    "sepwav_voting__random_forest",
}
excluded = [
    estimator["key"]
    for group in ("standard_estimators", "longitudinal_estimators")
    for estimator in roster[group]
    if estimator["key"] not in kept
]

benchmark.fit_predict(
    technique="run_benchmark",
    input_path="data/cohort_wide.csv",
    target_column="outcome_class",
    feature_columns=",".join(features),
    feature_groups_mode="suffix",
    feature_groups_suffix="_w",
    non_longitudinal_mode="auto",
    excluded_estimators=",".join(excluded),
    metric_keys="f1_macro,accuracy",
    evaluation_strategy="cross_validation",
    n_splits=5,
    random_seed=42,
    output_root_dir="outputs/first_benchmark",
)

# 5. The ranking, best first.
ranking = pd.read_csv(next(Path("outputs/first_benchmark").glob("*/benchmark_ranking.csv")))
print(ranking[["rank", "estimator_name", "mean_score", "std_score"]])
```

Steps 1 and 2 belong to the [Data Preparation](../toolkits/data-preparation.md) and [Data Preprocessing](../toolkits/data-preprocessing.md) toolkits. `TrendPatterns` samples each participant into a latent trend class (e.g., increasing), draws one value per wave around that class trend, and returns a long table, i.e., one row per participant and wave. `PivotLongToWide` then produces the wide table that longitudinal estimators expect, with columns such as `mood_score_w1` to `mood_score_w3`.

Step 3 is the only hand-written step, since the outcome is a modelling decision rather than a tool: here the models must predict whether the last-wave mood score lies above the median, from the earlier waves only, together with one static (non-longitudinal) covariate, `sex`. Note that `mood_score_w4` is removed from the predictors, as keeping it would leak the outcome into the features.

Step 4 is the [Benchmarking and Modelling Toolkit](../toolkits/benchmarking-and-modelling.md) proper. With `feature_groups_mode="suffix"`, the tool infers the feature groups (the columns that measure the same variable across waves) from the `_w` suffix, and with `non_longitudinal_mode="auto"` every remaining feature (here, `sex`) is treated as non-longitudinal. The roster is restricted to five estimators for speed, including `logistic_regression`, which plays the role of the statistical baseline that a study would typically rely on.

## What the Benchmark Writes

Every run writes one timestamped directory under `output_root_dir`, together with a reproducibility notebook, as follows.

| Artefact | Content |
| --- | --- |
| `benchmark_ranking.csv` | One row per estimator, ranked by the first metric of `metric_keys`, with mean, standard deviation and per-fold scores |
| `benchmark_report.txt` | The same ranking in plain text, preceded by the protocol (metrics, splits, seed, imputation) |
| `benchmark_summary.json` | The protocol and ranking in machine-readable form, with paths to every artefact |
| `estimators/<key>/` | For each estimator: the fitted model (`.pkl`), a classification report and a summary |
| `*_reproducibility_notebook.ipynb` | A notebook that reruns the same benchmark, step by step |

## Reading the Ranking

On our run, the ranking placed `logistic_regression` first (mean macro F1 of about 0.80 across five folds), ahead of every machine-learning alternative, with standard deviations across folds of about 0.06. The exact values may differ slightly across platforms and library versions; what matters is the pattern, which is that the statistical baseline was not beaten on this cohort.

Such an outcome is a legitimate answer, not a failed experiment. The synthetic trends are close to linear, so a linear model captures them well, whereas more flexible models mostly add variance; on real cohorts, with non-linear effects and interactions across waves, the ranking may well change. Note also that the ranking alone does not establish that one estimator is significantly better than another, since the fold-level scores overlap. The per-fold scores in `benchmark_ranking.csv` are reported precisely so that a statistical test (e.g., a Wilcoxon signed-rank test across folds) can be run before any claim is made.

In summary, a first benchmark costs a few lines of code and returns a defensible starting point: either machine learning earns its added complexity on the dataset at hand, or the established statistical analysis remains the better choice.
