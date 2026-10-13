---
icon: lucide/trophy
---

# Benchmarking and Modelling Toolkit

<div class="page-lead" markdown>

In a cohort study, the established analysis of an outcome is often a statistical model, such as a logistic regression on a handful of predictors. Machine learning may predict better, and longitudinal machine learning (models that exploit the repeated measures of each variable) may also use the temporal structure that standard models flatten; however, both cost complexity, tuning and interpretability.

The Benchmarking and Modelling Toolkit, the main toolkit of `Long-Toolkit`, measures whether that cost is worth paying on the dataset at hand. Its benchmarks rank many estimators at once, all against all, under one shared protocol; its modelling tools then train and evaluate one chosen estimator in depth.

</div>

## Why Benchmark Before Modelling

Choosing a model first tends to lock a study into one family of methods, whose advantage is then assumed rather than measured. A benchmark reverses that order: every candidate, from the statistical baseline to the most specialised longitudinal estimator, is evaluated on the same data with the same splits and metrics, so that the choice of model follows from evidence. One call of the application programming interface (API), or one form of the command-line interface (CLI), runs dozens of estimators and returns a ranking, a report and a notebook that replays the run.

## The Benchmarking Tools

| Tool | What it ranks | Typical use |
| --- | --- | --- |
| `BenchmarkStandardML` | Standard (non-longitudinal) estimators only | A sweep of conventional models, including the logistic-regression baseline |
| `BenchmarkLongitudinalML` | Longitudinal estimators only | A comparison of longitudinal strategies against each other |
| `BenchStandardAndLongitudinalML` | Both families, in one ranking | Does longitudinal machine learning beat standard machine learning, and do both beat the baseline? |

The three benchmarking tools share their parameters, apart from the feature-group parameters, which only apply to longitudinal estimators. Each tool lists its estimators with `technique="list_estimators"` and its metrics with `technique="list_metrics"`.

## What Is Compared

### Standard Estimators

Seven standard estimators treat every column as an independent feature, i.e., they ignore that `mood_w1` and `mood_w2` measure the same variable: `logistic_regression`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `knn` and `svm`. `logistic_regression` doubles as the statistical baseline.

### Longitudinal Estimators

Forty-seven longitudinal estimators are built on [Scikit-Longitudinal](https://scikit-longitudinal.readthedocs.io/), each named `<strategy>__<base estimator>` (e.g., `sepwav_voting__random_forest`). The strategy decides how the waves reach the base estimator. Read more in [Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python](https://doi.org/10.21105/joss.08481).

| Strategy | Mechanism | Trade-off |
| --- | --- | --- |
| `aggrfunc_mean`, `aggrfunc_median` | Summarises each feature group over its waves into one value | Simple and fast, but the temporal ordering is lost |
| `merwav_time_minus` | Merges all waves into one feature vector, discarding the wave indices | Keeps every value, but the estimator cannot tell which columns belong together |
| `merwav_time_plus` | Merges all waves whilst keeping the feature groups, for longitudinal-data-aware estimators (lexicographical trees, nested trees, lexicographical deep forest) | Uses the temporal structure directly, at a higher training cost |
| `sepwav_voting`, `sepwav_stacking_lr`, `sepwav_stacking_dt` | Trains one estimator per wave, then combines them by voting or stacking | Keeps each wave intact, but relationships across waves cannot inform the per-wave estimators |

Six strategies are combined with the seven standard base estimators, and `merwav_time_plus` with five longitudinal-data-aware ones, hence 47 keys.

### Feature Groups

Longitudinal estimators need the feature groups, i.e., the columns that measure the same variable across waves.

!!! warning "Wide format required"
    Longitudinal machine learning expects the data in wide format, i.e., one row per participant and one column per variable and wave (e.g., `mood_w1`, `mood_w2`, `mood_w3`). A long table (one row per participant and wave) must first be reshaped with `PivotLongToWide` from the [Data Preprocessing Toolkit](data-preprocessing.md).

| `feature_groups_mode` | How the groups are given | Example |
| --- | --- | --- |
| `suffix` | Inferred from a column suffix | `feature_groups_suffix="_w"` groups `mood_w1`, `mood_w2` and `mood_w3` |
| `manual` | Listed explicitly, one group per `;` | `feature_groups="mood_w1,mood_w2,mood_w3;sleep_w1,sleep_w2,sleep_w3"` |
| `preset` | Taken from a Scikit-Longitudinal preset | `feature_groups_preset="elsa"` |

Every other feature (e.g., `sex`) is non-longitudinal: `non_longitudinal_mode="auto"` infers them, whereas `"manual"` lists them in `non_longitudinal_features`. The Scikit-Longitudinal [temporal dependency tutorial](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/) details the data format.

## The Shared Protocol

| Parameter | Role | Default |
| --- | --- | --- |
| `evaluation_strategy` | `cross_validation` (stratified k-fold) or `holdout` (repeated stratified holdout) | `cross_validation` |
| `n_splits` | Number of folds, or of repeated holdout splits | `10` |
| `holdout_split` | Train/validation ratio for holdout (e.g., `80/20`, or `0.2`) | `80/20` |
| `metric_keys` | Comma-separated metrics; the first one ranks the estimators | `f1_macro` |
| `multiclass_mode` | `multiclass` or `one_vs_rest`, used only with more than two classes | `multiclass` |
| `missing_value_imputation_enabled` | Adds imputation inside each model, fitted on training data only | `false` |
| `missing_value_imputation_strategy` | `mean`, `median`, `most_frequent` or `constant` | `mean` |
| `hyperparameters_json` | One JSON object of hyperparameters, applied to every estimator that supports each key | `{}` |
| `excluded_estimators` | Comma-separated estimator keys to leave out | `none` |
| `random_seed` | Seed shared by splits and estimators | `42` |

Imputation is part of the models rather than of preprocessing on purpose: fitted inside each training split, it cannot leak validation data into the estimated performance. Fifty-nine metrics are supported, including `auroc`, `auprc` and `geometric_mean`.

## Running a Benchmark

=== "CLI"

    ```bash
    long-toolkit
    # Navigate: Machine Learning -> Benchmark Standard and Longitudinal ML
    # Or open the command palette with ":" and type: benchmark standard and longitudinal ml
    ```

    The interface prompts every parameter of the protocol, with its default and a hint, then runs the benchmark and reports where its artefacts were written.

=== "Python"

    The example below reuses `data/cohort_wide.csv` from the [First Benchmark](../tutorials/first-benchmark.md) tutorial, with a repeated holdout and training-time imputation.

    ```python
    import pandas as pd

    from long_toolkit.machine_learning import BenchStandardAndLongitudinalML

    features = [
        column
        for column in pd.read_csv("data/cohort_wide.csv").columns
        if column not in ("subject_id", "outcome_class")
    ]

    benchmark = BenchStandardAndLongitudinalML()
    roster = benchmark.fit_predict(technique="list_estimators")
    kept = {"logistic_regression", "gradient_boosting", "merwav_time_plus__lexico_gradient_boosting"}
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
        metric_keys="balanced_accuracy,auroc",
        evaluation_strategy="holdout",
        n_splits=5,
        holdout_split="80/20",
        missing_value_imputation_enabled=True,
        missing_value_imputation_strategy="median",
        random_seed=42,
        output_root_dir="outputs/holdout_benchmark",
    )
    ```

## What a Benchmark Produces

| Artefact | Content |
| --- | --- |
| `benchmark_ranking.csv` | One row per estimator, ranked by the first metric, with mean, standard deviation and per-split scores for every metric |
| `benchmark_report.txt` | The protocol, then the ranking, in plain text |
| `benchmark_summary.json` | The protocol and ranking in machine-readable form, with the path to every artefact |
| `estimators/<key>/` | For each estimator: the fitted model (`.pkl`), a classification report and a summary |
| `*_reproducibility_notebook.ipynb` | A notebook that replays the same benchmark (`generate_notebook=True` by default) |
| `pipelineprofiler_view.html` | An optional interactive view of the evaluated pipelines (`generate_pipeline_profiler_html=True`) |

## Reading a Ranking

A ranking describes one protocol on one dataset; it is not a general verdict on a family of models. Hence, three precautions apply. First, when the spread across splits exceeds the gap between two neighbouring estimators, their order is not meaningful. Second, claiming that one estimator outperforms another requires a statistical test over the per-split scores (e.g., a Wilcoxon signed-rank test), which the reported scores make possible but the ranking does not provide. Third, with imbalanced classes, `f1_macro`, `balanced_accuracy`, `auroc` or `geometric_mean` are usually more informative than `accuracy`.

A ranking in which the statistical baseline comes first is a valid outcome: it indicates that, under the evaluated protocol, machine learning does not justify its added complexity on that dataset.

## Modelling One Estimator

Once a benchmark has narrowed the candidates down, a study usually needs one model examined in depth, i.e., its scores per metric, its saved artefact and its classification report. Two tools cover this step, one estimator at a time.

| Tool | What it does |
| --- | --- |
| `StandardMachineLearning` | Evaluates one standard (non-longitudinal) estimator under cross-validation or repeated holdout, then saves the fitted model, its classification report and a reproducibility notebook |
| `LongitudinalMachineLearning` | Does the same for one longitudinal estimator, with the feature-group modes of the benchmarks (`suffix`, `manual` or `preset`) |

Both tools accept the same protocol parameters as the benchmarks (metrics, splits, multiclass mode, training-time imputation, hyperparameters), so that a model examined here is evaluated exactly as it was ranked. The estimator keys are those of the benchmark roster, listed with `technique="list_estimators"`.

The example below reuses `data/cohort_wide.csv` from the [First Benchmark](../tutorials/first-benchmark.md) tutorial, and evaluates the statistical baseline and one longitudinal estimator.

```python
import pandas as pd

from long_toolkit.machine_learning import (
    LongitudinalMachineLearning,
    StandardMachineLearning,
)

features = ",".join(
    column
    for column in pd.read_csv("data/cohort_wide.csv").columns
    if column not in ("subject_id", "outcome_class")
)

baseline = StandardMachineLearning().fit_predict(
    technique="run_experiment",
    input_path="data/cohort_wide.csv",
    target_column="outcome_class",
    feature_columns=features,
    estimator_key="logistic_regression",
    metric_keys="f1_macro,auroc",
    n_splits=5,
    output_dir="outputs/logistic_regression",
)
longitudinal = LongitudinalMachineLearning().fit_predict(
    technique="run_experiment",
    input_path="data/cohort_wide.csv",
    target_column="outcome_class",
    feature_columns=features,
    feature_groups_mode="suffix",
    feature_groups_suffix="_w",
    non_longitudinal_mode="auto",
    estimator_key="merwav_time_plus__lexico_random_forest",
    metric_keys="f1_macro,auroc",
    n_splits=5,
    output_dir="outputs/lexico_random_forest",
)
print(baseline["mean_score"], longitudinal["mean_score"])

```

Each run returns its scores and the path of the saved model (`model_path`), which the [Explainability Toolkit](explainability.md) can explain.

In the command-line interface, the benchmarking and modelling tools are all in the `Machine Learning` stage.

## API Reference

The full parameter references are in the [Benchmarks](../api/benchmarks.md), [Standard Machine Learning](../api/standard-machine-learning.md) and [Longitudinal Machine Learning](../api/longitudinal-machine-learning.md) pages of the API reference.
