---
icon: lucide/filter
---

# Data Preprocessing Toolkit

<div class="page-lead" markdown>

Between a prepared cohort table and a model lie many small decisions: which columns are unusable, how missing waves are handled, whether the same category is spelt three different ways across waves, and whether the table is long (one row per participant and wave) or wide (one row per participant). The Data Preprocessing Toolkit (Stage B of the hub) gathers these operations, each as one tool that reads a table and writes a new one.

</div>

## The Tools

| Tool | What it does | Main options |
| --- | --- | --- |
| `CleanDataset` | Cleans a table with the [skrub](https://skrub-data.org/) `Cleaner`: drops mostly-empty, constant or all-unique columns, parses dates, and normalises types | `drop_null_fraction`, `drop_if_constant`, `drop_if_unique`, `datetime_format` |
| `MissingImputation` | Fills missing values with a mean, median, most-frequent or constant strategy, or with multiple imputation by chained equations (MICE, through `miceforest`) | `technique`, MICE iterations and seed |
| `HarmoniseCategories` | Merges noisy spellings of the same category (e.g., `Edinburgh`, `edinburgh `, `Edinbrugh`) with skrub's `deduplicate`, and saves the mapping | `target_columns`, `n_clusters`, `mapping_path` |
| `PivotLongToWide` | Reshapes a long table into a wide one, naming columns `<feature><separator><label><wave>` (e.g., `mood_score_w1`) | `id_cols`, `time_col`, `longitudinal_columns`, `time_label` |
| `AggregateLongToCrossSectional` | Collapses repeated rows into one row per participant (numeric: mean, median, min, max or standard deviation; categorical: mode, first or last) | `numeric_agg`, `categorical_agg` |
| `RenameFeature`, `RemoveColumns` | Renames one column, or removes a list of columns | `feature_name`, `columns` |
| `ShowTable` | Writes an interactive HTML report of a table (skrub `TableReport`), for inspection before and after each step | `output_html`, `open_browser` |

Note the trade-off between the two reshaping tools: `PivotLongToWide` keeps every wave as its own column (which longitudinal estimators need), whereas `AggregateLongToCrossSectional` summarises the waves into one value per participant, which suits standard models but discards the temporal ordering.

## Imputation Here, or Inside the Models?

`MissingImputation` is fitted on the whole table it receives. This is appropriate for exploration and for producing a complete table to share, but it has the disadvantage that, when the imputed table is later used to evaluate a model, the imputation has already seen the validation participants, so that information leaks into the estimated performance. For evaluation, the modelling and benchmarking tools therefore offer `missing_value_imputation_enabled`, which fits the imputation inside each model, on the training data only (see the [shared protocol](benchmarking-and-modelling.md#the-shared-protocol)).

## Example

The example below inspects, cleans and imputes the synthetic cohort produced on the [Data Preparation](data-preparation.md) page, then reshapes it for the benchmark. Note that the generator also writes two bookkeeping columns (`missingness_mechanism` and `missing_feature_count`), which are removed before the pivot, since `missing_feature_count` varies across waves without being a measure of interest.

```python
from long_toolkit.data_preprocessing import (
    CleanDataset,
    MissingImputation,
    PivotLongToWide,
    RemoveColumns,
    ShowTable,
)

ShowTable().fit_preprocess(
    input_path="data/cohort_with_missing.csv",
    output_html="data/cohort_report.html",
    open_browser=False,
)
CleanDataset().fit_preprocess(
    input_path="data/cohort_with_missing.csv",
    output_path="data/cohort_clean.csv",
)
MissingImputation().fit_preprocess(
    technique="median_imputation",
    input_path="data/cohort_clean.csv",
    output_path="data/cohort_imputed.csv",
)
RemoveColumns().fit_preprocess(
    input_path="data/cohort_imputed.csv",
    output_path="data/cohort_imputed_tidy.csv",
    columns=["missingness_mechanism", "missing_feature_count"],
)
PivotLongToWide().fit_preprocess(
    input_path="data/cohort_imputed_tidy.csv",
    output_path="data/cohort_imputed_wide.csv",
    id_cols=["subject_id"],
    time_col="wave",
    longitudinal_columns=["mood_score", "sleep_score"],
    time_label="w",
    separator="_",
)
```

In the command-line interface (CLI), the same tools are in the `Data Preprocessing` stage.

## API Reference

The full parameter reference is in the [Data Preprocessing application programming interface (API) reference](../api/data-preprocessing.md).
