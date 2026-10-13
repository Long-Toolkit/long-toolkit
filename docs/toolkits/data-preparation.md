---
icon: lucide/database-zap
---

# Data Preparation Toolkit

<div class="page-lead" markdown>

Cohort releases rarely arrive in the format an analysis needs: many ship each wave as a folder of Stata (`.dta`) files, whereas most Python tools expect CSV or Parquet tables. The Data Preparation Toolkit (Stage A) produces usable tables, either by converting real cohort files or by generating synthetic cohorts with known properties.

</div>

## Synthetic Longitudinal Cohorts

A synthetic cohort is useful whenever the real data are restricted, or whenever a method must be checked against a pattern that is known in advance (e.g., an imputation method tested on a known missingness mechanism). Every generator returns a long table (one row per participant and wave, with `subject_id` and `wave` columns) and is seeded for reproducibility.

| Generator | Pattern it produces | Typical use |
| --- | --- | --- |
| `TrendPatterns` | Participants sampled into latent trend classes (e.g., persistently high, increasing), with class-specific intercepts and slopes | Checking whether models recover class-level trends |
| `EventShockRecovery` | A baseline trend, an acute shock at a chosen wave, then an exponential recovery | Testing sensitivity to abrupt deterioration followed by partial recovery |
| `PiecewiseChangepoint` | One structural break, where the slope changes at a chosen wave whilst the values stay continuous | Testing whether models capture phase changes over time |
| `MissingDataScenarios` | A complete panel, then missingness injected as MCAR, MAR, dropout (monotone) or a mixture | Stress-testing imputation and robustness under controlled missingness |
| `HarmonisationChallenge` | Clean categorical labels (e.g., `site`, `sex`) corrupted by typos, casing and abbreviations | Testing category harmonisation |

Note that synthetic data are a controlled test bed, not a substitute for a real cohort: their trends and noise follow the generator's assumptions, so that a result obtained on them indicates what a method can recover, not how it will behave on real participants.

## File Conversion

Six converters move tables between CSV, Parquet and Stata, each in `single` mode (one file) or `folder` mode (every matching file of a folder, optionally with subfolders): `CsvToParquet`, `CsvToStata`, `ParquetToCsv`, `ParquetToStata`, `StataToCsv` and `StataToParquet`.

## Example

The example below generates a cohort with mixed missingness, then stores it as Parquet.

```python
from pathlib import Path

from long_toolkit.data_preparation import CsvToParquet, MissingDataScenarios

Path("data").mkdir(exist_ok=True)

cohort = MissingDataScenarios().prepare(
    n_samples=400,
    n_waves=5,
    random_state=7,
    feature_cols=["mood_score", "sleep_score"],
    mechanism="mixed",
    missing_rate=0.20,
    dropout_rate=0.10,
    mar_strength=1.0,
)
cohort.to_csv("data/cohort_with_missing.csv", index=False)

CsvToParquet().prepare(
    run_mode="single",
    input_path="data/cohort_with_missing.csv",
    output_path="data/cohort_with_missing.parquet",
)
```

In the command-line interface (CLI), the same tools are in the `Data Preparation` stage, under `Synthetic Data Generation` and `Data Conversion`.

## API Reference

The full parameter reference is in the [Data Preparation application programming interface (API) reference](../api/data-preparation.md).
