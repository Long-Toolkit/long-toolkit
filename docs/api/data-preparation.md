---
icon: lucide/database-zap
---

# Data Preparation Tools

This page of the application programming interface (API) reference documents the tools of the [Data Preparation Toolkit](../toolkits/data-preparation.md), which generate synthetic longitudinal cohorts and convert cohort files between formats. Each class docstring lists the accepted parameters and the artefact produced.

## Synthetic Data Generation

## ::: long_toolkit.data_preparation.tools.synthetic_data_generation.generators.trend_patterns.TrendPatterns
    options:
      members:
        - prepare
        - default_class_specs
        - random_class_specs
        - default_feature_specs
        - default_class_proportions

## ::: long_toolkit.data_preparation.tools.synthetic_data_generation.generators.event_shock_recovery.EventShockRecovery

## ::: long_toolkit.data_preparation.tools.synthetic_data_generation.generators.piecewise_changepoint.PiecewiseChangepoint

## ::: long_toolkit.data_preparation.tools.synthetic_data_generation.generators.missing_data_scenarios.MissingDataScenarios

## ::: long_toolkit.data_preparation.tools.synthetic_data_generation.generators.harmonisation_challenge.HarmonisationChallenge

## Data Conversion

All converters below are part of the same **Data Conversion** family and each
one exposes `prepare(...)` (`run_mode="single"` for one file, `run_mode="folder"`
for batch conversion).

| Technique key | Converter class |
| --- | --- |
| `csv_to_parquet` | `CsvToParquet` |
| `csv_to_stata` | `CsvToStata` |
| `parquet_to_csv` | `ParquetToCsv` |
| `parquet_to_stata` | `ParquetToStata` |
| `stata_to_csv` | `StataToCsv` |
| `stata_to_parquet` | `StataToParquet` |

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.csv_to_parquet.CsvToParquet
    options:
      inherited_members: true
      members:
        - prepare

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.csv_to_stata.CsvToStata
    options:
      inherited_members: true
      members:
        - prepare

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.parquet_to_csv.ParquetToCsv
    options:
      inherited_members: true
      members:
        - prepare

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.parquet_to_stata.ParquetToStata
    options:
      inherited_members: true
      members:
        - prepare

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.stata_to_csv.StataToCsv
    options:
      inherited_members: true
      members:
        - prepare

## ::: long_toolkit.data_preparation.tools.data_conversion.converters.stata_to_parquet.StataToParquet
    options:
      inherited_members: true
      members:
        - prepare
