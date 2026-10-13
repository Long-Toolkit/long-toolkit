---
icon: lucide/filter
---

# Data Preprocessing Tools

This page of the application programming interface (API) reference documents the tools of the [Data Preprocessing Toolkit](../toolkits/data-preprocessing.md), which clean, impute, harmonise, reshape and inspect longitudinal tables. Each class docstring lists the accepted parameters, the expected input and the artefact written at each step.

!!! tip "Execution flow"
    `fit(...)` validates and stores configuration, `preprocess(...)` runs the transformation, and `fit_preprocess(...)` executes both in sequence.

## ::: long_toolkit.data_preprocessing.tools.clean_dataset.run.CleanDataset

## ::: long_toolkit.data_preprocessing.tools.missing_imputation.run.MissingImputation

## ::: long_toolkit.data_preprocessing.tools.harmonise_categories.run.HarmoniseCategories

## ::: long_toolkit.data_preprocessing.tools.show_table.run.ShowTable

## ::: long_toolkit.data_preprocessing.tools.aggregate_long_to_cross_sectional.run.AggregateLongToCrossSectional

## ::: long_toolkit.data_preprocessing.tools.rename_feature.run.RenameFeature

## ::: long_toolkit.data_preprocessing.tools.pivot_long_to_wide.run.PivotLongToWide

## ::: long_toolkit.data_preprocessing.tools.remove_columns.run.RemoveColumns
