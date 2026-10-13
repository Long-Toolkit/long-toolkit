---
icon: lucide/scan-search
---

# Explainability Tools

This page of the application programming interface (API) reference documents the tool of the [Explainability Toolkit](../toolkits/explainability.md), which computes SHAP values for a saved model (e.g., the best estimator of a benchmark). The class docstring lists the required inputs, the explanation options and the artefacts written.

!!! note
    `run_analysis` is the underlying explainability mode. In Python, `SHAPAnalysis.fit_explain(...)` (or `fit(...)` then `explain(...)`) defaults to this mode.

## ::: long_toolkit.explainability.tools.shap_analysis.run.SHAPAnalysis
