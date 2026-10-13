---
icon: lucide/layout-grid
---

# Toolkits

Each toolkit groups the tools of one stage of a longitudinal analysis. They chain as follows, although each one can also be used on its own, since every tool reads and writes files on disk (CSV for most tools, with dedicated converters for Parquet and Stata).

```mermaid
flowchart LR
    A["Data Preparation<br/>(Stage A)"] --> B["Data Preprocessing<br/>(Stage B)"]
    B --> C["Benchmarking and Modelling<br/>(Stage C)"]
    C -- "saved models" --> D["Explainability<br/>(Stage D)"]
```

<div class="toolkit-grid">
  <a class="toolkit-card" href="data-preparation/">
    <span class="toolkit-card__label">Stage A</span>
    <span class="toolkit-card__title">Data Preparation</span>
    <span class="toolkit-card__desc">Generation of synthetic longitudinal cohorts with known trends, shocks or missingness, and conversion of files between CSV, Parquet and Stata.</span>
    <span class="toolkit-card__tools"><code>TrendPatterns</code><code>MissingDataScenarios</code><code>StataToCsv</code></span>
  </a>
  <a class="toolkit-card" href="data-preprocessing/">
    <span class="toolkit-card__label">Stage B</span>
    <span class="toolkit-card__title">Data Preprocessing</span>
    <span class="toolkit-card__desc">Cleaning, imputation, category harmonisation, and long to wide (or cross-sectional) reshaping of repeated measures.</span>
    <span class="toolkit-card__tools"><code>CleanDataset</code><code>MissingImputation</code><code>PivotLongToWide</code></span>
  </a>
  <a class="toolkit-card toolkit-card--wide toolkit-card--main" href="benchmarking-and-modelling/">
    <div class="toolkit-card__copy">
      <span class="toolkit-card__label">Stage C · Main toolkit</span>
      <span class="toolkit-card__title">Benchmarking and Modelling</span>
      <span class="toolkit-card__desc">Ranks standard and longitudinal estimators all against all, under one shared protocol, so that the value of machine learning over the statistical analysis in place is measured on the data at hand; then trains one chosen estimator in depth.</span>
      <span class="toolkit-card__tools"><code>BenchStandardAndLongitudinalML</code><code>BenchmarkStandardML</code><code>BenchmarkLongitudinalML</code><code>StandardMachineLearning</code><code>LongitudinalMachineLearning</code></span>
    </div>
    <div class="toolkit-card__window" aria-label="Example ranking from the First Benchmark tutorial">
      <div class="toolkit-card__bar"><span></span><span></span><span></span><em>benchmark_report.txt</em></div>
      <pre>1. Logistic Regression                    0.796
2. MerWavTimePlus + Lexico Random Forest  0.779
3. Random Forest                          0.759
4. SepWav (Voting) + Random Forest        0.722
5. AggrFunc (Mean) + Logistic Regression  0.605</pre>
    </div>
  </a>
  <a class="toolkit-card toolkit-card--wide" href="explainability/">
    <div class="toolkit-card__copy">
      <span class="toolkit-card__label">Stage D</span>
      <span class="toolkit-card__title">Explainability</span>
      <span class="toolkit-card__desc">Explains the predictions of a saved model, e.g., the best estimator of a benchmark, with SHAP values: which features, and which waves, drive them.</span>
      <span class="toolkit-card__tools"><code>SHAPAnalysis</code></span>
    </div>
    <div class="toolkit-card__window" aria-label="Example SHAP report of a longitudinal model">
      <div class="toolkit-card__bar"><span></span><span></span><span></span><em>shap_analysis_report.txt</em></div>
      <pre>Top features by mean absolute SHAP value
1. mood_score_w3: 0.165802
2. mood_score_w2: 0.076778
3. sleep_score_w3: 0.042331
4. mood_score_w1: 0.040893</pre>
    </div>
  </a>
</div>

## Proposing a New Toolkit or Tool

A new tool (e.g., an imputer, an estimator or a file converter) usually fits an existing toolkit, whereas a new toolkit is warranted when a whole stage of longitudinal analysis is missing. Either way, the starting point is a feature request on GitHub.

<div class="button-row" markdown>
[:material-github: Request a data preparation tool](https://github.com/Long-Toolkit/long-toolkit/issues/new?template=feature_data_preparation.yml){ .md-button }
[:material-github: Request a preprocessing tool](https://github.com/Long-Toolkit/long-toolkit/issues/new?template=feature_data_preprocessing.yml){ .md-button }
[:material-github: Request a benchmarking or modelling tool](https://github.com/Long-Toolkit/long-toolkit/issues/new?template=feature_machine_learning.yml){ .md-button }
[:material-github: Request an explainability tool](https://github.com/Long-Toolkit/long-toolkit/issues/new?template=feature_explainability.yml){ .md-button }
</div>
