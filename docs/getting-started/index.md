---
icon: lucide/play
hide:
  - navigation
---

# Get Started

`Long-Toolkit` (short for Longitudinal-Toolkit) is a hub of Python toolkits for longitudinal data, i.e., data in which the same participants are measured at several waves (time points).

First, the hub provides tools to get longitudinal data ready, from two viewpoints. Data preparation produces a usable table, by generating synthetic cohorts or by converting cohort files between formats (CSV, Parquet and Stata). Data preprocessing then makes that table fit for modelling, by cleaning, imputing, harmonising and reshaping the repeated measures.

Second, with the data ready, the hub answers one question: is machine learning, and/or longitudinal machine learning (models that exploit the repeated measures of each variable), worth it compared with the statistical analysis already in place, usually a logistic regression [[1]](https://doi.org/10.1016/j.annepidem.2010.08.003)[[2]](https://doi.org/10.1200/JCO.2007.11.9859)[[3]](https://pmc.ncbi.nlm.nih.gov/articles/PMC3462287/)? The Benchmarking and Modelling Toolkit answers it, by ranking both families of models side by side, all against all, under one shared protocol; it can then train and evaluate one chosen model in depth.

Third, the hub explains the results. The ranking shows whether, and by how much, machine learning improves on the statistical analysis on that dataset; the Explainability Toolkit then shows which features drive the predictions of a saved model, e.g., the best estimator of the benchmark.

## Motivation

During Simon Provost's internship at the University of Edinburgh, in the [LEAP lab](https://life-epi-psych.github.io) supervised by Dr Alex Kwong, we ([Simon Provost](https://simonprovost.xyz/) and [Dr Alex Kwong](https://edwebprofiles.ed.ac.uk/profile/dr-alex-kwong)) noticed a recurring pattern in longitudinal studies. Considerable effort goes into data collection and curation, whereas machine learning is either left aside, since its barrier to entry is high, or approached naively.

In the first case, studies keep to statistical techniques, which have been well established and stable over the years. Machine learning has, however, proven its value empirically in other fields; hence, it is worth benchmarking it on longitudinal studies too, e.g., in epidemiology or oncology. In the second case, machine learning is applied without regard for the temporal dependency of the data. Conventional machine learning treats the same feature collected at several waves for each participant (e.g., a mood score at ages 11, 14 and 17) as distinct, unrelated features, which does not reflect how such data arise in the real world; hence, longitudinal-data-aware machine learning, which models these repeated measures as one variable over time, should also be explored.

As a result, we developed `Long-Toolkit`, an open-source initiative usable through an application programming interface (API) or a command-line interface (CLI). It helps experts of longitudinal studies (usually not computer scientists by background) to make their data ready for machine learning, and it accelerates a fair evaluation of machine learning on such data, through a benchmark of both conventional and longitudinal-data-aware machine learning primitives.

!!! info "No deep learning yet"
    `Long-Toolkit` does not include deep learning models for longitudinal data yet, although they could be added in the future. If they are of interest to you, please [open an issue](https://github.com/Long-Toolkit/long-toolkit/issues/new?template=feature_machine_learning.yml). Read more on both families of methods in [Machine and deep learning for longitudinal biomedical data: a review of methods and applications](https://doi.org/10.1007/s10462-023-10561-w).

## Works via API and CLI

Every tool can be used in two modes. The Python package exposes each tool as a class, through its API, for scripts and Jupyter notebooks. The CLI is a no-code terminal application, written in Go, that calls the same Python package, so that every tool is one menu, or one command-palette search, away. We recommend the CLI for a first exploration, and the API for full control over each step.

## Toolkits and Stages

Each toolkit belongs to one of the four stages of a longitudinal analysis, in the order described above: Stage A for data preparation, Stage B for data preprocessing, Stage C for benchmarking and modelling, and Stage D for explainability.

| Toolkit | Stage | Question it answers |
| --- | --- | --- |
| [Data Preparation](../toolkits/data-preparation.md) | A | How does a raw or synthetic cohort become a usable table? |
| [Data Preprocessing](../toolkits/data-preprocessing.md) | B | How are repeated measures cleaned, imputed and reshaped? |
| [Benchmarking and Modelling](../toolkits/benchmarking-and-modelling.md) | C | Is machine learning, and/or longitudinal machine learning, worth it on this dataset? |
| [Explainability](../toolkits/explainability.md) | D | Which features drive the predictions of a chosen model? |

## Installation

!!! tip "Recommended package manager"
    We recommend [uv](https://docs.astral.sh/uv/), the state-of-the-art Python package manager. The CLI also relies on it to call the Python package.

### Machine Requirements

The Python package requires Python `3.10` to `3.12` (installed directly, or managed by `uv`). Go is only needed to build the CLI from source.

=== "macOS"

    ```bash
    brew update
    brew install uv libomp
    ```

    `libomp` is the OpenMP runtime that XGBoost and LightGBM (two dependencies of the modelling tools) load on macOS.

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

!!! warning "Windows"
    Windows is not supported natively yet. Run the toolkit inside a Docker Linux environment, or use a macOS or Linux machine.

### Python Package

=== "uv (recommended)"

    ```bash
    uv add long-toolkit
    ```

=== "pip"

    ```bash
    pip install long-toolkit
    ```

### CLI

=== "Homebrew (macOS and Linux)"

    ```bash
    brew tap Long-Toolkit/homebrew-tap
    brew install long-toolkit
    ```

=== "From source"

    ```bash
    git clone https://github.com/Long-Toolkit/CLI.git
    cd CLI
    make build
    make install-bash  # or: make install-fish
    ```

## Launch the CLI

```bash
long-toolkit
```

!!! danger "Launch the CLI from inside a Python project"
    The CLI must be run from a directory containing a `pyproject.toml` in which `long-toolkit` is installed (or from any of its subdirectories). It locates that `pyproject.toml` and calls the Python package through `uv` from there; elsewhere, it stops with an installation hint.

## Where Next?

<div class="grid cards" markdown>

-   :material-trophy-outline:{ .lg .middle } __First Benchmark__

    ---

    Rank standard against longitudinal machine learning on a synthetic cohort, in about thirty lines of Python.

    [:octicons-arrow-right-24: Open the tutorial](../tutorials/first-benchmark.md)

-   :material-view-grid-outline:{ .lg .middle } __Toolkits__

    ---

    What each toolkit does, in depth, with its options and trade-offs.

    [:octicons-arrow-right-24: Open the toolkits](../toolkits/index.md)

-   :material-console:{ .lg .middle } __CLI Tutorial__

    ---

    Stages, navigation and the command palette of the no-code interface.

    [:octicons-arrow-right-24: Open the CLI tutorial](../tutorials/cli.md)

-   :material-book-open-variant:{ .lg .middle } __API Reference__

    ---

    Every class and parameter, generated from the docstrings.

    [:octicons-arrow-right-24: Open the API reference](../api/index.md)

</div>
