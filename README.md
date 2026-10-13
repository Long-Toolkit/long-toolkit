<div align="center">
  <img src="public/logo/logo.gif" alt="Long-Toolkit" width="820">
</div>

<p align="center"><big><big><strong>Long</strong>-<em>Toolkit</em></big></big><br/><em>A hub of Python toolkits for longitudinal data, from preparation to explainability</em></p>

<div align="center">
  <img src="https://img.shields.io/static/v1?label=Python&message=3.10%20to%203.12&color=DEA2C2&style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10 to 3.12">
  <img src="https://img.shields.io/static/v1?label=UV&message=managed&color=DEA2C2&style=for-the-badge&logo=uv&logoColor=white" alt="UV managed">
  <img src="https://img.shields.io/static/v1?label=Ruff&message=compliant&color=DEA2C2&style=for-the-badge&logo=ruff&logoColor=white" alt="Ruff compliant">
  <img src="https://img.shields.io/static/v1?label=Beartype&message=verified&color=DEA2C2&style=for-the-badge" alt="Beartype verified">
  <img src="https://img.shields.io/static/v1?label=Go%20CLI&message=supported&color=DEA2C2&style=for-the-badge&logo=go&logoColor=white" alt="Go CLI supported">
  <img src="https://img.shields.io/static/v1?label=Docs&message=Zensical&color=DEA2C2&style=for-the-badge" alt="Docs built with Zensical">
</div>

<div align="center">
  <a href="https://long-toolkit.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="https://github.com/Long-Toolkit/CLI">CLI</a> ·
  <a href="https://life-epi-psych.github.io">LEAP Group</a>
</div>

## <img src="public/icons/lucide/github.svg" width="32" alt="" /> About The Project

`Long-Toolkit` (short for Longitudinal-Toolkit) is a hub of Python toolkits for longitudinal data, i.e., data in which the same participants are measured at several waves (time points). It is designed for *social, medical and clinical researchers*.

First, the hub provides tools to get longitudinal data ready, from two viewpoints. *Data preparation* produces a usable table, by generating synthetic cohorts or by converting cohort files between formats (CSV, Parquet and Stata). *Data preprocessing* then makes that table fit for modelling, by cleaning, imputing, harmonising and reshaping the repeated measures.

Second, with the data ready, the hub answers one question: is machine learning, and/or longitudinal machine learning (models that exploit the repeated measures of each variable), worth it compared with the statistical analysis already in place, usually a logistic regression [[1]](https://doi.org/10.1016/j.annepidem.2010.08.003)[[2]](https://doi.org/10.1200/JCO.2007.11.9859)[[3]](https://pmc.ncbi.nlm.nih.gov/articles/PMC3462287/)? The *Benchmarking and Modelling Toolkit* answers it, by ranking both families of models side by side, all against all, under one shared protocol; it can then train and evaluate one chosen model in depth.

Third, the hub explains the results. The ranking shows whether, and by how much, machine learning improves on the statistical analysis on that dataset; the *Explainability Toolkit* then shows which features drive the predictions of a saved model, e.g., the best estimator of the benchmark.

| Toolkit | Stage | Question it answers |
| --- | --- | --- |
| Data Preparation | A | How does a raw or synthetic cohort become a usable table? |
| Data Preprocessing | B | How are repeated measures cleaned, imputed and reshaped? |
| Benchmarking and Modelling (main toolkit) | C | Is machine learning, and/or longitudinal machine learning, worth it on this dataset? |
| Explainability | D | Which features drive the predictions of a chosen model? |

The hub ships as two components that run the same tools: the `long-toolkit` *Python package*, i.e., this repository, used through its application programming interface (API) in scripts and notebooks, and the `long-toolkit` [command-line interface (CLI)](https://github.com/Long-Toolkit/CLI), a *no-code* terminal interface written in Go.

## <img src="public/icons/lucide/terminal.svg" width="32" alt="" /> Setup And Launch

> [!IMPORTANT]
> Install both components: `long-toolkit` (Python package) and `long-toolkit` (CLI). The CLI is the recommended no-code entry point.

<details>
<summary><strong>Machine Requirements</strong> (<a href="https://www.python.org/downloads/">Python</a>, <a href="https://docs.astral.sh/uv/getting-started/installation/">uv</a>)</summary>

Install the following first:
- Python `3.10` to `3.12`, or [`uv`](https://docs.astral.sh/uv/), the state-of-the-art Python package manager, to manage it
- On macOS, `libomp` (the OpenMP runtime loaded by XGBoost and LightGBM)
- Go, only to build the CLI from source

```bash
# macOS (Homebrew)
brew update
brew install uv libomp

# Linux (Ubuntu/Debian)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

### <img src="public/icons/lucide/tally-1.svg" width="28" alt="" /> Install `long-toolkit` ([PyPI Index](https://pypi.org/project/long-toolkit/))

Use Python `3.10` for now: it is the validated installation, while a longitudinal dependency currently fails to build from source on Python `3.11`.

Before installing, add the following to your project's `pyproject.toml`. Scikit-Longitudinal supplies `sklearn` through its lexicographical fork, so the stock package must be excluded, as described in its [installation guide](https://scikit-longitudinal.readthedocs.io/latest/getting-started/).

```toml
[tool.uv]
override-dependencies = ["scikit-learn ; sys_platform == 'never'"]
```

```bash
uv add long-toolkit
# We recommend uv as the Python package manager
```

### <img src="public/icons/lucide/tally-2.svg" width="28" alt="" /> Install the `long-toolkit` CLI (No-Code Terminal Interface)

Homebrew (macOS and Linux):

```bash
brew tap Long-Toolkit/homebrew-tap
brew install long-toolkit
```

<img src="public/icons/lucide/party-popper.svg" width="14" alt="" /> Then, from inside the Python project in which `long-toolkit` is installed, run:

```bash
long-toolkit
```

> [!WARNING]
> The full `long-toolkit` workflow is not supported natively on Windows yet.
> As a workaround, run the toolkit inside a Docker Linux environment, or use a macOS or Linux machine.
> Please open [an issue](https://github.com/Long-Toolkit/long-toolkit/issues) to contribute towards Windows support.

### <img src="public/icons/lucide/square-code.svg" width="32" alt="" /> Getting Started With the Toolkits

The documentation walks through every toolkit, with code that runs as is:

- [Get Started](https://long-toolkit.readthedocs.io/en/latest/getting-started/): what the hub does, and how to install it.
- [First Benchmark](https://long-toolkit.readthedocs.io/en/latest/tutorials/first-benchmark/): standard against longitudinal machine learning on a synthetic cohort, in about thirty lines of Python.
- [Toolkits](https://long-toolkit.readthedocs.io/en/latest/toolkits/): Data Preparation, Data Preprocessing, Benchmarking and Modelling, and Explainability, in depth.
- [Tutorials](https://long-toolkit.readthedocs.io/en/latest/tutorials/): the CLI and the Python API, step by step.

## <img src="public/icons/lucide/graduation-cap.svg" width="32" alt="" /> Citation

```bibtex
@software{Provost_Long_Toolkit,
    author = {Provost, Simon and Branco, Bianca and Chen, Honghui and Tassella, Isaac and Kwong, Alex},
    title = {{Long-Toolkit: A Hub of Python Toolkits for Longitudinal Data}},
    version = {0.0.4}
}
```

Use GitHub's "Cite this repository" for citation metadata updates.

## <img src="public/icons/lucide/fingerprint-pattern.svg" width="32" alt="" /> License, Data, Security

- Review [SECURITY.md](./SECURITY.md) for handling expectations.
- Licensed under the [MIT License](./LICENSE).

_Special thanks to [@charm.land](https://charm.land) for their amazing TUI framework!_
