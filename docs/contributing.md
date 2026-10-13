---
icon: simple/github
---

# Contributing to Long-Toolkit

Contributions are welcome across both components of the hub, i.e., the Python package (where every tool lives) and the command-line interface (CLI), i.e., the no-code interface that drives them. Before implementing a major change, open an issue in the relevant repository, so that the proposal can be discussed and duplicated work avoided.

- Python package issues: <https://github.com/Long-Toolkit/long-toolkit/issues>
- CLI issues: <https://github.com/Long-Toolkit/CLI/issues>

## Contributing to the Python Package

Fork the repository, create a feature branch, and add tests and documentation alongside the change. The package is managed with `uv`, the state-of-the-art Python package manager:

```bash
git clone https://github.com/Long-Toolkit/long-toolkit.git
cd long-toolkit
uv sync
uv run pytest tests/
uv run ruff check src
```

A new tool usually fits an existing toolkit (e.g., a new imputer, estimator or file converter), in which case it follows the same template as its neighbours and is discovered automatically; a new toolkit is warranted when a whole stage of longitudinal analysis is missing.

## Contributing to the CLI

The CLI lives in its own repository and is built with Go:

```bash
git clone https://github.com/Long-Toolkit/CLI.git
cd CLI
make build
make install-bash  # or: make install-fish
```

The `Makefile` also provides `make check` (formatting, `go vet` and linting) for local changes.

## Building the Documentation

The documentation is built with [Zensical](https://zensical.org/), from the root of the Python package:

```bash
uv sync
uv run zensical serve
```

A production build, identical to the one published on Read the Docs, is obtained with `uv run zensical build` (or with `./build_docs.sh`).
