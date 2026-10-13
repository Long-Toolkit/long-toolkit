"""Generate per-fold SLURM scripts for the ALSPAC sweep.

Mirrors ``experiments/generate_assets.py`` but targets ALSPAC. The existing
launcher Python files in ``experiments/launchers/`` are reused unchanged — only
``--dataset-path``, ``--split-manifest`` and ``--output-root`` change so that
results land under ``experiments/alspac/results/``.

This script is deliberately **self-contained**: it only walks the launcher
directory and writes shell scripts, so it can run on a workstation without the
heavy ML dependency stack (no lightgbm/libomp/miceforest required). Split
manifest generation happens on the HPC node the first time a SLURM job runs.

Run with:

    uv run python -m experiments.alspac.generate_assets
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
LAUNCHERS_ROOT = EXPERIMENTS_ROOT / "launchers"

ALSPAC_ROOT = EXPERIMENTS_ROOT / "alspac"
ALSPAC_DATASET_PATH = PROJECT_ROOT / "data" / "ALSPAC" / "ALSPAC_READY_engine.csv"
ALSPAC_RESULTS_ROOT = ALSPAC_ROOT / "results"
ALSPAC_SPLITS_ROOT = ALSPAC_ROOT / "splits"
ALSPAC_LOGS_ROOT = ALSPAC_ROOT / "logs"
ALSPAC_REPORTS_ROOT = ALSPAC_ROOT / "reports"
ALSPAC_SCENARIOS_ROOT = ALSPAC_ROOT / "scenarios"
ALSPAC_SPLIT_MANIFEST_PATH = (
    ALSPAC_SPLITS_ROOT / "alspac_holdout_cv_80_20_seed42.json"
)
ALSPAC_REPORT_SLURM_PATH = ALSPAC_ROOT / "report_results.slurm.sh"
ALSPAC_REPORT_NO_SHAP_SLURM_PATH = ALSPAC_ROOT / "report_results_no_shap.slurm.sh"

N_FOLDS = 10
ALSPAC_TARGET_COLUMN = "class"

SCOPE_KEYS = (
    "standard_ml",
    "longitudinal_ml",
    "balanced_bagging_standard_ml",
    "balanced_bagging_longitudinal_ml",
)
SCOPE_SHORT_NAMES = {
    "standard_ml": "std",
    "longitudinal_ml": "long",
    "balanced_bagging_standard_ml": "bbstd",
    "balanced_bagging_longitudinal_ml": "bblong",
}

# Memory overrides copied verbatim from ``experiments/generate_assets.py`` so
# we don't pay an import-time cost for the registry. Keep in sync if MCS values
# change.
FIFTEEN_GB_MEMORY_OVERRIDES: set[tuple[str, str]] = {
    ("balanced_bagging_longitudinal_ml", "merwav_time_plus__lexico_deep_forest"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__decision_tree"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__extra_trees"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__gradient_boosting"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__knn"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__random_forest"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__svm"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__extra_trees"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__gradient_boosting"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__knn"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__random_forest"),
    ("balanced_bagging_longitudinal_ml", "sepwav_voting__extra_trees"),
    ("balanced_bagging_longitudinal_ml", "sepwav_voting__random_forest"),
}

THIRTY_GB_MEMORY_OVERRIDES: set[tuple[str, str]] = {
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__extra_trees"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_dt__random_forest"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__extra_trees"),
    ("balanced_bagging_longitudinal_ml", "sepwav_stacking_lr__random_forest"),
}

THIRTY_GB_FOLD_MEMORY_OVERRIDES: set[tuple[str, str, int]] = {
    ("balanced_bagging_longitudinal_ml", "sepwav_voting__random_forest", 5),
}

ALSPAC_DATASET_RELATIVE = ALSPAC_DATASET_PATH.relative_to(PROJECT_ROOT)
ALSPAC_SPLIT_MANIFEST_RELATIVE = ALSPAC_SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT)
ALSPAC_RESULTS_ROOT_RELATIVE = ALSPAC_RESULTS_ROOT.relative_to(PROJECT_ROOT)
ALSPAC_REPORTS_ROOT_RELATIVE = ALSPAC_REPORTS_ROOT.relative_to(PROJECT_ROOT)
ALSPAC_LOGS_ROOT_RELATIVE = ALSPAC_LOGS_ROOT.relative_to(PROJECT_ROOT)
ALSPAC_SCENARIOS_ROOT_RELATIVE = ALSPAC_SCENARIOS_ROOT.relative_to(PROJECT_ROOT)


def _discover_launcher_specs() -> list[tuple[str, str, str]]:
    """Return ``(launcher_key, scope_key, primitive_key)`` from launcher files."""
    if not LAUNCHERS_ROOT.is_dir():
        raise SystemExit(
            f"Launcher directory {LAUNCHERS_ROOT} not found. "
            "Run `uv run python -m experiments.generate_assets` first to "
            "create the per-primitive launcher .py files."
        )

    specs: list[tuple[str, str, str]] = []
    for path in sorted(LAUNCHERS_ROOT.glob("*.py")):
        launcher_key = path.stem
        if launcher_key.startswith("__"):
            continue
        scope_key: str | None = None
        for candidate in sorted(SCOPE_KEYS, key=len, reverse=True):
            prefix = f"{candidate}__"
            if launcher_key.startswith(prefix):
                scope_key = candidate
                break
        if scope_key is None:
            print(f"Skipping launcher with unknown scope: {launcher_key}")
            continue
        primitive_key = launcher_key[len(scope_key) + 2 :]
        specs.append((launcher_key, scope_key, primitive_key))
    return specs


def _resolve_memory_gb(*, scope_key: str, primitive_key: str, fold_number: int) -> int:
    if (scope_key, primitive_key, fold_number) in THIRTY_GB_FOLD_MEMORY_OVERRIDES:
        return 30
    if (scope_key, primitive_key) in THIRTY_GB_MEMORY_OVERRIDES:
        return 30
    if (scope_key, primitive_key) in FIFTEEN_GB_MEMORY_OVERRIDES:
        return 15
    return 5


def _slurm_template(
    *,
    launcher_key: str,
    scope_key: str,
    primitive_key: str,
    fold_number: int,
) -> str:
    scope_short = SCOPE_SHORT_NAMES[scope_key]
    fold_display = f"{fold_number:02d}"
    job_name = f"alspac_{scope_short}_{primitive_key}_f{fold_display}"
    log_path = (
        f"{ALSPAC_LOGS_ROOT_RELATIVE.as_posix()}/{scope_key}/{primitive_key}/"
        f"fold_{fold_display}_%j.log"
    )
    scenario_path = (
        f"{ALSPAC_SCENARIOS_ROOT_RELATIVE.as_posix()}/{scope_key}/{primitive_key}/"
        f"fold_{fold_display}.sh"
    )
    launcher_name = f"{launcher_key}.py"
    memory_gb = _resolve_memory_gb(
        scope_key=scope_key,
        primitive_key=primitive_key,
        fold_number=fold_number,
    )
    return f"""#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="{job_name}"
#SBATCH --output="{log_path}"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="sgp28@kent.ac.uk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory_gb}GB
#SBATCH --partition=cpu

# ========================================
# Environment Setup
# ========================================
if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in current directory."
    echo "Run sbatch from the project root, for example:"
    echo "  cd /home/arc/sgp28/long-toolkit"
    echo "  sbatch {scenario_path}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv python install 3.10
if ! uv sync --check >/dev/null 2>&1; then
    uv sync --all-groups
fi

# ========================================
# Ensure ALSPAC split manifest exists
# ========================================
if [ ! -f "{ALSPAC_SPLIT_MANIFEST_RELATIVE.as_posix()}" ]; then
    uv run python -m experiments.alspac.prepare_splits
fi

# ========================================
# Run Experiment (ALSPAC)
# ========================================
uv run python experiments/launchers/{launcher_name} \\
    --fold-index {fold_number - 1} \\
    --dataset-path "{ALSPAC_DATASET_RELATIVE.as_posix()}" \\
    --split-manifest "{ALSPAC_SPLIT_MANIFEST_RELATIVE.as_posix()}" \\
    --output-root "{ALSPAC_RESULTS_ROOT_RELATIVE.as_posix()}"
"""


def _report_slurm_template(*, with_shap: bool) -> str:
    job_suffix = "report" if with_shap else "report_no_shap"
    log_path = f"{ALSPAC_LOGS_ROOT_RELATIVE.as_posix()}/{job_suffix}_%j.log"
    extra_flag = "" if with_shap else " --skip-shap"
    return f"""#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="alspac_{job_suffix}"
#SBATCH --output="{log_path}"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="sgp28@kent.ac.uk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

# ========================================
# Environment Setup
# ========================================
if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in current directory."
    echo "Run sbatch from the project root."
    exit 1
fi

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv python install 3.10
if ! uv sync --check >/dev/null 2>&1; then
    uv sync --all-groups
fi

# ========================================
# Run Reporting (ALSPAC)
# ========================================
uv run python experiments/report_results.py \\
    --dataset-path "{ALSPAC_DATASET_RELATIVE.as_posix()}" \\
    --split-manifest "{ALSPAC_SPLIT_MANIFEST_RELATIVE.as_posix()}" \\
    --results-root "{ALSPAC_RESULTS_ROOT_RELATIVE.as_posix()}" \\
    --report-root "{ALSPAC_REPORTS_ROOT_RELATIVE.as_posix()}"{extra_flag} "$@"
"""


def _write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def main() -> None:
    for directory in (
        ALSPAC_ROOT,
        ALSPAC_SPLITS_ROOT,
        ALSPAC_LOGS_ROOT,
        ALSPAC_RESULTS_ROOT,
        ALSPAC_REPORTS_ROOT,
        ALSPAC_SCENARIOS_ROOT,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _write_text(ALSPAC_RESULTS_ROOT / ".gitkeep", "")
    _write_text(ALSPAC_REPORTS_ROOT / ".gitkeep", "")
    _write_text(ALSPAC_LOGS_ROOT / ".gitkeep", "")

    specs = _discover_launcher_specs()
    n_scenarios = 0
    for launcher_key, scope_key, primitive_key in specs:
        scope_dir = ALSPAC_SCENARIOS_ROOT / scope_key / primitive_key
        logs_dir = ALSPAC_LOGS_ROOT / scope_key / primitive_key
        scope_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        for fold_number in range(1, N_FOLDS + 1):
            scenario_path = scope_dir / f"fold_{fold_number:02d}.sh"
            _write_text(
                scenario_path,
                _slurm_template(
                    launcher_key=launcher_key,
                    scope_key=scope_key,
                    primitive_key=primitive_key,
                    fold_number=fold_number,
                ),
                executable=True,
            )
            n_scenarios += 1

    _write_text(
        ALSPAC_REPORT_SLURM_PATH,
        _report_slurm_template(with_shap=True),
        executable=True,
    )
    _write_text(
        ALSPAC_REPORT_NO_SHAP_SLURM_PATH,
        _report_slurm_template(with_shap=False),
        executable=True,
    )

    if not ALSPAC_DATASET_PATH.exists():
        print(
            "Note: ALSPAC engine dataset not found at",
            ALSPAC_DATASET_PATH.relative_to(PROJECT_ROOT),
            "- run `uv run python -m experiments.alspac.prepare_engine_dataset`",
            "before submitting jobs.",
        )

    print(
        "Generated",
        n_scenarios,
        "ALSPAC SLURM scripts across",
        len(specs),
        "primitives, plus 2 report SLURM scripts.",
    )


if __name__ == "__main__":
    main()
