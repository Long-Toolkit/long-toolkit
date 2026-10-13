from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.registry import (
    DEFAULT_DATASET_RELATIVE_PATH,
    DEFAULT_LAUNCHERS_ROOT,
    DEFAULT_LOGS_ROOT,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SCENARIOS_ROOT,
    DEFAULT_SPLIT_MANIFEST_PATH,
    N_FOLDS,
    SCOPE_SHORT_NAMES,
    get_all_primitive_specs,
)
from experiments.splits import prepare_split_manifest

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


def _launcher_template(launcher_key: str) -> str:
    return f"""from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.engine import run_launcher_cli


if __name__ == "__main__":
    run_launcher_cli({launcher_key!r})
"""


def _resolve_memory_gb(*, scope_key: str, primitive_key: str, fold_number: int) -> int:
    if (scope_key, primitive_key, fold_number) in THIRTY_GB_FOLD_MEMORY_OVERRIDES:
        return 30
    if (scope_key, primitive_key) in THIRTY_GB_MEMORY_OVERRIDES:
        return 30
    if (scope_key, primitive_key) in FIFTEEN_GB_MEMORY_OVERRIDES:
        return 15
    return 5


def _slurm_template(*, launcher_key: str, scope_key: str, primitive_key: str, fold_number: int) -> str:
    scope_short = SCOPE_SHORT_NAMES[scope_key]
    fold_display = f"{fold_number:02d}"
    job_name = f"long_{scope_short}_{primitive_key}_f{fold_display}"
    log_path = f"experiments/logs/{scope_key}/{primitive_key}/fold_{fold_display}_%j.log"
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
    echo "  sbatch experiments/scenarios/{scope_key}/{primitive_key}/fold_{fold_display}.sh"
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
# Run Experiment
# ========================================
uv run python experiments/launchers/{launcher_name} \\
    --fold-index {fold_number - 1} \\
    --dataset-path "{DEFAULT_DATASET_RELATIVE_PATH.as_posix()}" \\
    --split-manifest "{DEFAULT_SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}"
"""


def _write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def main() -> None:
    DEFAULT_LAUNCHERS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_SCENARIOS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    _write_text(DEFAULT_RESULTS_ROOT / ".gitkeep", "")
    _write_text(DEFAULT_REPORTS_ROOT / ".gitkeep", "")
    _write_text(DEFAULT_LOGS_ROOT / ".gitkeep", "")

    prepare_split_manifest(output_path=DEFAULT_SPLIT_MANIFEST_PATH)

    for spec in get_all_primitive_specs():
        _write_text(spec.launcher_path, _launcher_template(spec.launcher_key))
        spec.logs_dir.mkdir(parents=True, exist_ok=True)
        spec.scenario_dir.mkdir(parents=True, exist_ok=True)
        for fold_number in range(1, N_FOLDS + 1):
            scenario_path = spec.scenario_dir / f"fold_{fold_number:02d}.sh"
            _write_text(
                scenario_path,
                _slurm_template(
                    launcher_key=spec.launcher_key,
                    scope_key=spec.scope_key,
                    primitive_key=spec.primitive_key,
                    fold_number=fold_number,
                ),
                executable=True,
            )

    print(
        "Generated",
        len(get_all_primitive_specs()),
        "launcher scripts and",
        len(get_all_primitive_specs()) * N_FOLDS,
        "SLURM scripts.",
    )


if __name__ == "__main__":
    main()
