#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="alspac_report_no_shap"
#SBATCH --output="experiments/alspac/logs/report_no_shap_%j.log"
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
uv run python experiments/report_results.py \
    --dataset-path "data/ALSPAC/ALSPAC_READY_engine.csv" \
    --split-manifest "experiments/alspac/splits/alspac_holdout_cv_80_20_seed42.json" \
    --results-root "experiments/alspac/results" \
    --report-root "experiments/alspac/reports" --skip-shap "$@"
