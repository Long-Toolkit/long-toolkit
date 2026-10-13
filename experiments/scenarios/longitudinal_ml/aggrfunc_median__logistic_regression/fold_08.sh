#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="long_long_aggrfunc_median__logistic_regression_f08"
#SBATCH --output="experiments/logs/longitudinal_ml/aggrfunc_median__logistic_regression/fold_08_%j.log"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="sgp28@kent.ac.uk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --partition=cpu

# ========================================
# Environment Setup
# ========================================
if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in current directory."
    echo "Run sbatch from the project root, for example:"
    echo "  cd /home/arc/sgp28/long-toolkit"
    echo "  sbatch experiments/scenarios/longitudinal_ml/aggrfunc_median__logistic_regression/fold_08.sh"
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
uv run python experiments/launchers/longitudinal_ml__aggrfunc_median__logistic_regression.py \
    --fold-index 7 \
    --dataset-path "data/processed/MCS/mcs_longitudinal_wide_preprocessed.csv" \
    --split-manifest "experiments/splits/mcs_holdout_cv_80_20_seed42.json"
