#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="alspac_long_sepwav_stacking_dt__svm_f05"
#SBATCH --output="experiments/alspac/logs/longitudinal_ml/sepwav_stacking_dt__svm/fold_05_%j.log"
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
    echo "  sbatch experiments/alspac/scenarios/longitudinal_ml/sepwav_stacking_dt__svm/fold_05.sh"
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
if [ ! -f "experiments/alspac/splits/alspac_holdout_cv_80_20_seed42.json" ]; then
    uv run python -m experiments.alspac.prepare_splits
fi

# ========================================
# Run Experiment (ALSPAC)
# ========================================
uv run python experiments/launchers/longitudinal_ml__sepwav_stacking_dt__svm.py \
    --fold-index 4 \
    --dataset-path "data/ALSPAC/ALSPAC_READY_engine.csv" \
    --split-manifest "experiments/alspac/splits/alspac_holdout_cv_80_20_seed42.json" \
    --output-root "experiments/alspac/results"
