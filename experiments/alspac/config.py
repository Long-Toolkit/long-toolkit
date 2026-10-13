"""Shared paths and constants for the ALSPAC sweep.

Kept separate from ``experiments/registry.py`` so the existing MCS pipeline is
unaffected. The engine, primitive registry, and reporting utilities are reused.
"""

from __future__ import annotations

from pathlib import Path

from experiments.registry import EXPERIMENTS_ROOT, PROJECT_ROOT

ALSPAC_ROOT = EXPERIMENTS_ROOT / "alspac"
ALSPAC_DATASET_PATH = PROJECT_ROOT / "data" / "ALSPAC" / "ALSPAC_READY_engine.csv"
ALSPAC_RESULTS_ROOT = ALSPAC_ROOT / "results"
ALSPAC_SPLITS_ROOT = ALSPAC_ROOT / "splits"
ALSPAC_LOGS_ROOT = ALSPAC_ROOT / "logs"
ALSPAC_REPORTS_ROOT = ALSPAC_ROOT / "reports"
ALSPAC_SPLIT_MANIFEST_PATH = (
    ALSPAC_SPLITS_ROOT / "alspac_holdout_cv_80_20_seed42.json"
)
ALSPAC_TARGET_COLUMN = "class"
