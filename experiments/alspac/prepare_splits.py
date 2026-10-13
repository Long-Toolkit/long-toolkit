"""Prepare a stratified 80/20 holdout + 10-fold CV manifest for ALSPAC.

Run with:

    uv run python -m experiments.alspac.prepare_splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.alspac.config import ALSPAC_DATASET_PATH, ALSPAC_SPLIT_MANIFEST_PATH
from experiments.registry import N_FOLDS, RANDOM_SEED, TEST_SIZE
from experiments.splits import prepare_split_manifest

DEFAULT_TARGET_COLUMN = "class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=ALSPAC_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=ALSPAC_SPLIT_MANIFEST_PATH)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_split_manifest(
        dataset_path=args.dataset_path,
        target_column=args.target,
        output_path=args.output_path,
        random_seed=RANDOM_SEED,
        test_size=TEST_SIZE,
        n_folds=N_FOLDS,
        force=args.force,
    )
    print(
        "Prepared ALSPAC split manifest:",
        args.output_path,
        f"| train={len(manifest.train_indices)}",
        f"| test={len(manifest.test_indices)}",
        f"| folds={len(manifest.folds)}",
    )


if __name__ == "__main__":
    main()
