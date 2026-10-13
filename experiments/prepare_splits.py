from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.registry import DEFAULT_DATASET_PATH, DEFAULT_SPLIT_MANIFEST_PATH
from experiments.splits import prepare_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed 80/20 holdout split and 10-fold CV manifest."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the wide preprocessed MCS CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Where to write the split manifest JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the manifest even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_split_manifest(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        force=args.force,
    )
    print(
        "Prepared split manifest:",
        args.output_path,
        f"| train={len(manifest.train_indices)}",
        f"| test={len(manifest.test_indices)}",
        f"| folds={len(manifest.folds)}",
    )


if __name__ == "__main__":
    main()
