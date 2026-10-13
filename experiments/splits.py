from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.registry import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_RELATIVE_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    N_FOLDS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class SplitManifest:
    dataset_relpath: str
    target_column: str
    random_seed: int
    test_size: float
    n_folds: int
    dataset_summary: dict[str, object]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    folds: tuple[FoldSplit, ...]


def normalise_target_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    if pd.api.types.is_numeric_dtype(cleaned):
        non_null = cleaned.dropna()
        if not non_null.empty and np.allclose(non_null, np.round(non_null)):
            cleaned = cleaned.round().astype(int)
    return cleaned


def _build_dataset_summary(data: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    return {
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "feature_column_count": int(data.shape[1] - 1),
        "class_distribution": {
            str(label): int(count)
            for label, count in target.value_counts(dropna=False).sort_index().items()
        },
    }


def prepare_split_manifest(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    target_column: str = TARGET_COLUMN,
    output_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    n_folds: int = N_FOLDS,
    force: bool = False,
) -> SplitManifest:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return load_split_manifest(output_path)

    data = pd.read_csv(dataset_path)
    target = normalise_target_series(data[target_column])
    row_indices = np.arange(len(data), dtype=int)

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=target,
    )

    train_target = target.iloc[train_indices]
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    folds: list[FoldSplit] = []
    for fold_index, (relative_train, relative_validation) in enumerate(
        cv.split(train_indices, train_target),
        start=0,
    ):
        fold_train_indices = tuple(int(index) for index in train_indices[relative_train])
        fold_validation_indices = tuple(
            int(index) for index in train_indices[relative_validation]
        )
        folds.append(
            FoldSplit(
                fold_index=fold_index,
                train_indices=fold_train_indices,
                validation_indices=fold_validation_indices,
            )
        )

    dataset_relpath = (
        dataset_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if dataset_path.resolve().is_relative_to(PROJECT_ROOT.resolve())
        else DEFAULT_DATASET_RELATIVE_PATH.as_posix()
    )

    manifest = SplitManifest(
        dataset_relpath=dataset_relpath,
        target_column=target_column,
        random_seed=random_seed,
        test_size=test_size,
        n_folds=n_folds,
        dataset_summary=_build_dataset_summary(data, target),
        train_indices=tuple(int(index) for index in train_indices),
        test_indices=tuple(int(index) for index in test_indices),
        folds=tuple(folds),
    )

    output_path.write_text(json.dumps(asdict(manifest), indent=2))
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    pd.DataFrame(
        [
            {
                "fold_index": fold.fold_index,
                "train_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices),
            }
            for fold in manifest.folds
        ]
    ).to_csv(summary_path, index=False)
    return manifest


def load_split_manifest(path: Path = DEFAULT_SPLIT_MANIFEST_PATH) -> SplitManifest:
    payload = json.loads(path.read_text())
    return SplitManifest(
        dataset_relpath=str(payload["dataset_relpath"]),
        target_column=str(payload["target_column"]),
        random_seed=int(payload["random_seed"]),
        test_size=float(payload["test_size"]),
        n_folds=int(payload["n_folds"]),
        dataset_summary=dict(payload["dataset_summary"]),
        train_indices=tuple(int(index) for index in payload["train_indices"]),
        test_indices=tuple(int(index) for index in payload["test_indices"]),
        folds=tuple(
            FoldSplit(
                fold_index=int(item["fold_index"]),
                train_indices=tuple(int(index) for index in item["train_indices"]),
                validation_indices=tuple(
                    int(index) for index in item["validation_indices"]
                ),
            )
            for item in payload["folds"]
        ),
    )


def resolve_dataset_path_from_manifest(
    manifest: SplitManifest,
    dataset_path: Path | None = None,
) -> Path:
    if dataset_path is not None:
        return dataset_path
    return PROJECT_ROOT / manifest.dataset_relpath
