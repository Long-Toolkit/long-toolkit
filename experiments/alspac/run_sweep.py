"""Run the full sweep of every primitive × every CV fold against ALSPAC_READY.

For each registered ``PrimitiveSpec`` (standard, longitudinal, balanced-bagging
variants of both) this script invokes the engine's ``run_fold_for_launcher`` —
the same code path the MCS launchers use — but with the ALSPAC dataset and a
dedicated results root under ``experiments/alspac/results/``.

By default it skips folds whose ``fold_result.json`` already exists (the engine
itself short-circuits unless ``--force`` is set).

Run with:

    uv run python -m experiments.alspac.run_sweep
    uv run python -m experiments.alspac.run_sweep --scope standard_ml --jobs 4
    uv run python -m experiments.alspac.run_sweep --primitive standard_ml__random_forest
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.alspac.config import (
    ALSPAC_DATASET_PATH,
    ALSPAC_RESULTS_ROOT,
    ALSPAC_SPLIT_MANIFEST_PATH,
)
from experiments.alspac.prepare_splits import main as prepare_splits_main  # noqa: F401
from experiments.engine import run_fold_for_launcher
from experiments.registry import (
    N_FOLDS,
    PrimitiveSpec,
    SCOPE_ORDER,
    get_all_primitive_specs,
)
from experiments.splits import prepare_split_manifest


@dataclass(frozen=True)
class FoldJob:
    launcher_key: str
    fold_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ALSPAC_DATASET_PATH,
        help="ALSPAC dataset (engine-formatted) CSV.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ALSPAC_SPLIT_MANIFEST_PATH,
        help="Where to read/write the ALSPAC split manifest.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ALSPAC_RESULTS_ROOT,
        help="Root directory where fold outputs are written.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        choices=sorted(SCOPE_ORDER.keys()),
        help="Restrict to one or more scope keys (repeatable).",
    )
    parser.add_argument(
        "--primitive",
        action="append",
        default=None,
        help="Restrict to specific launcher_keys (repeatable).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Subset of zero-based fold indices to run (default: all 10).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Process-level parallelism (each job runs a single fold).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run folds even if outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs and exit.",
    )
    return parser.parse_args()


def _filter_specs(args: argparse.Namespace) -> tuple[PrimitiveSpec, ...]:
    specs = get_all_primitive_specs()
    if args.scope:
        specs = tuple(spec for spec in specs if spec.scope_key in set(args.scope))
    if args.primitive:
        wanted = set(args.primitive)
        specs = tuple(spec for spec in specs if spec.launcher_key in wanted)
        missing = wanted - {spec.launcher_key for spec in specs}
        if missing:
            raise SystemExit(f"Unknown launcher_keys: {sorted(missing)}")
    return specs


def _resolve_fold_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if args.folds is None:
        return tuple(range(N_FOLDS))
    folds = tuple(sorted(set(args.folds)))
    for index in folds:
        if not 0 <= index < N_FOLDS:
            raise SystemExit(f"Fold index {index} out of [0, {N_FOLDS}).")
    return folds


def _run_one(job: FoldJob, *, dataset_path: Path, manifest_path: Path,
             results_root: Path, force: bool) -> dict[str, Any]:
    return run_fold_for_launcher(
        launcher_key=job.launcher_key,
        dataset_path=dataset_path,
        split_manifest_path=manifest_path,
        fold_index=job.fold_index,
        output_root=results_root,
        force=force,
    )


def _ensure_manifest(args: argparse.Namespace) -> None:
    if args.split_manifest.exists() and not args.force:
        return
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepare_split_manifest(
        dataset_path=args.dataset_path,
        target_column="class",
        output_path=args.split_manifest,
        force=args.force,
    )


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)

    if not args.dataset_path.exists():
        raise SystemExit(
            f"Dataset {args.dataset_path} not found. "
            "Run `experiments.alspac.prepare_engine_dataset` first."
        )
    _ensure_manifest(args)

    specs = _filter_specs(args)
    fold_indices = _resolve_fold_indices(args)
    jobs = [
        FoldJob(launcher_key=spec.launcher_key, fold_index=fold_index)
        for spec in specs
        for fold_index in fold_indices
    ]

    print(
        f"Sweep plan: {len(specs)} primitive(s) × {len(fold_indices)} fold(s) "
        f"= {len(jobs)} job(s) | jobs={args.jobs} | force={args.force}"
    )
    if args.dry_run:
        for job in jobs:
            print(f"  • {job.launcher_key}  fold={job.fold_index}")
        return

    failures: list[tuple[FoldJob, str]] = []
    if args.jobs <= 1:
        for index, job in enumerate(jobs, start=1):
            label = f"[{index}/{len(jobs)}] {job.launcher_key} fold={job.fold_index}"
            print(label, flush=True)
            try:
                _run_one(
                    job,
                    dataset_path=args.dataset_path,
                    manifest_path=args.split_manifest,
                    results_root=args.results_root,
                    force=args.force,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((job, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_job = {
                pool.submit(
                    _run_one,
                    job,
                    dataset_path=args.dataset_path,
                    manifest_path=args.split_manifest,
                    results_root=args.results_root,
                    force=args.force,
                ): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                label = f"[{index}/{len(jobs)}] {job.launcher_key} fold={job.fold_index}"
                try:
                    future.result()
                    print(f"{label} ✓", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failures.append((job, f"{type(exc).__name__}: {exc}"))
                    print(f"{label} ✗ {exc}", flush=True)

    print()
    print(f"Completed: {len(jobs) - len(failures)}/{len(jobs)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for job, message in failures:
            print(f"  • {job.launcher_key} fold={job.fold_index}  →  {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
