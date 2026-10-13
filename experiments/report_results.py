from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.registry import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RANKING_METRIC,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SHAP_BACKGROUND_SIZE,
    DEFAULT_SHAP_MAX_DISPLAY,
    DEFAULT_SHAP_ROWS_PER_CLASS,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TOP_K,
)
from experiments.reporting import run_reporting_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed fold runs, refit full-train models, score the "
            "held-out test split, export comparison CSVs/bar plots, and optionally "
            "run SHAP on the top winners."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the MCS wide CSV dataset.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Prepared holdout + CV split manifest JSON.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing fold outputs.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Directory where aggregated outputs will be written.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="Optional scope filter. Repeat to include multiple scopes.",
    )
    parser.add_argument(
        "--ranking-metric",
        default=DEFAULT_RANKING_METRIC,
        choices=("auprc", "auroc", "geometric_mean"),
        help="Metric used to rank top test-set winners.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top winners to explain with SHAP.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis and only export metrics/tables/plots.",
    )
    parser.add_argument(
        "--force-final-evaluation",
        action="store_true",
        help="Re-fit the final train/test models even if cached outputs exist.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help=(
            "Optional cap on how many completed primitives receive the expensive "
            "final train/test refit. Ranked by CV validation metrics."
        ),
    )
    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=DEFAULT_SHAP_BACKGROUND_SIZE,
        help="Approximate total number of test rows used for SHAP background.",
    )
    parser.add_argument(
        "--shap-explain-rows-per-class",
        type=int,
        default=DEFAULT_SHAP_ROWS_PER_CLASS,
        help="Number of test rows per class to explain.",
    )
    parser.add_argument(
        "--shap-max-display",
        type=int,
        default=DEFAULT_SHAP_MAX_DISPLAY,
        help="Maximum number of features to display in SHAP plots/tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scope_keys = set(args.scope) if args.scope else None
    run_reporting_workflow(
        results_root=args.results_root,
        report_root=args.report_root,
        dataset_path=args.dataset_path,
        split_manifest_path=args.split_manifest,
        scope_keys=scope_keys,
        ranking_metric=args.ranking_metric,
        top_k=args.top_k,
        skip_shap=args.skip_shap,
        force_final_evaluation=args.force_final_evaluation,
        final_top_k=args.final_top_k,
        shap_background_size=args.shap_background_size,
        shap_explain_rows_per_class=args.shap_explain_rows_per_class,
        shap_max_display=args.shap_max_display,
    )


if __name__ == "__main__":
    main()
