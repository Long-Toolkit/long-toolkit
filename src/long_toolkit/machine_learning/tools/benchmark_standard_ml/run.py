from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beartype import beartype

from long_toolkit.machine_learning.catalog import resolve_technique_with_defaults
from long_toolkit.machine_learning.support.inputs import (
    as_bool,
    as_optional_object,
    as_optional_string,
    as_required_int,
    as_required_string,
    load_input_dataset,
    parse_excluded_estimators,
    parse_metric_keys,
    resolve_evaluation_config,
    resolve_feature_columns,
    resolve_notebook_path,
    resolve_target_column,
    run_with_validation,
)
from long_toolkit.machine_learning.support.runtime_validation import (
    filter_supported_hyperparameters,
    validate_evaluation_split_feasibility,
    validate_one_vs_rest_evaluation_splits,
)
from long_toolkit.machine_learning.tools.metrics import list_supported_metrics
from long_toolkit.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from long_toolkit.machine_learning.tools.target_scenarios import TargetScenarioPlanner
from long_toolkit.machine_learning.tools.templates import (
    BenchmarkEstimatorSpec,
    ClassificationBenchmarkTemplate,
    StandardMLBenchmarkNotebookConfig,
    StandardMLBenchmarkNotebookTemplate,
)
from long_toolkit.machine_learning.tools.templates.missing_value_imputation import (
    validate_missing_value_imputation_strategy,
)
from long_toolkit.utils.errors import InputValidationError
from long_toolkit.utils.metadata import ComponentMetadata, resolve_component_metadata
from long_toolkit.utils.templates.tools.machine_learning import MachineLearningTool

_BENCHMARK_NAME = "benchmark_standard_ml"


@beartype
class BenchmarkStandardML(MachineLearningTool):
    """Benchmark standard machine-learning estimators on one dataset.

    Estimator references:

    - [Standard Machine Learning API reference](https://long-toolkit.readthedocs.io/en/latest/api/standard-machine-learning/)

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns benchmark estimator roster metadata. |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_benchmark` | Trains/evaluates included estimators and ranks them. |

    Examples:
        ```python
        from long_toolkit.machine_learning import BenchmarkStandardML

        tool = BenchmarkStandardML()
        result = tool.fit_predict(
            technique="run_benchmark",
            input_path="./dataset.csv",
            target_column="outcome_class",
            feature_columns="age,sex,sleep_score,anxiety_score",
            excluded_estimators="",
            metric_keys="accuracy,roc_auc",
            evaluation_strategy="cross_validation",
            n_splits=5,
            multiclass_mode="multiclass",
            random_seed=42,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="benchmark_standard_ml",
        full_name="Benchmark Standard Machine Learning",
        abstract_description="Benchmark cross-sectional classifiers and rank results.",
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> BenchmarkStandardML:
        """Validate and store one execution payload.

        Args:
            **kwargs (Any): Configuration keys:

                - `technique` (str): `list_estimators`, `list_metrics`, or
                  `run_benchmark`.
                - `params` (Mapping[str, Any] | None): Optional parameter object.
                - for `run_benchmark`, expected keys include:
                  `input_path`, `target_column`, `feature_columns`,
                  `excluded_estimators`, `metric_keys`, `evaluation_strategy`,
                  `n_splits`, `holdout_split`, `multiclass_mode`, `random_seed`,
                  `output_root_dir`, `max_parallel_candidates`,
                  `hyperparameters_json`,
                  `silent_training_output`, `generate_notebook`,
                  `notebook_output_path`,
                  `include_pipeline_profiler_at_end`, and
                  `generate_pipeline_profiler_html`.
                - for `list_estimators` and `list_metrics`, no extra keys are
                  required.
                - any additional keys are merged into `params` as direct
                  library-friendly shorthand.

        Returns:
            BenchmarkStandardML: The fitted tool instance.
        """

        technique_raw = kwargs.get("technique", self._technique)
        if not isinstance(technique_raw, str) or not technique_raw.strip():
            raise InputValidationError("Missing required string parameter: technique")

        params_payload: dict[str, Any] = {}
        params_raw = kwargs.get("params")
        if params_raw is not None:
            if not isinstance(params_raw, Mapping):
                raise InputValidationError("`params` must be an object.")
            params_payload.update(dict(params_raw))
        params_payload.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"technique", "params"}
            }
        )

        self._technique = technique_raw.strip()
        self._params = params_payload
        return self

    @beartype
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the configured benchmark workflow.

        Args:
            **kwargs (Any): Optional keys identical to `fit(...)`. If provided,
                they override any existing fitted configuration. Expected keys:
                `technique` and optional `params`, or direct shorthand keys for
                `run_benchmark`:
                `input_path`, `target_column`, `feature_columns`,
                `excluded_estimators`, `metric_keys`, `evaluation_strategy`,
                `n_splits`, `holdout_split`, `multiclass_mode`, `random_seed`,
                `output_root_dir`, `max_parallel_candidates`,
                `hyperparameters_json`,
                `silent_training_output`, `generate_notebook`,
                `notebook_output_path`,
                `include_pipeline_profiler_at_end`, and
                `generate_pipeline_profiler_html`.

        Returns:
            dict[str, Any]: Technique-specific benchmark result payload.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._technique is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `predict(...)`."
            )

        return run_with_validation(
            lambda: _run_benchmark_standard_ml_tool(
                technique=self._technique or "",
                params=self._params,
            )
        )

    @beartype
    def predict_proba(self, **kwargs: Any) -> dict[str, Any]:
        """Benchmark tool-level probability payloads are not exposed.

        Args:
            **kwargs (Any): Ignored. Present for template contract compliance.

        Returns:
            dict[str, Any]: Never returned.

        Raises:
            InputValidationError: Always raised because this benchmark tool
                returns ranking metadata rather than raw probability arrays.
        """

        _ = kwargs
        raise InputValidationError(
            "`BenchmarkStandardML` does not expose tool-level `predict_proba(...)`."
        )


def run_benchmark_standard_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one standard benchmark technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `BenchmarkStandardML` directly from `long_toolkit.machine_learning`.

    Args:
        technique (str): Workflow technique (`list_estimators`, `list_metrics`,
            or `run_benchmark`).
        params (Mapping[str, Any]): Technique parameters. For `run_benchmark`,
            expected keys include:
            `input_path`, `target_column`, `feature_columns`,
            `excluded_estimators`, `metric_keys`, `evaluation_strategy`,
            `n_splits`, `holdout_split`, `multiclass_mode`, `random_seed`;
            optional keys:
            `output_root_dir`, `max_parallel_candidates`, `hyperparameters_json`,
            `silent_training_output`, `generate_notebook`,
            `notebook_output_path`, `include_pipeline_profiler_at_end`, and
            `generate_pipeline_profiler_html`.

    Returns:
        dict[str, Any]: Technique-specific serialised result payload.
    """

    return BenchmarkStandardML().fit_predict(
        technique=technique,
        params=params,
    )


def _run_benchmark_standard_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="benchmark_standard_ml",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode == "list_estimators":
        estimators = discover_standard_estimators()
        return {
            "estimators": [
                {
                    "key": key,
                    "name": resolve_component_metadata(template).full_name,
                    "description": resolve_component_metadata(
                        template
                    ).abstract_description,
                }
                for key, template in estimators.items()
            ],
            "count": len(estimators),
        }

    if mode == "list_metrics":
        metrics = list_supported_metrics()
        return {
            "metrics": [
                {
                    "key": metric.key,
                    "source": metric.source,
                    "description": metric.description,
                }
                for metric in metrics
            ],
            "count": len(metrics),
        }

    if mode != "run_benchmark":
        raise InputValidationError(
            f"Unsupported benchmark-standard-ml technique: {technique}"
        )

    estimators = discover_standard_estimators()
    if not estimators:
        raise InputValidationError("No cross-sectional estimators were found.")

    excluded = parse_excluded_estimators(
        as_required_string(resolved, "excluded_estimators")
    )
    included_estimators = {
        key: template for key, template in estimators.items() if key not in excluded
    }
    if not included_estimators:
        raise InputValidationError(
            "All estimators were excluded. Include at least one estimator."
        )
    effective_max_parallel_candidates = 1

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    dataset = load_input_dataset(input_path=input_path)

    target_column = resolve_target_column(
        dataset=dataset,
        requested=as_required_string(resolved, "target_column"),
    )
    feature_columns = resolve_feature_columns(
        dataset=dataset,
        target_column=target_column,
        raw_feature_columns=as_required_string(resolved, "feature_columns"),
    )

    modelling_data = dataset[[*feature_columns, target_column]].dropna(
        subset=[target_column]
    )
    if modelling_data.empty:
        raise InputValidationError(
            "No rows available after dropping missing target values."
        )

    X = modelling_data[feature_columns].copy()
    y = modelling_data[target_column].copy()

    metric_keys = parse_metric_keys(as_required_string(resolved, "metric_keys"))
    evaluation = resolve_evaluation_config(
        evaluation_strategy_raw=as_required_string(resolved, "evaluation_strategy"),
        n_splits=as_required_int(resolved, "n_splits", minimum=2),
        holdout_split_raw=as_optional_string(resolved, "holdout_split"),
    )
    multiclass_mode = as_required_string(resolved, "multiclass_mode").strip().lower()
    if multiclass_mode not in {"multiclass", "one_vs_rest"}:
        raise InputValidationError(
            "multiclass_mode must be `multiclass` or `one_vs_rest`."
        )

    random_seed = as_required_int(resolved, "random_seed")
    missing_value_imputation_enabled = as_bool(
        resolved.get("missing_value_imputation_enabled", False),
        field_name="missing_value_imputation_enabled",
    )
    missing_value_imputation_strategy = validate_missing_value_imputation_strategy(
        strategy=as_optional_string(resolved, "missing_value_imputation_strategy")
        or "mean"
    )
    max_parallel_candidates = as_required_int(
        resolved,
        "max_parallel_candidates",
        minimum=1,
    )
    effective_max_parallel_candidates = min(
        max_parallel_candidates,
        len(included_estimators),
    )
    silent_training_output = as_bool(
        resolved.get("silent_training_output", True),
        field_name="silent_training_output",
    )

    output_root_raw = as_optional_string(resolved, "output_root_dir")
    output_root_dir = (
        Path(output_root_raw).expanduser()
        if output_root_raw is not None
        else Path("outputs/benchmarks/standard_ml").expanduser()
    )
    hyperparameter_overrides = as_optional_object(resolved, "hyperparameters_json")

    generate_notebook = as_bool(
        resolved.get("generate_notebook", True),
        field_name="generate_notebook",
    )
    notebook_path = resolve_notebook_path(
        generate_notebook=generate_notebook,
        requested_path=as_optional_string(resolved, "notebook_output_path"),
        default_path=output_root_dir
        / "benchmark_standard_ml_reproducibility_notebook.ipynb",
    )
    include_pipeline_profiler_at_end = as_bool(
        resolved.get("include_pipeline_profiler_at_end", False),
        field_name="include_pipeline_profiler_at_end",
    )
    generate_pipeline_profiler_html = as_bool(
        resolved.get("generate_pipeline_profiler_html", False),
        field_name="generate_pipeline_profiler_html",
    )

    estimator_specs = tuple(
        BenchmarkEstimatorSpec(
            estimator_key=key,
            estimator_name=resolve_component_metadata(template).full_name,
            estimator=template.build_estimator(
                hyperparameters=filter_supported_hyperparameters(
                    estimator_template=template,
                    hyperparameter_overrides=hyperparameter_overrides,
                ),
                random_seed=random_seed,
            ),
        )
        for key, template in included_estimators.items()
    )

    unique_class_count = int(y.nunique(dropna=True))
    if unique_class_count < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    if unique_class_count > 2 and multiclass_mode == "one_vs_rest":
        scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y)
        validate_one_vs_rest_evaluation_splits(
            n_splits=evaluation.n_splits,
            evaluation_strategy=evaluation.evaluation_strategy,
            holdout_split=evaluation.holdout_split,
            scenarios=scenarios,
        )

        benchmark_runs: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_output_root = output_root_dir / scenario.scenario_key
            benchmark = ClassificationBenchmarkTemplate().run(
                benchmark_name=_BENCHMARK_NAME,
                estimator_specs=estimator_specs,
                X=X,
                y=scenario.y_binary,
                metric_keys=metric_keys,
                output_root_dir=scenario_output_root,
                n_splits=evaluation.n_splits,
                evaluation_strategy=evaluation.evaluation_strategy,
                random_seed=random_seed,
                holdout_split=evaluation.holdout_split,
                silent_training_output=silent_training_output,
                missing_value_imputation_enabled=missing_value_imputation_enabled,
                missing_value_imputation_strategy=missing_value_imputation_strategy,
                max_parallel_candidates=max_parallel_candidates,
                generate_pipeline_profiler_html=generate_pipeline_profiler_html,
            )
            benchmark_runs.append(
                {
                    "scenario_key": scenario.scenario_key,
                    "scenario_label": scenario.scenario_label,
                    "benchmark_output_dir": str(
                        benchmark.artifacts.benchmark_output_dir.resolve()
                    ),
                    "ranking_csv": str(benchmark.artifacts.ranking_path.resolve()),
                    "summary_json": str(benchmark.artifacts.summary_path.resolve()),
                    "report_path": str(benchmark.artifacts.report_path.resolve()),
                    "ranked_estimators": [
                        {
                            "rank": idx + 1,
                            "estimator_key": item.estimator_key,
                            "estimator_name": item.estimator_name,
                            "mean_score": item.mean_score,
                            "std_score": item.std_score,
                            "model_artifact": str(item.artifacts.model_path.resolve()),
                        }
                        for idx, item in enumerate(benchmark.ranked_estimators)
                    ],
                }
            )

        saved_notebook = None
        if notebook_path is not None:
            notebook_cfg = StandardMLBenchmarkNotebookConfig(
                notebook_path=notebook_path,
                dataset_path=input_path,
                output_root_dir=output_root_dir,
                benchmark_name=_BENCHMARK_NAME,
                excluded_estimators=tuple(sorted(excluded)),
                hyperparameters_json=hyperparameter_overrides,
                multiclass_mode=multiclass_mode,
                target_column=target_column,
                feature_columns=tuple(feature_columns),
                metric_keys=metric_keys,
                evaluation_strategy=evaluation.evaluation_strategy,
                n_splits=evaluation.n_splits,
                holdout_split=evaluation.holdout_split,
                random_seed=random_seed,
                silent_training_output=silent_training_output,
                missing_value_imputation_enabled=missing_value_imputation_enabled,
                missing_value_imputation_strategy=missing_value_imputation_strategy,
                max_parallel_candidates=max_parallel_candidates,
                include_pipeline_profiler_at_end=include_pipeline_profiler_at_end,
            )
            saved_notebook = StandardMLBenchmarkNotebookTemplate().write_notebook(
                config=notebook_cfg
            )

        return {
            "mode": "one_vs_rest",
            "benchmark_name": _BENCHMARK_NAME,
            "evaluation_strategy": evaluation.evaluation_strategy,
            "n_splits": evaluation.n_splits,
            "holdout_split": evaluation.holdout_split,
            "metric_keys": list(metric_keys),
            "max_parallel_candidates": effective_max_parallel_candidates,
            "included_estimators": len(included_estimators),
            "excluded_estimators": sorted(excluded),
            "runs": benchmark_runs,
            "scenario_count": len(benchmark_runs),
            "output_root_dir": str(output_root_dir.resolve()),
            "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
            "missing_value_imputation_enabled": missing_value_imputation_enabled,
            "missing_value_imputation_strategy": missing_value_imputation_strategy,
            "dropped_rows": int(len(dataset) - len(modelling_data)),
        }

    validate_evaluation_split_feasibility(
        y=y,
        n_splits=evaluation.n_splits,
        evaluation_strategy=evaluation.evaluation_strategy,
        holdout_split=evaluation.holdout_split,
    )

    benchmark = ClassificationBenchmarkTemplate().run(
        benchmark_name=_BENCHMARK_NAME,
        estimator_specs=estimator_specs,
        X=X,
        y=y,
        metric_keys=metric_keys,
        output_root_dir=output_root_dir,
        n_splits=evaluation.n_splits,
        evaluation_strategy=evaluation.evaluation_strategy,
        random_seed=random_seed,
        holdout_split=evaluation.holdout_split,
        silent_training_output=silent_training_output,
        missing_value_imputation_enabled=missing_value_imputation_enabled,
        missing_value_imputation_strategy=missing_value_imputation_strategy,
        max_parallel_candidates=max_parallel_candidates,
        generate_pipeline_profiler_html=generate_pipeline_profiler_html,
    )

    saved_notebook = None
    if notebook_path is not None:
        notebook_cfg = StandardMLBenchmarkNotebookConfig(
            notebook_path=notebook_path,
            dataset_path=input_path,
            output_root_dir=output_root_dir,
            benchmark_name=_BENCHMARK_NAME,
            excluded_estimators=tuple(sorted(excluded)),
            hyperparameters_json=hyperparameter_overrides,
            multiclass_mode=multiclass_mode,
            target_column=target_column,
            feature_columns=tuple(feature_columns),
            metric_keys=metric_keys,
            evaluation_strategy=evaluation.evaluation_strategy,
            n_splits=evaluation.n_splits,
            holdout_split=evaluation.holdout_split,
            random_seed=random_seed,
            silent_training_output=silent_training_output,
            missing_value_imputation_enabled=missing_value_imputation_enabled,
            missing_value_imputation_strategy=missing_value_imputation_strategy,
            max_parallel_candidates=max_parallel_candidates,
            include_pipeline_profiler_at_end=include_pipeline_profiler_at_end,
        )
        saved_notebook = StandardMLBenchmarkNotebookTemplate().write_notebook(
            config=notebook_cfg
        )

    ranking = [
        {
            "rank": idx + 1,
            "estimator_key": item.estimator_key,
            "estimator_name": item.estimator_name,
            "mean_score": item.mean_score,
            "std_score": item.std_score,
            "model_artifact": str(item.artifacts.model_path.resolve()),
        }
        for idx, item in enumerate(benchmark.ranked_estimators)
    ]

    return {
        "mode": "multiclass",
        "benchmark_name": _BENCHMARK_NAME,
        "metric_key": benchmark.metric_key,
        "metric_keys": list(benchmark.metric_keys),
        "evaluation_strategy": benchmark.evaluation_strategy,
        "n_splits": benchmark.n_splits,
        "holdout_split": benchmark.holdout_split,
        "max_parallel_candidates": benchmark.max_parallel_candidates,
        "included_estimators": len(included_estimators),
        "excluded_estimators": sorted(excluded),
        "ranked_estimators": ranking,
        "skipped_estimators": [
            {
                "estimator_key": skipped.estimator_key,
                "reason": skipped.reason,
            }
            for skipped in benchmark.skipped_estimators
        ],
        "benchmark_output_dir": str(benchmark.artifacts.benchmark_output_dir.resolve()),
        "ranking_csv": str(benchmark.artifacts.ranking_path.resolve()),
        "summary_json": str(benchmark.artifacts.summary_path.resolve()),
        "report_path": str(benchmark.artifacts.report_path.resolve()),
        "pipeline_profiler_json": str(
            benchmark.artifacts.pipeline_profiler.input_json_path.resolve()
        ),
        "pipeline_profiler_html": (
            str(benchmark.artifacts.pipeline_profiler.html_path.resolve())
            if benchmark.artifacts.pipeline_profiler.html_path is not None
            else None
        ),
        "pipeline_profiler_warning": benchmark.artifacts.pipeline_profiler.warning,
        "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
        "missing_value_imputation_enabled": missing_value_imputation_enabled,
        "missing_value_imputation_strategy": missing_value_imputation_strategy,
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
