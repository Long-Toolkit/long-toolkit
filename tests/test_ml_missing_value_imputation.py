from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from long_toolkit.machine_learning.tools.longitudinal_machine_learning.estimators.merwav_time_minus_decision_tree import (
    MerWavTimeMinusDecisionTreeLongitudinalEstimatorTemplate,
)
from long_toolkit.machine_learning.tools.standard_machine_learning.estimators.logistic_regression import (
    LogisticRegressionEstimatorTemplate,
)
from long_toolkit.machine_learning.tools.templates.experiment_template import (
    ClassificationExperimentTemplate,
)
from long_toolkit.machine_learning.tools.templates.missing_value_imputation import (
    ImputedLongitudinalPipeline,
    MissingValueImputationConfig,
)
from long_toolkit.utils.errors import InputValidationError


def test_standard_experiment_can_opt_in_to_training_time_imputation(tmp_path) -> None:
    X = pd.DataFrame(
        {
            "score_1": [1.0, 2.0, None, 2.5, 1.5, 2.2],
            "score_2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], name="target")
    experiment = ClassificationExperimentTemplate()
    estimator = LogisticRegressionEstimatorTemplate.build_estimator(
        hyperparameters={},
        random_seed=42,
    )

    with pytest.raises(InputValidationError, match="Cross-validation failed"):
        experiment.run(
            estimator=estimator,
            estimator_name="Logistic Regression",
            X=X,
            y=y,
            metric_keys=("accuracy",),
            output_dir=tmp_path / "without_imputation",
            n_splits=2,
            evaluation_strategy="cross_validation",
            random_seed=42,
            silent_training_output=True,
            missing_value_imputation_enabled=False,
        )

    result = experiment.run(
        estimator=estimator,
        estimator_name="Logistic Regression",
        X=X,
        y=y,
        metric_keys=("accuracy",),
        output_dir=tmp_path / "with_imputation",
        n_splits=2,
        evaluation_strategy="cross_validation",
        random_seed=42,
        silent_training_output=True,
        missing_value_imputation_enabled=True,
        missing_value_imputation_strategy="mean",
    )

    assert isinstance(result.fitted_estimator, Pipeline)
    assert result.fitted_estimator.steps[0][0] == "missing_value_imputer"
    assert result.missing_value_imputation_enabled is True
    assert result.missing_value_imputation_strategy == "mean"


def test_longitudinal_workflow_uses_longitudinal_pipeline_wrapper_for_imputation() -> None:
    estimator = (
        MerWavTimeMinusDecisionTreeLongitudinalEstimatorTemplate.build_estimator(
            random_seed=42,
            hyperparameters={},
            feature_groups=((0, 1, 2), (3, 4, 5)),
            non_longitudinal_features=(),
            feature_list_names=(
                "mood_w1",
                "mood_w2",
                "mood_w3",
                "sleep_w1",
                "sleep_w2",
                "sleep_w3",
            ),
        )
    )
    X = pd.DataFrame(
        columns=[
            "mood_w1",
            "mood_w2",
            "mood_w3",
            "sleep_w1",
            "sleep_w2",
            "sleep_w3",
        ]
    )

    wrapped = ClassificationExperimentTemplate._ensure_model_pipeline(
        estimator=estimator,
        X=X,
        missing_value_imputation=MissingValueImputationConfig(
            enabled=True,
            strategy="median",
        ),
    )

    assert isinstance(wrapped, ImputedLongitudinalPipeline)
    assert wrapped.features_group == [[0, 1, 2], [3, 4, 5]]
    assert wrapped.missing_value_imputation_strategy == "median"
