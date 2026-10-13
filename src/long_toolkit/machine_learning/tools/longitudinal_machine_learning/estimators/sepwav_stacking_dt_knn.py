from __future__ import annotations

from long_toolkit.machine_learning.tools.standard_machine_learning.estimators.knn import (
    KNNEstimatorTemplate,
)
from long_toolkit.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingDTKNNLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `sepwav_stacking_dt__knn` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_dt__knn",
        full_name="SepWav (Stacking + Decision Tree) + K-Nearest Neighbours",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Decision Tree. Base estimator: "
            "K-Nearest Neighbours."
        ),
    )
    strategy_key = "sepwav_stacking_dt"
    base_estimator_key = "knn"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_dt",
        standard_template=KNNEstimatorTemplate,
    )
