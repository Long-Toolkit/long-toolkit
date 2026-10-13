from __future__ import annotations

from long_toolkit.machine_learning.tools.standard_machine_learning.estimators.extra_trees import (
    ExtraTreesEstimatorTemplate,
)
from long_toolkit.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingDTExtraTreesLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_stacking_dt__extra_trees` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_dt__extra_trees",
        full_name="SepWav (Stacking + Decision Tree) + Extra Trees",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Decision Tree. Base estimator: "
            "Extra Trees."
        ),
    )
    strategy_key = "sepwav_stacking_dt"
    base_estimator_key = "extra_trees"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_dt",
        standard_template=ExtraTreesEstimatorTemplate,
    )
