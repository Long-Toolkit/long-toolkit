from __future__ import annotations

from long_toolkit.machine_learning.tools.standard_machine_learning.estimators.extra_trees import (
    ExtraTreesEstimatorTemplate,
)
from long_toolkit.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMeanExtraTreesLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_mean__extra_trees` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_mean__extra_trees",
        full_name="AggrFunc (Mean) + Extra Trees",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using mean. Base estimator: "
            "Extra Trees."
        ),
    )
    strategy_key = "aggrfunc_mean"
    base_estimator_key = "extra_trees"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_mean",
        standard_template=ExtraTreesEstimatorTemplate,
    )
