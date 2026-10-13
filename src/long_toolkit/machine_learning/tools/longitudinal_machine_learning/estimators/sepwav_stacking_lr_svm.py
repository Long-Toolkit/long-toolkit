from __future__ import annotations

from long_toolkit.machine_learning.tools.standard_machine_learning.estimators.svm import (
    SVMEstimatorTemplate,
)
from long_toolkit.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingLRSVMLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `sepwav_stacking_lr__svm` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_lr__svm",
        full_name="SepWav (Stacking + Logistic Regression) + Support Vector Machine",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Logistic Regression. Base estimator: "
            "Support Vector Machine."
        ),
    )
    strategy_key = "sepwav_stacking_lr"
    base_estimator_key = "svm"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_lr",
        standard_template=SVMEstimatorTemplate,
    )
