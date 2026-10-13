---
icon: lucide/history
---

# Longitudinal Machine Learning Tools

This page of the application programming interface (API) reference documents the longitudinal modelling tool of the [Benchmarking and Modelling Toolkit](../toolkits/benchmarking-and-modelling.md), which evaluates one longitudinal-data-aware estimator on repeated measures. The class docstring lists the feature-group inputs, the tuning parameters and the artefacts written.

!!! tip "Feature Groups Guidance"
    To understand how to configure and experiment with feature groups (temporal dependency style), read the [scikit-longitudinal temporal dependency tutorial](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/).

!!! info "Feature-group Input Modes"
    The longitudinal tools require an explicit `feature_groups_mode` and support three feature-group modes:

    - Manual: `feature_groups_mode="manual"` with `feature_groups="[[0,1,2],[3,4,5]]"` or `feature_groups="mood_w1,mood_w2,mood_w3;sleep_w1,sleep_w2,sleep_w3"`
    - Preset: `feature_groups_mode="preset"` with `feature_groups_preset="elsa"`
    - Suffix inference: `feature_groups_mode="suffix"` with `feature_groups_suffix="_w"`

    Non-longitudinal features also require an explicit `non_longitudinal_mode`:

    - Auto: `non_longitudinal_mode="auto"`
    - Manual: `non_longitudinal_mode="manual"` with `non_longitudinal_features="[6]"` or CSV names/indices

    In auto mode, every feature not used by `feature_groups` is treated as non-longitudinal.

## ::: long_toolkit.machine_learning.tools.longitudinal_machine_learning.run.LongitudinalMachineLearning
