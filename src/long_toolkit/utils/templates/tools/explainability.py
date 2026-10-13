from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from beartype import beartype

from long_toolkit.utils.metadata import ComponentMetadata

SelfType = TypeVar("SelfType", bound="ExplainabilityTool")


@beartype
@dataclass(frozen=True)
class ToolParameterDefinition:
    """Describe one explainability tool parameter for wrappers and user interfaces.

    This schema is intentionally lightweight and is mainly consumed by external
    orchestration layers such as the Go CLI.
    """

    key: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


@beartype
class ExplainabilityTool:
    """Authoring template for every explainability tool in the Python API.

    An explainability tool reads a model that was already trained (e.g., by the
    modelling or benchmarking tools of `long_toolkit.machine_learning`) and
    explains its predictions on a dataset.

    A compliant explainability tool should:
        - expose a clear class name and a meaningful `metadata` object
        - accept `**kwargs` in `fit(...)` and `explain(...)`
        - validate input early and fail with explicit error messages
        - return Python-native payloads for library users
        - keep Go CLI translation and bridge formatting in separate `run.py`
          wrappers

    Method contract:
        - `fit(**kwargs)`: optional configuration step that parses, validates
          and stores the explanation settings.
        - `explain(**kwargs)`: required execution step that computes the
          explanation and writes its artefacts.
        - `fit_explain(**kwargs)`: convenience method that runs `fit(...)`
          followed by `explain(...)` with the same payload.
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Explainability Tool",
    )

    @classmethod
    @beartype
    def params_definition(cls) -> tuple[ToolParameterDefinition, ...]:
        """Return an optional parameter schema for wrappers or UI discovery."""

        return ()

    @beartype
    def fit(self: SelfType, **kwargs: Any) -> SelfType:
        """Run an optional fitting or configuration step.

        Args:
            **kwargs (Any): Tool-specific configuration values supplied by the
                caller. There is no stage-wide fixed key set at this level.
                Every concrete tool must document supported keys and value types
                in its own `fit(...)` docstring.

        Returns:
            SelfType: The fitted tool instance.
        """

        _ = kwargs
        return self

    @beartype
    def explain(self, **kwargs: Any) -> Any:
        """Execute the primary explainability behaviour.

        Args:
            **kwargs (Any): Tool-specific runtime values.

        Returns:
            Any: Tool-specific output payload.

        Raises:
            NotImplementedError: Always raised by the base template.
        """

        _ = kwargs
        raise NotImplementedError

    @beartype
    def fit_explain(self, **kwargs: Any) -> Any:
        """Run `fit(...)` then `explain(...)` with one shared payload.

        Args:
            **kwargs (Any): Tool-specific values forwarded unchanged to both
                `fit(...)` and `explain(...)`.

        Returns:
            Any: Tool-specific output payload returned by `explain(...)`.
        """

        self.fit(**kwargs)
        return self.explain(**kwargs)
