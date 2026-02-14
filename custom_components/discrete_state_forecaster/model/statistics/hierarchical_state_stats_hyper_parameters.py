"""Configuration hyper parameters for hierarchical state statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
        ForecasterEngineHyperParameters,
    )


class HierarchicalStateStatsHyperParameters:
    """
    Hyper parameters for hierarchical state statistics.

    This class encapsulates the base hyper parameters that influence the behavior
    of the hierarchical state statistics.
    """

    _hyper_parameters: Final[ForecasterEngineHyperParameters]
    """Base hyper parameters containing global configuration values."""

    def __init__(self: Self, hyper_parameters: ForecasterEngineHyperParameters):
        """
        Initializes hierarchical state statistics configuration.

        Args:
            hyper_parameters: Base hyper parameters containing global settings
                like half_life.

        """
        self._hyper_parameters: Final = hyper_parameters

    @property
    def half_life(self: Self) -> float:
        """
        Retrieves the half-life parameter from the base hyper parameters.

        Returns:
            The half-life value used for decay weighting in statistics calculations.

        """
        return self._hyper_parameters.half_life
