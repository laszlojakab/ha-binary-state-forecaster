"""Hyper-parameters for online error tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )


class OnlineErrorTrackerHyperParameters:
    """
    Hyper parameters for hierarchical state statistics.

    This class encapsulates the base hyper parameters that influence the behavior
    of the hierarchical state statistics.
    """

    _hyper_parameters: Final[HyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    def __init__(self: Self, hyper_parameters: HyperParameters) -> None:
        """
        Initialize error tracker hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters providing reference half-life.

        """
        self._hyper_parameters: Final = hyper_parameters

    @property
    def half_life(self: Self) -> float:
        """Retrieves the half-life parameter from the base hyper parameters."""
        return self._hyper_parameters.half_life
