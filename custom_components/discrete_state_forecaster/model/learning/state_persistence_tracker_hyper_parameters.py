"""Hyper-parameters for state persistence tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )


class StatePersistenceTrackerHyperParameters:
    """Hyper-parameters for state persistence tracking behavior."""

    _hyper_parameters: Final[HyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    def __init__(
        self: Self,
        hyper_parameters: HyperParameters,
    ) -> None:
        """
        Initialize state persistence tracker hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters providing reference half-life.
        """
        self._hyper_parameters: Final = hyper_parameters

    @property
    def half_life(self: Self) -> float:
        """Gets the base half-life from the parent hyper-parameters."""
        return self._hyper_parameters.half_life
