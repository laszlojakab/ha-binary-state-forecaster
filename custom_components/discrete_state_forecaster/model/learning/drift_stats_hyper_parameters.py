"""Hyper-parameters for drift statistics tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DriftStatsHyperParameters:
    """Hyper parameters for drift statistics decay behavior."""

    _hyper_parameters: Final[DriftMonitorHyperParameters]
    """Parent drift monitor configuration."""

    def __init__(self: Self, hyper_parameters: DriftMonitorHyperParameters) -> None:
        """
        Initialize drift statistics hyper-parameters.

        Args:
            hyper_parameters: Parent drift monitor configuration.

        """
        self._hyper_parameters: Final = hyper_parameters

    @property
    def half_life(self: Self) -> float:
        """
        Gets the base half-life from the parent drift monitor hyper-parameters.

        Returns:
            The base half-life (in same units as base) for drift statistics decay.

        """
        return self._hyper_parameters.half_life
