"""
Hyper-parameters for drift statistics tracking.

This module provides configuration for DriftStats, controlling the decay rate
used when computing exponentially weighted drift statistics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DriftStatsHyperParameters:
    """
    Configuration for drift statistics decay behavior.

    Wraps DriftMonitorHyperParameters and applies a factor to determine
    the half-life for drift statistics. The drift half-life controls how
    quickly old drift measurements are forgotten.

    Example:
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
        >>> hp = DriftStatsHyperParameters(
        ...     hyper_parameters=drift_hp,
        ...     half_life_factor=2.0,
        ... )
        >>> hp.drift_half_life
        100.0

    """

    _hyper_parameters: Final[DriftMonitorHyperParameters]
    """Parent drift monitor configuration."""

    _half_life_factor: Final[float]
    """Multiplier applied to base half-life for drift statistics decay."""

    def __init__(
        self: Self,
        *,
        hyper_parameters: DriftMonitorHyperParameters,
        half_life_factor: float,
    ) -> None:
        """
        Initialize drift statistics hyper-parameters.

        Args:
            hyper_parameters: Parent drift monitor configuration.
            half_life_factor: Multiplier for base half-life to get drift half-life.

        """
        self._hyper_parameters: Final = hyper_parameters
        self._half_life_factor: Final = half_life_factor

    @property
    def drift_half_life(self: Self) -> float:
        """
        Get the half-life for drift statistics.

        Returns:
            The half-life (in same units as base) for drift statistics decay.

        """
        return self._hyper_parameters.half_life * self._half_life_factor

    def to_dict(self: Self) -> dict[str, float]:
        """
        Serialize hyper-parameters to a dictionary.

        Returns:
            A dictionary containing the half-life factor for drift statistics.
        """
        return {"half_life_factor": self._half_life_factor}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, float],
        hyper_parameters: DriftMonitorHyperParameters,
    ) -> DriftStatsHyperParameters:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing the half_life_factor value.
            hyper_parameters: Parent drift monitor configuration providing the base
                half-life value used to compute the drift half-life.

        Returns:
            A new DriftStatsHyperParameters instance.
        """
        return cls(
            hyper_parameters=hyper_parameters,
            half_life_factor=data["half_life_factor"],
        )
