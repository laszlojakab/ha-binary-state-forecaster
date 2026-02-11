"""
Hyper-parameters for drift statistics tracking.

This module provides configuration for DriftStats, controlling the decay rate
used when computing exponentially weighted drift statistics.
"""

from typing import Final, Self

from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DriftStatsHyperParameters:
    """
    Configuration for drift statistics decay behavior.

    Wraps DriftMonitorHyperParameters and applies a factor to determine
    the half-life for drift statistics. The drift half-life controls how
    quickly old drift measurements are forgotten.

    Attributes:
        _hyper_parameters: Parent drift monitor configuration.
        _half_life_factor: Multiplier applied to base half-life.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        ...     HyperParameters,
        ... )
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
