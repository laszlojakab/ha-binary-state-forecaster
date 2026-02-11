"""
Hyper-parameters for duration-weighted baseline distribution.

This module provides configuration for DurationWeightedBaseline, controlling
the decay rate, pruning threshold, and Laplace smoothing parameter.
"""

from typing import Final, Self

from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DurationWeightedBaselineHyperParameters:
    """
    Configuration for duration-weighted baseline behavior.

    Wraps DriftMonitorHyperParameters and provides configuration for a
    duration-weighted baseline including half-life factor, pruning threshold,
    and Laplace smoothing parameter.

    Attributes:
        _hyper_parameters: Parent drift monitor configuration.
        _half_life_factor: Multiplier applied to base half-life.
        _prune_threshold: Minimum mass to retain a state.
        _epsilon: Laplace smoothing parameter.

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
        >>> hp = DurationWeightedBaselineHyperParameters(
        ...     hyper_parameters=drift_hp,
        ...     half_life_factor=2.0,
        ... )
        >>> hp.baseline_half_life
        100.0

    """

    def __init__(
        self: Self,
        *,
        hyper_parameters: DriftMonitorHyperParameters,
        half_life_factor: float,
        prune_threshold: float = 1e-6,
        epsilon: float = 1e-9,
    ) -> None:
        """
        Initialize duration-weighted baseline hyper-parameters.

        Args:
            hyper_parameters: Parent drift monitor configuration.
            half_life_factor: Multiplier for base half-life.
            prune_threshold: Minimum mass to retain a state (default 1e-6).
            epsilon: Laplace smoothing parameter (default 1e-9).

        """
        self._hyper_parameters: Final = hyper_parameters
        self._half_life_factor: Final = half_life_factor
        self._prune_threshold: Final = prune_threshold
        self._epsilon: Final = epsilon

    @property
    def baseline_half_life(self: Self) -> float:
        """
        Get the half-life for baseline decay.

        Returns:
            The half-life (in same units as base) for baseline mass decay.

        """
        return self._hyper_parameters.half_life * self._half_life_factor

    @property
    def prune_threshold(self: Self) -> float:
        """
        Get the pruning threshold for removing low-mass states.

        Returns:
            Minimum mass value below which states are pruned.

        """
        return self._prune_threshold

    @property
    def epsilon(self: Self) -> float:
        """
        Get the Laplace smoothing parameter.

        Returns:
            Small value added to all state probabilities for smoothing.

        """
        return self._epsilon
