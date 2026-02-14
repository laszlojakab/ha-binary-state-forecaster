"""
Hyper-parameters for duration-weighted baseline distribution.

This module provides configuration for DurationWeightedBaseline, controlling
the decay rate, pruning threshold, and Laplace smoothing parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters
    from .duration_weighted_baseline_runtime_parameters import (
        DurationWeightedBaselineRuntimeParameters,
    )


@dataclass(frozen=True)
class DurationWeightedBaselineParameters:
    """Configuration for duration-weighted baseline behavior."""

    hyper_parameters: Final[DriftMonitorHyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    runtime_parameters: Final[DurationWeightedBaselineRuntimeParameters]
    """Runtime parameters containing dynamic configuration values."""

    @property
    def baseline_half_life(self: Self) -> float:
        """
        Get the half-life for baseline decay.

        Returns:
            The half-life (in same units as base) for baseline mass decay.

        """
        return (
            self.hyper_parameters.half_life * self.runtime_parameters.half_life_factor
        )

    @property
    def prune_threshold(self: Self) -> float:
        """
        Get the pruning threshold for removing low-mass states.

        Returns:
            Minimum mass value below which states are pruned.

        """
        return self.runtime_parameters.prune_threshold

    @property
    def epsilon(self: Self) -> float:
        """
        Get the Laplace smoothing parameter.

        Returns:
            Small value added to all state probabilities for smoothing.

        """
        return self.runtime_parameters.epsilon
