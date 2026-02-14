"""
Hyper-parameters for drift statistics tracking.

This module provides configuration for DriftStats, controlling the decay rate
used when computing exponentially weighted drift statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

from custom_components.discrete_state_forecaster.model.learning.drift_stats_hyper_parameters import (
    DriftStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_runtime_parameters import (
    DriftStatsRuntimeParameters,
)

if TYPE_CHECKING:
    from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


@dataclass(frozen=True)
class DriftStatsParameters:
    """Configuration for drift statistics decay behavior."""

    hyper_parameters: Final[DriftStatsHyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    runtime_parameters: Final[DriftStatsRuntimeParameters]
    """Runtime parameters containing dynamic configuration values."""

    @property
    def drift_half_life(self: Self) -> float:
        """
        Get the half-life for drift statistics.

        Returns:
            The half-life (in same units as base) for drift statistics decay.

        """
        return (
            self.hyper_parameters.half_life * self.runtime_parameters.half_life_factor
        )
