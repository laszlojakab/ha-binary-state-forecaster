"""
Hyper-parameters for drift monitoring.

This module provides configuration for the DriftMonitor, controlling how concept
drift is detected through comparison of fast and slow baseline distributions.
"""

from dataclasses import dataclass
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)

# TODO: make hyper and runtime fields private... in all parameter classes!


@dataclass(frozen=True)
class DriftMonitorParameters:
    """
    Configuration for concept drift detection.

    Controls the behavior of drift detection including baseline half-lives,
    drift thresholds, and adaptive threshold adjustment.
    """

    hyper_parameters: Final[ForecasterEngineHyperParameters]
    """Base hyper-parameters providing the half-life value used for baselines."""

    runtime_parameters: Final[DriftMonitorRuntimeParameters]
    """Runtime parameters containing dynamic configuration values."""

    @property
    def tau_enter(self: Self) -> float:
        """Get fixed threshold for entering drifting state."""
        return self.runtime_parameters.tau_enter

    @property
    def tau_exit(self: Self) -> float:
        """Get fixed threshold for exiting drifting state."""
        return self.runtime_parameters.tau_exit

    @property
    def adaptive_tau(self: Self) -> bool:
        """Get whether thresholds are computed adaptively."""
        return self.runtime_parameters.adaptive_tau

    @property
    def n_enter(self: Self) -> int:
        """Get number of consecutive high-drift updates to enter drifting."""
        return self.runtime_parameters.n_enter

    @property
    def n_exit(self: Self) -> int:
        """Get number of consecutive low-drift updates to exit drifting."""
        return self.runtime_parameters.n_exit
