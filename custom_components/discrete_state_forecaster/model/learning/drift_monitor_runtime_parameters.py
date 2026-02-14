"""Runtime parameters for drift monitoring."""

from dataclasses import dataclass
from typing import Any, Self

from custom_components.discrete_state_forecaster.model.learning.drift_stats_runtime_parameters import (  # noqa: E501
    DriftStatsRuntimeParameters,
)

from .duration_weighted_baseline_runtime_parameters import (
    DurationWeightedBaselineRuntimeParameters,
)


@dataclass()
class DriftMonitorRuntimeParameters:
    """Runtime parameters for concept drift detection."""

    slow_baseline: DurationWeightedBaselineRuntimeParameters
    """Runtime parameters for slow baseline behavior."""

    fast_baseline: DurationWeightedBaselineRuntimeParameters
    """Runtime parameters for fast baseline behavior."""

    drift_stats: DriftStatsRuntimeParameters
    """Runtime parameters for drift statistics behavior."""

    tau_enter: float
    """Fixed drift threshold to enter drifting state."""

    tau_exit: float
    """Fixed drift threshold to exit drifting state."""

    adaptive_tau: bool
    """Whether to compute thresholds adaptively based on drift history."""

    n_enter: int
    """Consecutive high-drift updates needed to enter drifting state."""

    n_exit: int
    """Consecutive low-drift updates needed to exit drifting state."""

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize runtime parameters to a dictionary.

        Returns:
            A dictionary containing all drift monitor runtime parameters.
        """
        return {
            "slow_baseline_runtime_parameters": self.slow_baseline.to_dict(),
            "fast_baseline_runtime_parameters": self.fast_baseline.to_dict(),
            "drift_stats_runtime_parameters": self.drift_stats.to_dict(),
            "tau_enter": self.tau_enter,
            "tau_exit": self.tau_exit,
            "adaptive_tau": self.adaptive_tau,
            "n_enter": self.n_enter,
            "n_exit": self.n_exit,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> Self:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing all drift monitor runtime parameters.

        Returns:
            A new DriftMonitorRuntimeParameters instance initialized with the provided data.
        """
        return cls(
            slow_baseline_runtime_parameters=DurationWeightedBaselineRuntimeParameters.from_dict(
                data["slow_baseline_runtime_parameters"]
            ),
            fast_baseline_runtime_parameters=DurationWeightedBaselineRuntimeParameters.from_dict(
                data["fast_baseline_runtime_parameters"]
            ),
            drift_stats_runtime_parameters=DriftStatsRuntimeParameters.from_dict(
                data["drift_stats_runtime_parameters"]
            ),
            tau_enter=data["tau_enter"],
            tau_exit=data["tau_exit"],
            adaptive_tau=data["adaptive_tau"],
            n_enter=data["n_enter"],
            n_exit=data["n_exit"],
        )
