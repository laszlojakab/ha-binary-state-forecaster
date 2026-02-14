"""
Exponentially weighted statistics for drift measurements.

This module provides DriftStats, which maintains running mean and variance
of drift measurements using exponential decay. This is used to compute
adaptive thresholds for drift detection.
"""

import math
from typing import Any, Final, Self

from .drift_stats_hyper_parameters import DriftStatsHyperParameters
from .drift_stats_parameters import DriftStatsParameters
from .drift_stats_runtime_parameters import DriftStatsRuntimeParameters


class DriftStats:
    """
    Tracks exponentially weighted mean and variance of drift values.

    Maintains running statistics using exponential decay to give more weight
    to recent observations. Used to establish adaptive thresholds for drift
    detection based on historical drift patterns.

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
        ...     half_life_factor=1.0,
        ... )
        >>> stats = DriftStats(hp)
        >>> stats.update(0.1, 100.0)
        >>> stats.mean
        0.1

    """

    _parameters: Final[DriftStatsParameters]
    """Configuration controlling decay behavior."""

    _mean: float
    """Current mean of drift values."""

    _var: float
    """Current variance of drift values."""

    _last_ts: float | None
    """Timestamp of last update, or None if never updated."""

    def __init__(
        self: Self,
        hyper_parameters: DriftStatsHyperParameters,
        runtime_parameters: DriftStatsRuntimeParameters,
    ) -> None:
        """
        Initialize drift statistics tracker.

        Args:
            hyper_parameters: Configuration controlling decay behavior.
            runtime_parameters: Runtime parameters for drift statistics.
        """
        self._parameters: Final = DriftStatsParameters(
            hyper_parameters=hyper_parameters,
            runtime_parameters=runtime_parameters,
        )

        self._mean = 0.0
        self._var = 0.0
        self._last_ts: float | None = None

    def update(self: Self, x: float, timestamp: float) -> None:
        """
        Update statistics with new drift observation.

        Applies exponential decay to existing statistics, then incorporates
        the new value using exponentially weighted moving average and variance.

        Args:
            x: New drift measurement.
            timestamp: Current timestamp for computing decay.

        """
        if self._last_ts is None:
            self._mean = x
            self._var = 0.0
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        lambda_ = math.log(2) / (self._parameters.drift_half_life)
        decay = math.exp(-lambda_ * dt)
        diff = x - self._mean

        self._mean = decay * self._mean + (1 - decay) * x
        self._var = decay * self._var + (1 - decay) * diff * diff
        self._last_ts = timestamp

    @property
    def std(self: Self) -> float:
        """
        Get standard deviation of drift values.

        Returns:
            Standard deviation with a floor of 1e-6 to avoid division by zero.

        """
        return math.sqrt(max(self._var, 1e-12))

    @property
    def mean(self: Self) -> float:
        """
        Get mean of drift values.

        Returns:
            Current exponentially weighted mean of drift measurements.

        """
        return self._mean

    @property
    def var(self: Self) -> float:
        """
        Get variance of drift values.

        Returns:
            Current exponentially weighted variance of drift measurements.

        """
        return self._var

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the drift statistics to a dictionary.

        Returns:
            A dictionary containing hyper-parameters, mean, variance,
            and the timestamp of the last update.
        """
        return {
            "mean": self._mean,
            "var": self._var,
            "last_ts": self._last_ts,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        hyper_parameters: DriftStatsHyperParameters,
        runtime_parameters: DriftStatsRuntimeParameters,
    ) -> Self:
        """
        Deserialize drift statistics from a dictionary.

        Args:
            data: Dictionary containing serialized statistics including
                mean, variance, and last timestamp.
            hyper_parameters: Hyper-parameters to use for the reconstructed instance.
            runtime_parameters: Runtime parameters to use for the reconstructed instance.

        Returns:
            A new DriftStats instance initialized with the provided data,
            with all internal state restored.
        """
        stats = cls(
            hyper_parameters=hyper_parameters, runtime_parameters=runtime_parameters
        )
        stats._mean = data["mean"]
        stats._var = data["var"]
        stats._last_ts = data["last_ts"]
        return stats
