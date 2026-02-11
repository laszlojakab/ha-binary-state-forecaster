"""
Hyper-parameters for drift monitoring.

This module provides configuration for the DriftMonitor, controlling how concept
drift is detected through comparison of fast and slow baseline distributions.
"""

from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class DriftMonitorHyperParameters:
    """
    Configuration for concept drift detection.

    Controls the behavior of drift detection including baseline half-lives,
    drift thresholds, and adaptive threshold adjustment.

    Attributes:
        _hyper_parameters: Base hyper-parameters.
        _slow_half_life_factor: Multiplier for slow baseline half-life.
        _slow_prune_threshold: Pruning threshold for slow baseline.
        _slow_epsilon: Laplace smoothing for slow baseline.
        _fast_half_life_factor: Multiplier for fast baseline half-life.
        _fast_prune_threshold: Pruning threshold for fast baseline.
        _fast_epsilon: Laplace smoothing for fast baseline.
        _drift_half_life_factor: Multiplier for drift statistics half-life.
        _tau_enter: Fixed drift threshold to enter drifting state.
        _tau_exit: Fixed drift threshold to exit drifting state.
        _adaptive_tau: Whether to compute thresholds adaptively.
        _n_enter: Consecutive high-drift updates needed to enter drifting.
        _n_exit: Consecutive low-drift updates needed to exit drifting.

    """

    def __init__(
        self: Self,
        *,
        hyper_parameters: HyperParameters,
        slow_half_life_factor: float = 20,
        slow_prune_threshold: float = 1e-6,
        slow_epsilon: float = 1e-9,
        fast_half_life_factor: float = 1.5,
        fast_prune_threshold: float = 1e-6,
        fast_epsilon: float = 1e-9,
        drift_half_life_factor: float = 30,
        tau_enter: float = 0.1,
        tau_exit: float = 0.05,
        adaptive_tau: bool = True,
        n_enter: int = 3,
        n_exit: int = 5,
    ):
        """
        Initialize drift monitor hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters.
            slow_half_life_factor: Slow baseline half-life = base * this.
            slow_prune_threshold: Slow baseline pruning threshold.
            slow_epsilon: Slow baseline Laplace smoothing.
            fast_half_life_factor: Fast baseline half-life = base * this.
            fast_prune_threshold: Fast baseline pruning threshold.
            fast_epsilon: Fast baseline Laplace smoothing.
            drift_half_life_factor: Drift stats half-life = base * this.
            tau_enter: Fixed threshold to enter drifting (if not adaptive).
            tau_exit: Fixed threshold to exit drifting (if not adaptive).
            adaptive_tau: Whether to adapt thresholds based on drift history.
            n_enter: Consecutive updates above threshold to enter drifting.
            n_exit: Consecutive updates below threshold to exit drifting.

        """
        self._hyper_parameters: Final = hyper_parameters
        self._slow_half_life_factor: Final = slow_half_life_factor
        self._slow_prune_threshold: Final = slow_prune_threshold
        self._slow_epsilon: Final = slow_epsilon
        self._fast_half_life_factor: Final = fast_half_life_factor
        self._fast_prune_threshold: Final = fast_prune_threshold
        self._fast_epsilon: Final = fast_epsilon
        self._drift_half_life_factor: Final = drift_half_life_factor
        self._tau_enter: Final = tau_enter
        self._tau_exit: Final = tau_exit
        self._adaptive_tau: Final = adaptive_tau
        self._n_enter: Final = n_enter
        self._n_exit: Final = n_exit

    @property
    def half_life(self: Self) -> float:
        """Get base half-life from underlying hyper-parameters."""
        return self._hyper_parameters.half_life

    @property
    def slow_half_life_factor(self: Self) -> float:
        """Get slow baseline half-life multiplier."""
        return self._slow_half_life_factor

    @property
    def slow_prune_threshold(self: Self) -> float:
        """Get slow baseline pruning threshold."""
        return self._slow_prune_threshold

    @property
    def slow_epsilon(self: Self) -> float:
        """Get slow baseline Laplace smoothing parameter."""
        return self._slow_epsilon

    @property
    def fast_half_life_factor(self: Self) -> float:
        """Get fast baseline half-life multiplier."""
        return self._fast_half_life_factor

    @property
    def fast_prune_threshold(self: Self) -> float:
        """Get fast baseline pruning threshold."""
        return self._fast_prune_threshold

    @property
    def fast_epsilon(self: Self) -> float:
        """Get fast baseline Laplace smoothing parameter."""
        return self._fast_epsilon

    @property
    def drift_half_life_factor(self: Self) -> float:
        """Get drift statistics half-life multiplier."""
        return self._drift_half_life_factor

    @property
    def tau_enter(self: Self) -> float:
        """Get fixed threshold for entering drifting state."""
        return self._tau_enter

    @property
    def tau_exit(self: Self) -> float:
        """Get fixed threshold for exiting drifting state."""
        return self._tau_exit

    @property
    def adaptive_tau(self: Self) -> bool:
        """Get whether thresholds are computed adaptively."""
        return self._adaptive_tau

    @property
    def n_enter(self: Self) -> int:
        """Get number of consecutive high-drift updates to enter drifting."""
        return self._n_enter

    @property
    def n_exit(self: Self) -> int:
        """Get number of consecutive low-drift updates to exit drifting."""
        return self._n_exit
