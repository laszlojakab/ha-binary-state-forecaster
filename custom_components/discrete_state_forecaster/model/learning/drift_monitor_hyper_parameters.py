"""
Hyper-parameters for drift monitoring.

This module provides configuration for the DriftMonitor, controlling how concept
drift is detected through comparison of fast and slow baseline distributions.
"""

from typing import Any, Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class DriftMonitorHyperParameters:
    """
    Configuration for concept drift detection.

    Controls the behavior of drift detection including baseline half-lives,
    drift thresholds, and adaptive threshold adjustment.
    """

    _hyper_parameters: Final[HyperParameters]
    """Base hyper-parameters providing the half-life value used for baselines."""

    _slow_half_life_factor: Final[float]
    """Multiplier applied to base half-life to get slow baseline half-life."""

    _slow_prune_threshold: Final[float]
    """Pruning threshold for slow baseline to remove low-probability states."""

    _slow_epsilon: Final[float]
    """Laplace smoothing parameter for slow baseline to handle zero counts."""

    _fast_half_life_factor: Final[float]
    """Multiplier applied to base half-life to get fast baseline half-life."""

    _fast_prune_threshold: Final[float]
    """Pruning threshold for fast baseline to remove low-probability states."""

    _fast_epsilon: Final[float]
    """Laplace smoothing parameter for fast baseline to handle zero counts."""

    _drift_half_life_factor: Final[float]
    """Multiplier applied to base half-life to get drift statistics half-life."""

    _tau_enter: Final[float]
    """Fixed drift threshold to enter drifting state."""

    _tau_exit: Final[float]
    """Fixed drift threshold to exit drifting state."""

    _adaptive_tau: Final[bool]
    """Whether to compute thresholds adaptively based on drift history."""

    _n_enter: Final[int]
    """Consecutive high-drift updates needed to enter drifting state."""

    _n_exit: Final[int]
    """Consecutive low-drift updates needed to exit drifting state."""

    def __init__(  # noqa: PLR0913
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

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize hyper-parameters to a dictionary.

        Returns:
            A dictionary containing all drift monitor hyper-parameters.
        """
        return {
            "half_life": self.half_life,
            "slow_half_life_factor": self.slow_half_life_factor,
            "slow_prune_threshold": self.slow_prune_threshold,
            "slow_epsilon": self.slow_epsilon,
            "fast_half_life_factor": self.fast_half_life_factor,
            "fast_prune_threshold": self.fast_prune_threshold,
            "fast_epsilon": self.fast_epsilon,
            "drift_half_life_factor": self.drift_half_life_factor,
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
        hyper_parameters: HyperParameters,
    ) -> Self:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing all drift monitor hyper-parameters.
            hyper_parameters: Base hyper-parameters to use for the instance.

        Returns:
            A new DriftMonitorHyperParameters instance initialized with the provided data.
        """
        return cls(
            hyper_parameters=hyper_parameters,
            slow_half_life_factor=data["slow_half_life_factor"],
            slow_prune_threshold=data["slow_prune_threshold"],
            slow_epsilon=data["slow_epsilon"],
            fast_half_life_factor=data["fast_half_life_factor"],
            fast_prune_threshold=data["fast_prune_threshold"],
            fast_epsilon=data["fast_epsilon"],
            drift_half_life_factor=data["drift_half_life_factor"],
            tau_enter=data["tau_enter"],
            tau_exit=data["tau_exit"],
            adaptive_tau=data["adaptive_tau"],
            n_enter=data["n_enter"],
            n_exit=data["n_exit"],
        )
