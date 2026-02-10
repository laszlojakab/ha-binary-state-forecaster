"""
Concept drift monitoring using dual-baseline comparison.

This module provides DriftMonitor, which detects concept drift by comparing
fast and slow exponentially weighted baseline distributions. When the distributions
diverge significantly, it indicates that the underlying data patterns are changing.
"""
import math
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.state import (
    State,
)

from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters
from .drift_stats import DriftStats
from .drift_stats_hyper_parameters import DriftStatsHyperParameters
from .duration_weighted_baseline import DurationWeightedBaseline
from .duration_weighted_baseline_hyper_parameters import (
    DurationWeightedBaselineHyperParameters,
)


class DriftMonitor:
    """
    Monitors concept drift using dual-baseline Jensen-Shannon divergence.

    Maintains fast and slow baseline distributions and compares them using
    Jensen-Shannon divergence to detect when the data distribution is changing.
    Uses consecutive threshold crossings with adaptive or fixed thresholds.

    Attributes:
        _hyper_parameters: Configuration for drift detection.
        _fast_baseline: Quickly adapting baseline for recent patterns.
        _slow_baseline: Slowly adapting baseline for established patterns.
        _drift_stats: Statistics of drift magnitudes for adaptive thresholds.
        _enter_counter: Count of consecutive high-drift updates.
        _exit_counter: Count of consecutive low-drift updates.
        _is_drifting: Current drift state.
        _last_drift: Most recent drift magnitude.
        _tau_enter: Current threshold for entering drift state.
        _tau_exit: Current threshold for exiting drift state.

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
        >>> hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
        >>> monitor = DriftMonitor(hp)
        >>> dist = {"on": 0.6, "off": 0.4}
        >>> monitor.update(dist, 100.0)
        >>> monitor.is_drifting
        False

    """

    def __init__(
        self: Self,
        hyper_parameters: DriftMonitorHyperParameters,
    ) -> None:
        """
        Initialize drift monitor with dual baselines.

        Args:
            hyper_parameters: Configuration controlling drift detection behavior.

        """
        self._hyper_parameters: Final = hyper_parameters
        self._fast_baseline: Final = DurationWeightedBaseline(
            DurationWeightedBaselineHyperParameters(
                hyper_parameters=hyper_parameters,
                half_life_factor=hyper_parameters.fast_half_life_factor,
                prune_threshold=hyper_parameters.fast_prune_threshold,
                epsilon=hyper_parameters.fast_epsilon,
            )
        )
        self._slow_baseline: Final = DurationWeightedBaseline(
            DurationWeightedBaselineHyperParameters(
                hyper_parameters=hyper_parameters,
                half_life_factor=hyper_parameters.slow_half_life_factor,
                prune_threshold=hyper_parameters.slow_prune_threshold,
                epsilon=hyper_parameters.slow_epsilon,
            )
        )

        self._drift_stats: Final = DriftStats(
            DriftStatsHyperParameters(
                hyper_parameters=hyper_parameters,
                half_life_factor=hyper_parameters.drift_half_life_factor,
            )
        )

        self._enter_counter = 0
        self._exit_counter = 0
        self._is_drifting = False
        self._last_drift: float = 0.0
        self._tau_enter = hyper_parameters.tau_enter
        self._tau_exit = hyper_parameters.tau_exit

    @property
    def is_drifting(self: Self) -> bool:
        """
        Check if drift is currently detected.

        Returns:
            True if in drifting state, False otherwise.

        """
        return self._is_drifting

    @property
    def last_drift(self: Self) -> float:
        """
        Get the most recent drift magnitude.

        Returns:
            Jensen-Shannon divergence between fast and slow baselines.

        """
        return self._last_drift

    def update(self: Self, dist: dict[State, float], timestamp: float) -> None:
        """
        Update drift monitor with new distribution observation.

        Updates both baselines, computes drift, and updates drift state using
        consecutive threshold crossing logic. When not drifting, drift statistics
        are updated and adaptive thresholds may be recomputed.

        Args:
            dist: Probability distribution over states (should sum to ~1.0).
            timestamp: Current timestamp for computing decay.

        """
        self._fast_baseline.update(dist, timestamp)
        self._slow_baseline.update(dist, timestamp)

        drift = self._compute_drift()

        self._last_drift = drift

        if not self._is_drifting:
            self._drift_stats.update(drift, timestamp)

            if self._hyper_parameters.adaptive_tau:
                self._tau_enter = self._drift_stats.mean + 3.0 * self._drift_stats.std
                self._tau_exit = self._drift_stats.mean + 1.5 * self._drift_stats.std

            # Entry logic: if the drift exceeds the entry threshold for n_enter consecutive updates,
            # we enter drifting state
            if drift >= self._tau_enter:
                self._enter_counter += 1
                if self._enter_counter >= self._hyper_parameters.n_enter:
                    self._is_drifting = True
                    self._exit_counter = 0
            else:
                self._enter_counter = 0

        elif drift <= self._tau_exit:
            # Exit logic: if the drift goes below the exit threshold for n_exit consecutive updates,
            # we exit drifting state
            self._exit_counter += 1
            if self._exit_counter >= self._hyper_parameters.n_exit:
                self._is_drifting = False
                self._enter_counter = 0
        else:
            # If we're drifting but the drift is above the exit threshold, we reset the exit counter
            self._exit_counter = 0

    def _compute_drift(self: Self) -> float:
        """
        Compute current drift magnitude using JS divergence.

        Returns:
            Jensen-Shannon divergence between fast and slow baseline distributions,
                or 0.0 if either distribution is empty.

        """
        p = self._fast_baseline.distribution()
        q = self._slow_baseline.distribution()
        if not p or not q:
            return 0.0
        return self._js_divergence(p, q)

    def _js_divergence(
        self: Self,
        p: dict[State, float],
        q: dict[State, float],
        eps: float = 1e-12,
    ) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.

        JS divergence is a symmetric measure of distance between probability
        distributions, bounded between 0 (identical) and 1 (completely different).

        Args:
            p: First probability distribution.
            q: Second probability distribution.
            eps: Small value to avoid log(0) errors.

        Returns:
            Jensen-Shannon divergence in bits (using log2).

        """
        keys = set(p) | set(q)
        js = 0.0

        for k in keys:
            pk = p.get(k, eps)
            qk = q.get(k, eps)
            mk = 0.5 * (pk + qk)

            js += 0.5 * pk * math.log2(pk / mk)
            js += 0.5 * qk * math.log2(qk / mk)

        return js
