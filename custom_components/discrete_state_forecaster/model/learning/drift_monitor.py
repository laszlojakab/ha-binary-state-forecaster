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
    def __init__(
        self: Self,
        hyper_parameters: DriftMonitorHyperParameters,
    ) -> None:
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

    @property
    def is_drifting(self: Self) -> bool:
        return self._is_drifting

    @property
    def last_drift(self: Self) -> float:
        return self._last_drift

    def update(self: Self, dist: dict[State, float], timestamp: float):
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

    def _compute_drift(self) -> float:
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
        keys = set(p) | set(q)
        js = 0.0

        for k in keys:
            pk = p.get(k, eps)
            qk = q.get(k, eps)
            mk = 0.5 * (pk + qk)

            js += 0.5 * pk * math.log2(pk / mk)
            js += 0.5 * qk * math.log2(qk / mk)

        return js
