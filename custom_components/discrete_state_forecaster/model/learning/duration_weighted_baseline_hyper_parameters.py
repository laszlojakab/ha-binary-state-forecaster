from typing import Final, Self

from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DurationWeightedBaselineHyperParameters:
    def __init__(
        self: Self,
        *,
        hyper_parameters: DriftMonitorHyperParameters,
        half_life_factor: float,
        prune_threshold: float = 1e-6,
        epsilon: float = 1e-9,
    ):
        self._hyper_parameters: Final = hyper_parameters
        self._half_life_factor: Final = half_life_factor
        self._prune_threshold: Final = prune_threshold
        self._epsilon: Final = epsilon

    @property
    def baseline_half_life(self: Self) -> float:
        return self._hyper_parameters.half_life * self._half_life_factor

    @property
    def prune_threshold(self: Self) -> float:
        return self._prune_threshold

    @property
    def epsilon(self: Self) -> float:
        return self._epsilon
