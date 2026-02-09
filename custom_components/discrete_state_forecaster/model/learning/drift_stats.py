import math
from typing import Final, Self

from .drift_stats_hyper_parameters import DriftStatsHyperParameters


class DriftStats:
    def __init__(
        self: Self,
        hyper_parameters: DriftStatsHyperParameters,
    ):
        self._hyper_parameters: Final = hyper_parameters
        self._mean = 0.0
        self._var = 0.0
        self._last_ts = None

    def update(self: Self, x: float, timestamp: float):
        if self._last_ts is None:
            self._mean = x
            self._var = 0.0
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        lambda_ = math.log(2) / (self._hyper_parameters.drift_half_life)
        decay = math.exp(-lambda_ * dt)
        diff = x - self._mean

        self._mean = decay * self._mean + (1 - decay) * x
        self._var = decay * self._var + (1 - decay) * diff * diff
        self._last_ts = timestamp

    @property
    def std(self: Self) -> float:
        return math.sqrt(max(self._var, 1e-12))

    @property
    def mean(self: Self) -> float:
        return self._mean

    @property
    def var(self: Self) -> float:
        return self._var
