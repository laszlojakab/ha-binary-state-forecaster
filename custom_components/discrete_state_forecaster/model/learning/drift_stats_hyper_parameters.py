from typing import Self

from .drift_monitor_hyper_parameters import DriftMonitorHyperParameters


class DriftStatsHyperParameters:
    def __init__(
        self: Self,
        *,
        hyper_parameters: DriftMonitorHyperParameters,
        half_life_factor: float,
    ):
        self._hyper_parameters = hyper_parameters
        self._half_life_factor = half_life_factor

    @property
    def drift_half_life(self: Self) -> float:
        return self._hyper_parameters.half_life * self._half_life_factor
