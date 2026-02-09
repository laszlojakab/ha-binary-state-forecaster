from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class HierarchicalStateStatsHyperParameters:
    def __init__(
        self: Self, hyper_parameters: HyperParameters, min_support_factor: float = 1.0
    ):
        self._hyper_parameters: Final = hyper_parameters
        self._min_support_factor: Final = min_support_factor

    @property
    def min_support(self: Self) -> float:
        return self._hyper_parameters.half_life * self._min_support_factor
