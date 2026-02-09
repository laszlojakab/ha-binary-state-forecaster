from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class OnlineErrorTrackerHyperParameters:
    def __init__(
        self: Self, *, hyper_parameters: HyperParameters, error_half_life_factor: float
    ) -> None:
        self._hyper_parameters: Final = hyper_parameters
        self._half_life_factor: Final = error_half_life_factor

    @property
    def error_half_life(self: Self) -> float:
        return self._hyper_parameters.half_life * self._half_life_factor
