from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class StatePersistenceTrackerHyperParameters:
    def __init__(
        self: Self,
        *,
        hyper_parameters: HyperParameters,
        persistence_half_life_factor: float,
    ) -> None:
        self._hyper_parameters: Final = hyper_parameters
        self._persistence_half_life_factor: Final = persistence_half_life_factor

    @property
    def persistence_half_life(self: Self) -> float:
        return self._hyper_parameters.half_life * self._persistence_half_life_factor
