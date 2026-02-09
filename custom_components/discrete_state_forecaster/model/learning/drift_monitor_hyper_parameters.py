from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class DriftMonitorHyperParameters:
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
        return self._hyper_parameters.half_life

    @property
    def slow_half_life_factor(self: Self) -> float:
        return self._slow_half_life_factor

    @property
    def slow_prune_threshold(self: Self) -> float:
        return self._slow_prune_threshold

    @property
    def slow_epsilon(self: Self) -> float:
        return self._slow_epsilon

    @property
    def fast_half_life_factor(self: Self) -> float:
        return self._fast_half_life_factor

    @property
    def fast_prune_threshold(self: Self) -> float:
        return self._fast_prune_threshold

    @property
    def fast_epsilon(self: Self) -> float:
        return self._fast_epsilon

    @property
    def drift_half_life_factor(self: Self) -> float:
        return self._drift_half_life_factor

    @property
    def tau_enter(self: Self) -> float:
        return self._tau_enter

    @property
    def tau_exit(self: Self) -> float:
        return self._tau_exit

    @property
    def adaptive_tau(self: Self) -> float:
        return self._adaptive_tau

    @property
    def n_enter(self: Self) -> float:
        return self._n_enter

    @property
    def n_exit(self: Self) -> float:
        return self._n_exit
