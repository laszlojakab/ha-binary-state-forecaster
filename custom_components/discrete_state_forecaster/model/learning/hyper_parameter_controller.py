import math
from enum import Enum, auto
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class AdaptationMode(Enum):
    STABLE = auto()
    DRIFTING_OK = auto()
    MODEL_DEGRADING = auto()
    CONCEPT_DRIFT = auto()


class HyperParameterController:
    def __init__(
        self: Self,
        *,
        hyper_parameters: HyperParameters,
        base_half_life: float,
        min_half_life: float = 60.0,
        max_half_life: float = 3600.0 * 48,
    ):
        self._hyper_parameters: Final = hyper_parameters

        self._log_half_life = math.log(base_half_life)

        self._min_half_life = min_half_life
        self._max_half_life = max_half_life

        self._mode = AdaptationMode.STABLE

    @property
    def hyper_parameters(self: Self) -> HyperParameters:
        return self._hyper_parameters

    @property
    def mode(self: Self) -> AdaptationMode:
        return self._mode

    def update(
        self: Self,
        *,
        is_drifting: bool,
        short_term_error: float | None,
        long_term_error: float | None,
    ) -> None:
        error_worsening = (
            short_term_error is not None
            and long_term_error is not None
            and short_term_error > long_term_error * 1.1
        )

        # ----- Mode decision -----

        if is_drifting and error_worsening:
            self._mode = AdaptationMode.CONCEPT_DRIFT
        elif is_drifting:
            self._mode = AdaptationMode.DRIFTING_OK
        elif error_worsening:
            self._mode = AdaptationMode.MODEL_DEGRADING
        else:
            self._mode = AdaptationMode.STABLE

        # ----- Half-life adaptation (continuous) -----

        if self._mode == AdaptationMode.CONCEPT_DRIFT:
            self._log_half_life -= 0.08

        elif self._mode == AdaptationMode.MODEL_DEGRADING:
            self._log_half_life -= 0.03

        elif self._mode == AdaptationMode.DRIFTING_OK:
            self._log_half_life -= 0.015

        else:  # STABLE
            self._log_half_life += 0.01

        # Clamp
        self._log_half_life = max(
            math.log(self._min_half_life),
            min(math.log(self._max_half_life), self._log_half_life),
        )

    def _update_params(self: Self) -> None:
        self._hyper_parameters.reset()

        half_life = math.exp(self._log_half_life)

        # ---- persistence derived from half-life ----
        ratio = (half_life - self._min_half_life) / (
            self._max_half_life - self._min_half_life
        )
        ratio = max(0.0, min(1.0, ratio))

        persistence_strength = 0.2 + 0.8 * ratio

        # ---- pruning policy by mode ----
        if self._mode == AdaptationMode.CONCEPT_DRIFT:
            prune_enabled = False
            prune_interval = self._hyper_parameters.min_prune_interval * 4

        elif self._mode == AdaptationMode.DRIFTING_OK:
            prune_enabled = False
            prune_interval = self._hyper_parameters.min_prune_interval * 2

        elif self._mode == AdaptationMode.MODEL_DEGRADING:
            prune_enabled = True
            prune_interval = self._hyper_parameters.min_prune_interval * 0.7

        else:  # STABLE
            prune_enabled = True
            prune_interval = self._hyper_parameters.min_prune_interval

        self._hyper_parameters.update(
            half_life=half_life,
            min_prune_interval=prune_interval,
            prune_enabled=prune_enabled,
            persistence_strength=persistence_strength,
        )
