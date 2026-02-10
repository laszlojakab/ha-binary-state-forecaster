"""
Adaptive hyper-parameter control based on drift and error signals.

This module provides HyperParameterController, which dynamically adjusts
model hyper-parameters (particularly half-life) in response to concept drift
and prediction error trends. This allows the model to adapt its learning rate
to changing data patterns.
"""
import math
from enum import Enum, auto
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class AdaptationMode(Enum):
    """
    Operating mode for hyper-parameter adaptation.

    Attributes:
        STABLE: No drift, no error increase - increase memory.
        DRIFTING_OK: Drift detected but errors stable - minor adaptation.
        MODEL_DEGRADING: No drift but errors increasing - moderate adaptation.
        CONCEPT_DRIFT: Both drift and errors increasing - aggressive adaptation.

    """

    STABLE = auto()
    DRIFTING_OK = auto()
    MODEL_DEGRADING = auto()
    CONCEPT_DRIFT = auto()


class HyperParameterController:
    """
    Controls adaptive hyper-parameter adjustment based on model performance.

    Monitors drift and error signals to determine operating mode and adjusts
    half-life accordingly. Shorter half-life means faster adaptation (less memory),
    longer half-life means more stability (more memory).

    Attributes:
        _hyper_parameters: Configuration object to update.
        _log_half_life: Log of current half-life (for smooth exponential updates).
        _min_half_life: Lower bound for half-life.
        _max_half_life: Upper bound for half-life.
        _mode: Current adaptation mode.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        ...     HyperParameters,
        ... )
        >>> hp = HyperParameters(
        ...     half_life=300.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> controller = HyperParameterController(
        ...     hyper_parameters=hp,
        ...     base_half_life=300.0,
        ... )
        >>> controller.update(is_drifting=False, short_term_error=None, long_term_error=None)
        >>> controller.mode == AdaptationMode.STABLE
        True

    """

    def __init__(
        self: Self,
        *,
        hyper_parameters: HyperParameters,
        base_half_life: float,
        min_half_life: float = 60.0,
        max_half_life: float = 3600.0 * 48,
    ) -> None:
        """
        Initialize hyper-parameter controller.

        Args:
            hyper_parameters: Configuration object to update.
            base_half_life: Initial half-life value.
            min_half_life: Lower bound for half-life (default 60 seconds).
            max_half_life: Upper bound for half-life (default 48 hours).

        """
        self._hyper_parameters: Final = hyper_parameters

        self._log_half_life = math.log(base_half_life)

        self._min_half_life = min_half_life
        self._max_half_life = max_half_life

        self._mode = AdaptationMode.STABLE

    @property
    def hyper_parameters(self: Self) -> HyperParameters:
        """
        Get the managed hyper-parameters object.

        Returns:
            The HyperParameters instance being controlled.

        """
        return self._hyper_parameters

    @property
    def mode(self: Self) -> AdaptationMode:
        """
        Get the current adaptation mode.

        Returns:
            Current operating mode determining adaptation strategy.

        """
        return self._mode

    def update(
        self: Self,
        *,
        is_drifting: bool,
        short_term_error: float | None,
        long_term_error: float | None,
    ) -> None:
        """
        Update adaptation mode and adjust hyper-parameters.

        Analyzes drift and error signals to determine operating mode, then
        adjusts half-life and other parameters accordingly. Call this method
        periodically (e.g., after each prediction) to enable adaptation.

        Args:
            is_drifting: Whether concept drift is currently detected.
            short_term_error: Recent prediction error (or None if unavailable).
            long_term_error: Historical prediction error (or None if unavailable).

        """
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

        self._update_params()
    def _update_params(self: Self) -> None:
        """
        Update hyper-parameters based on current mode and half-life.

        Resets parameters and recomputes them based on current log_half_life
        and adaptation mode. Adjusts persistence strength, pruning policy,
        and pruning interval according to the mode.

        """
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
