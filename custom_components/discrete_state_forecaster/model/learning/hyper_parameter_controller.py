"""
Adaptive hyper-parameter control based on drift and error signals.

This module provides HyperParameterController, which dynamically adjusts
model hyper-parameters (particularly half-life) in response to concept drift
and prediction error trends. This allows the model to adapt its learning rate
to changing data patterns.
"""

import math
from enum import Enum, auto
from typing import Any, Final, Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)

from .hyper_parameter_controller_runtime_parameters import (
    HyperParameterControllerRuntimeParameters,
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
        _runtime_parameters: Runtime parameters for hyper-parameter control.
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
        runtime_parameters: HyperParameterControllerRuntimeParameters,
    ) -> None:
        """
        Initialize hyper-parameter controller.

        Args:
            runtime_parameters: Runtime parameters for hyper-parameter control.
        """
        self._hyper_parameters: Final[ForecasterEngineHyperParameters] = (
            ForecasterEngineHyperParameters(
                half_life=runtime_parameters.base_half_life,  # TODO: ez nem fog atmenni...
                min_prune_interval_factor=runtime_parameters.min_prune_interval_factor,
                prune_enabled=True,
                persistence_strength=runtime_parameters.base_persistence_strength,
            )
        )
        self._runtime_parameters: Final[HyperParameterControllerRuntimeParameters] = (
            runtime_parameters
        )

        self._min_half_life: float = self._runtime_parameters.min_half_life
        self._max_half_life: float = self._runtime_parameters.max_half_life

        self._mode: AdaptationMode = AdaptationMode.STABLE

    @property
    def hyper_parameters(self: Self) -> ForecasterEngineHyperParameters:
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
        entropy_confidence: float | None,
        fallback_depth: int | None = None,
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
            entropy_confidence: Confidence level from entropy (or None if unavailable).
            fallback_depth: Depth of fallback used (if applicable, None if not applicable).

        """
        error_worsening = (
            short_term_error is not None
            and long_term_error is not None
            and short_term_error > long_term_error * 1.1
        )

        # Consider low confidence as a signal of potential degradation,
        # even without explicit error increase
        low_confidence = (
            entropy_confidence is not None and entropy_confidence < 0.4  # noqa: PLR2004
        )

        # Consider deep fallback as a signal of potential drift impact,
        # even without explicit drift detection
        deep_fallback = (
            fallback_depth is not None and fallback_depth > 2  # noqa: PLR2004
        )

        # Determine adaptation mode based on signals
        if is_drifting and error_worsening:
            self._mode = AdaptationMode.CONCEPT_DRIFT
        elif is_drifting or deep_fallback:
            self._mode = AdaptationMode.DRIFTING_OK
        elif error_worsening or low_confidence:
            self._mode = AdaptationMode.MODEL_DEGRADING
        else:
            self._mode = AdaptationMode.STABLE

        # Update hyper-parameters based on mode and half-life
        self._update_params()

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize controller state to a dictionary.

        Returns:
            A dictionary containing current mode and hyper-parameter values.
        """
        return {
            "mode": self._mode.name,
            "hyper_parameters": self._hyper_parameters.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        runtime_parameters: HyperParameterControllerRuntimeParameters,
    ) -> Self:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing mode and hyper-parameter values.
            runtime_parameters: Runtime parameters for the controller.

        Returns:
            A new HyperParameterController instance initialized with the provided data.
        """
        instance = cls(runtime_parameters=runtime_parameters)

        instance._mode = AdaptationMode[data["mode"]]
        instance._hyper_parameters = ForecasterEngineHyperParameters.from_dict(
            data["hyper_parameters"]
        )

        return instance

    def _update_params(self: Self) -> None:
        """
        Update hyper-parameters based on current mode and half-life.

        Resets parameters and recomputes them based on current half-life
        and adaptation mode. Adjusts persistence strength, pruning policy,
        and pruning interval according to the mode.
        """
        if self._runtime_parameters.adaptation_config.adapt_half_life:
            if self._mode == AdaptationMode.CONCEPT_DRIFT:
                # If there is a concept drift, we want to adapt very quickly (short half-life) to
                # forget old patterns and learn new ones.
                log_change = -0.08
            elif self._mode == AdaptationMode.MODEL_DEGRADING:
                # If the model is degrading but we don't detect drift,
                # we want to adapt moderately (medium half-life)
                log_change = -0.03
            elif self._mode == AdaptationMode.DRIFTING_OK:
                # If the model is drifting but still performing okay,
                # we want to adapt slowly (long half-life)
                log_change = -0.015
            else:
                # If the model is stable, we want to maintain a long half-life
                # so we increase the half-life slightly to reinforce memory of stable patterns.
                log_change = +0.01

            half_life = math.exp(
                max(
                    math.log(self._min_half_life),
                    min(
                        math.log(self._max_half_life),
                        math.log(self._runtime_parameters.base_half_life) + log_change,
                    ),
                )
            )
        else:
            # We reset to baseline half-life
            half_life = self._runtime_parameters.base_half_life

        # ---- persistence derived from half-life ----
        ratio = (half_life - self._min_half_life) / (
            self._max_half_life - self._min_half_life
        )
        ratio = max(0.0, min(1.0, ratio))

        if self._runtime_parameters.adaptation_config.adapt_persistence:
            persistence_strength = 0.2 + 0.8 * ratio
        else:
            persistence_strength = self._hyper_parameters.persistence_strength

        if self._runtime_parameters.adaptation_config.adapt_prune_interval:
            base = self._runtime_parameters.min_prune_interval_factor

            if self._mode == AdaptationMode.CONCEPT_DRIFT:
                min_prune_interval_factor = base * 4
                prune_enabled = False

            elif self._mode == AdaptationMode.DRIFTING_OK:
                min_prune_interval_factor = base * 2
                prune_enabled = False

            elif self._mode == AdaptationMode.MODEL_DEGRADING:
                min_prune_interval_factor = base * 0.7
                prune_enabled = True

            else:
                min_prune_interval_factor = base
                prune_enabled = True
        else:
            min_prune_interval_factor = self._hyper_parameters.min_prune_interval_factor
            prune_enabled = self._hyper_parameters.prune_enabled

        self._hyper_parameters.update(
            half_life=half_life,
            min_prune_interval_factor=min_prune_interval_factor,
            prune_enabled=prune_enabled,
            persistence_strength=persistence_strength,
        )
