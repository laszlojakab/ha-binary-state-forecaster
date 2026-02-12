"""
Hyper-parameters for online error tracking.

This module provides configuration for the OnlineErrorTracker, controlling how
prediction errors are tracked over time with exponential decay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )


class OnlineErrorTrackerHyperParameters:
    """
    Configuration for online prediction error tracking.

    Wraps base HyperParameters and applies a factor to determine the half-life
    for error statistics. The error half-life controls how quickly old prediction
    errors are forgotten in favor of recent errors.

    Attributes:
        _hyper_parameters: Base hyper-parameters containing the reference half-life.
        _half_life_factor: Multiplier applied to base half-life for error tracking.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.hyper_parameters import (  # noqa: E501
        ...     HyperParameters,
        ... )
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> hp = OnlineErrorTrackerHyperParameters(
        ...     hyper_parameters=base_hp,
        ...     error_half_life_factor=0.5,
        ... )
        >>> hp.error_half_life
        25.0

    """

    def __init__(
        self: Self, *, hyper_parameters: HyperParameters, error_half_life_factor: float
    ) -> None:
        """
        Initialize error tracker hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters providing reference half-life.
            error_half_life_factor: Factor to multiply base half-life by for
                error tracking. Values < 1.0 make error tracking more responsive
                to recent errors, values > 1.0 make it more stable over time.

        """
        self._hyper_parameters: Final = hyper_parameters
        self._half_life_factor: Final = error_half_life_factor

    @property
    def error_half_life(self: Self) -> float:
        """
        Gets the half-life for error statistics.

        Returns:
            The half-life (in the same units as base half-life) for tracking
                prediction errors. After this time, the weight of old errors
                decays to 50% of their original value.

        """
        return self._hyper_parameters.half_life * self._half_life_factor

    def to_dict(self: Self) -> dict[str, Any]:
        return {"half_life_factor": self._half_life_factor}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> OnlineErrorTrackerHyperParameters:
        return cls(
            hyper_parameters=hyper_parameters,
            error_half_life_factor=data["half_life_factor"],
        )
