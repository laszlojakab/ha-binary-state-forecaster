"""
Hyper-parameters for state persistence tracking.

This module provides configuration for StatePersistenceTracker, controlling
the decay rate used when computing exponentially weighted mean durations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )


class StatePersistenceTrackerHyperParameters:
    """
    Configuration for state persistence tracking behavior.

    Wraps HyperParameters and applies a factor to determine the half-life
    for persistence duration statistics. The persistence half-life controls
    how quickly old duration observations are forgotten.

    Example:
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> hp = StatePersistenceTrackerHyperParameters(
        ...     hyper_parameters=base_hp,
        ...     persistence_half_life_factor=2.0,
        ... )
        >>> hp.persistence_half_life
        100.0

    """

    _hyper_parameters: Final[HyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    _persistence_half_life_factor: Final[float]
    """Multiplier applied to base half-life for persistence tracking."""

    def __init__(
        self: Self,
        *,
        hyper_parameters: HyperParameters,
        persistence_half_life_factor: float,
    ) -> None:
        """
        Initialize state persistence tracker hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters providing reference half-life.
            persistence_half_life_factor: Multiplier for base half-life.
        """
        self._hyper_parameters: Final = hyper_parameters
        self._persistence_half_life_factor: Final = persistence_half_life_factor

    @property
    def persistence_half_life(self: Self) -> float:
        """
        Get the half-life for persistence duration statistics.

        Returns:
            The half-life (in same units as base) for duration statistics decay.

        """
        return self._hyper_parameters.half_life * self._persistence_half_life_factor

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the instance into a dictionary.

        Returns:
            A dictionary containing the persistence_half_life_factor.
        """
        return {
            "persistence_half_life_factor": self._persistence_half_life_factor,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> StatePersistenceTrackerHyperParameters:
        """
        Deserialize an instance from a dictionary.

        Args:
            data: Dictionary containing the serialized persistence_half_life_factor.
            hyper_parameters: Base hyper-parameters providing the reference half_life
                value used to compute the persistence half-life.

        Returns:
            A new StatePersistenceTrackerHyperParameters instance.
        """
        return cls(
            hyper_parameters=hyper_parameters,
            persistence_half_life_factor=data["persistence_half_life_factor"],
        )
