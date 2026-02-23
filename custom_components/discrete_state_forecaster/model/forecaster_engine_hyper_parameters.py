"""
Hyper-parameters for the forecaster engine.

This module defines the ForecasterEngineHyperParameters class, which manages
the configurable hyper-parameters used by the forecasting engine, including
half-life, pruning intervals, and persistence strength.
"""

from __future__ import annotations

from typing import Any, Self


class ForecasterEngineHyperParameters:
    """
    Manages hyper-parameters for forecaster engine.

    This class stores and manages configurable hyper-parameters that control
    the behavior of the forecasting engine. It supports dynamic updates while
    preserving baseline values for reset functionality.

    Attributes:
        _half_life: Current half-life for exponential decay (in seconds).
        _min_prune_interval_factor: Current minimum pruning interval factor.
        _prune_enabled: Whether pruning is currently enabled.
        _persistence_strength: Current persistence modeling strength.

    Example:
        >>> hp = ForecasterEngineHyperParameters(
        ...     half_life=300.0,
        ...     min_prune_interval_factor=5.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> hp.half_life
        300.0
        >>> hp.update(half_life=400.0)
        >>> hp.half_life
        400.0
        >>> hp.reset()
        >>> hp.half_life
        300.0

    """

    def __init__(
        self: Self,
        half_life: float,
        min_prune_interval_factor: float,
        prune_enabled: bool,
        persistence_strength: float,
    ) -> None:
        """
        Initialize forecaster engine hyper-parameters.

        Args:
            half_life: Half-life for exponential decay (in seconds).
            min_prune_interval_factor: Minimum interval factor for pruning.
            prune_enabled: Whether pruning is enabled.
            persistence_strength: Strength of persistence modeling (0.0 to 1.0).

        """
        self._half_life: float = half_life
        self._min_prune_interval_factor: float = min_prune_interval_factor
        self._prune_enabled: bool = prune_enabled
        self._persistence_strength: float = persistence_strength

    def update(
        self: Self,
        *,
        half_life: float | None = None,
        min_prune_interval_factor: float | None = None,
        prune_enabled: bool | None = None,
        persistence_strength: float | None = None,
    ) -> None:
        """
        Update hyper-parameters with new values.

        Only provided (non-None) parameters are updated. This allows
        selective updating of individual parameters without affecting others.

        Args:
            half_life: New half-life value (if provided).
            min_prune_interval_factor: New pruning interval factor (if provided).
            prune_enabled: New pruning enabled state (if provided).
            persistence_strength: New persistence strength (if provided).

        """
        if half_life is not None:
            self._half_life = half_life
        if min_prune_interval_factor is not None:
            self._min_prune_interval_factor = min_prune_interval_factor
        if prune_enabled is not None:
            self._prune_enabled = prune_enabled
        if persistence_strength is not None:
            self._persistence_strength = persistence_strength

    @property
    def half_life(self: Self) -> float:
        """
        Get current half-life value.

        Returns:
            Half-life for exponential decay (in seconds).

        """
        return self._half_life

    @property
    def prune_enabled(self: Self) -> bool:
        """
        Get current pruning enabled state.

        Returns:
            True if pruning is enabled, False otherwise.

        """
        return self._prune_enabled

    @property
    def persistence_strength(self: Self) -> float:
        """
        Get current persistence strength.

        Returns:
            Persistence modeling strength (0.0 to 1.0).

        """
        return self._persistence_strength

    @property
    def min_prune_interval_factor(self: Self) -> float:
        """
        Get current minimum pruning interval factor.

        Returns:
            Minimum interval factor for pruning operations.

        """
        return self._min_prune_interval_factor

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize hyper-parameters to a dictionary.

        Returns:
            Dictionary containing all current and baseline parameter values.

        """
        return {
            "half_life": self._half_life,
            "min_prune_interval_factor": self._min_prune_interval_factor,
            "prune_enabled": self._prune_enabled,
            "persistence_strength": self._persistence_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecasterEngineHyperParameters:
        """
        Create instance from a dictionary.

        Args:
            data: Dictionary containing parameter values.

        Returns:
            A new ForecasterEngineHyperParameters instance.

        """
        return cls(
            half_life=data["half_life"],
            min_prune_interval_factor=data["min_prune_interval_factor"],
            prune_enabled=data["prune_enabled"],
            persistence_strength=data["persistence_strength"],
        )
