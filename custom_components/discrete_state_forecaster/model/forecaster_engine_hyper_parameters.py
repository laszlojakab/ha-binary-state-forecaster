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
        _background_decay_half_life_factor: Multiplier for background (dormant-key)
            decay.  ``0.0`` disables background decay entirely (pure per-key
            observation-weighted decay).  A positive value ``f`` causes all
            keys to receive a slow background decay whose effective half-life
            is ``f * half_life``; e.g. ``20.0`` means dormant keys decay
            20× slower than actively observed keys.

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
        background_decay_half_life_factor: float = 0.0,
    ) -> None:
        """
        Initialize forecaster engine hyper-parameters.

        Args:
            half_life: Half-life for exponential decay (in seconds).
            min_prune_interval_factor: Minimum interval factor for pruning.
            prune_enabled: Whether pruning is enabled.
            persistence_strength: Strength of persistence modeling (0.0 to 1.0).
            background_decay_half_life_factor: Multiplier for background decay
                applied to all keys (including dormant ones) at each update.
                ``0.0`` (default) disables background decay.  A positive value
                ``f`` gives dormant keys a half-life of ``f * half_life``.

        """
        self._half_life: float = half_life
        self._min_prune_interval_factor: float = min_prune_interval_factor
        self._prune_enabled: bool = prune_enabled
        self._persistence_strength: float = persistence_strength
        self._background_decay_half_life_factor: float = background_decay_half_life_factor

    def update(
        self: Self,
        *,
        half_life: float | None = None,
        min_prune_interval_factor: float | None = None,
        prune_enabled: bool | None = None,
        persistence_strength: float | None = None,
        background_decay_half_life_factor: float | None = None,
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
            background_decay_half_life_factor: New background decay multiplier
                (if provided).  ``0.0`` disables background decay.

        """
        if half_life is not None:
            self._half_life = half_life
        if min_prune_interval_factor is not None:
            self._min_prune_interval_factor = min_prune_interval_factor
        if prune_enabled is not None:
            self._prune_enabled = prune_enabled
        if persistence_strength is not None:
            self._persistence_strength = persistence_strength
        if background_decay_half_life_factor is not None:
            self._background_decay_half_life_factor = background_decay_half_life_factor

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

    @property
    def background_decay_half_life_factor(self: Self) -> float:
        """
        Get background decay half-life multiplier.

        Returns:
            Multiplier applied to ``half_life`` to obtain the effective
            half-life for background (dormant-key) decay.  ``0.0`` means
            background decay is disabled.

        """
        return self._background_decay_half_life_factor

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
            "background_decay_half_life_factor": self._background_decay_half_life_factor,
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
            background_decay_half_life_factor=data.get(
                "background_decay_half_life_factor", 0.0
            ),
        )
