"""
Individual state statistics tracking.

This module provides `StateStats`, a data class that tracks support (weight)
for a single state observation. Support is accumulated over time and can be
decayed to weight recent observations more heavily than older ones.
"""

from __future__ import annotations

from typing import Self


class StateStats:
    """
    Tracks cumulative support (weight) for a single state.

    Support is accumulated as observations are made and can be decayed
    over time to emphasize recent observations.

    Example:
        >>> stats = StateStats()
        >>> stats.update(2.0)
        >>> stats.support
        2.0
        >>> stats.is_active(1.0)
        True
        >>> stats.apply_decay(0.5)
        >>> stats.support
        1.0
        >>> stats.is_active(1.0)
        True
        >>> stats.apply_decay(0.5)
        >>> stats.support
        0.5
        >>> stats.is_active(1.0)
        False
    """

    _support: float = 0.0
    """Cumulative support for this state. Represents the total weight of observations."""

    def __init__(self) -> None:
        """Initializes a new instance of `StateStats` class."""
        self._support = 0.0

    def update(self: Self, weight: float) -> None:
        """
        Adds weighted observation to state support.

        Args:
            weight: Weight to add.

        Raises:
            ValueError: If weight is negative.
        """
        if weight < 0:
            raise ValueError("weight must be non negative")

        self._support += weight

    def apply_decay(self: Self, factor: float) -> None:
        """
        Applies exponential decay to state support.

        Args:
            factor: Decay multiplier in range (0, 1].

        Raises:
            ValueError: If factor is not in range (0, 1].

        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"decay factor must be in (0, 1]. Got: {factor}")

        self._support *= factor

        if self._support < 1e-12:  # noqa: PLR2004
            self._support = 0.0

    @property
    def support(self: Self) -> float:
        """
        Gets the current support value.

        Returns:
            The accumulated weight/support for this state.

        """
        return self._support

    def is_active(self: Self, min_support: float) -> bool:
        """
        Checks if state support meets minimum threshold.

        Args:
            min_support: Minimum support threshold to check against.

        Returns:
            True if state's support >= min_support.

        """
        return self._support >= min_support

    def to_dict(self: Self) -> dict:
        """Returns a JSON-serializable representation of the instance."""
        return {"support": self._support}

    @classmethod
    def from_dict(cls, data: dict) -> StateStats:
        """
        Reconstructs `StateStats` from a `dict` representation.

        Args:
            data: Dictionary containing state statistics.

        Returns:
            A new `StateStats` instance initialized from data.
        """
        stats = cls()
        stats._support = data["support"]

        return stats
