"""
Individual state statistics tracking.

This module provides `StateStats`, a data class that tracks support (weight)
for a single state observation. Support is accumulated over time and can be
decayed to weight recent observations more heavily than older ones.
"""

from typing import Self


class StateStats:
    """
    Tracks cumulative support (weight) for a single state.

    Support is accumulated as observations are made and can be decayed
    over time to emphasize recent observations.

    Attributes:
        _support: Cumulative weight/support for this state. Defaults to 0.0.

    """

    _support: float = 0.0

    def __init__(self, support: float = 0.0) -> None:
        """
        Initializes a new instance of StateStats.

        Args:
          support: Initial support value for this state. Must be non-negative. Defaults to 0.0.
        """
        self._support = support

    def update(self: Self, weight: float = 1.0) -> None:
        """
        Adds weighted observation to state support.

        Args:
            weight: Weight to add. Must be non-negative. Defaults to 1.0.

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

        if self._support < 1e-12:
            self._support = 0.0

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
        """Returns a JSON-serializable representation of this StateStats."""
        return {"support": self._support}

    @classmethod
    def from_dict(cls, data: dict) -> "StateStats":
        """
        Reconstructs StateStats from a dict representation.

        Args:
            data: Dictionary containing state statistics.

        Returns:
            A new StateStats instance with support initialized from data.
        """
        return cls(support=data.get("support", 0.0))
