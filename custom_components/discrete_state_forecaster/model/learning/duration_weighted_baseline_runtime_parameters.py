"""
Hyper-parameters for duration-weighted baseline distribution.

This module provides configuration for DurationWeightedBaseline, controlling
the decay rate, pruning threshold, and Laplace smoothing parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Self


@dataclass()
class DurationWeightedBaselineRuntimeParameters:
    """Runtime parameters for duration-weighted baseline behavior."""

    half_life_factor: Final[float]
    """Multiplier applied to base half-life for baseline decay."""

    prune_threshold: Final[float]
    """Minimum mass value below which states are pruned."""

    epsilon: Final[float]
    """Laplace smoothing parameter added to all state probabilities."""

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the instance into a dictionary.

        Returns:
            A dictionary containing half_life_factor, prune_threshold, and epsilon.
        """
        return {
            "half_life_factor": self.half_life_factor,
            "prune_threshold": self.prune_threshold,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> DurationWeightedBaselineRuntimeParameters:
        """
        Deserialize an instance from a dictionary.

        Args:
            data: Dictionary containing serialized half_life_factor, prune_threshold,
                and epsilon values.

        Returns:
            A new DurationWeightedBaselineRuntimeParameters instance.
        """
        return cls(
            half_life_factor=data["half_life_factor"],
            prune_threshold=data["prune_threshold"],
            epsilon=data["epsilon"],
        )
