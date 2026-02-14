"""
Hyper-parameters for drift statistics tracking.

This module provides configuration for DriftStats, controlling the decay rate
used when computing exponentially weighted drift statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self


@dataclass()
class DriftStatsRuntimeParameters:
    """Runtime parameters for drift statistics decay behavior."""

    half_life_factor: Final[float]
    """Multiplier applied to base half-life for drift statistics decay."""

    def to_dict(self: Self) -> dict[str, float]:
        """
        Serialize runtime parameters to a dictionary.

        Returns:
            A dictionary containing the half-life factor for drift statistics.
        """
        return {"half_life_factor": self.half_life_factor}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, float],
    ) -> DriftStatsRuntimeParameters:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing the half_life_factor value.

        Returns:
            A new DriftStatsRuntimeParameters instance.
        """
        return cls(
            half_life_factor=data["half_life_factor"],
        )
