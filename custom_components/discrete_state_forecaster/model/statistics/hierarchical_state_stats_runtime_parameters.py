"""Configuration parameters for hierarchical state statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self


@dataclass()
class HierarchicalStateStatsRuntimeParameters:
    """Runtime parameters for hierarchical state statistics."""

    min_support_factor: Final[float]
    """Scaling factor for minimum support threshold calculation."""

    def to_dict(self: Self) -> dict[str, float]:
        """Returns a JSON-serializable representation of this instance."""
        return {"min_support_factor": self.min_support_factor}

    @classmethod
    def from_dict(
        cls, data: dict[str, float]
    ) -> HierarchicalStateStatsRuntimeParameters:
        """
        Reconstructs `HierarchicalStateStatsRuntimeParameters` from a `dict` representation.

        Args:
            data: Dictionary containing runtime parameters.

        Returns:
            A new `HierarchicalStateStatsRuntimeParameters` instance initialized from data.
        """
        return cls(min_support_factor=data["min_support_factor"])
