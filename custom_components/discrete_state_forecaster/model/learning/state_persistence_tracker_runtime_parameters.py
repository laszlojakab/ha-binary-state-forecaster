"""Runtime parameters for state persistence tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Self


@dataclass
class StatePersistenceTrackerRuntimeParameters:
    """Runtime parameters for state persistence tracking behavior."""

    persistence_half_life_factor: Final[float]
    """Multiplier applied to base half-life for persistence tracking."""

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the instance into a dictionary.

        Returns:
            A dictionary containing the persistence_half_life_factor.
        """
        return {
            "persistence_half_life_factor": self.persistence_half_life_factor,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> StatePersistenceTrackerRuntimeParameters:
        """
        Deserialize an instance from a dictionary.

        Args:
            data: Dictionary containing the serialized persistence_half_life_factor.

        Returns:
            A new StatePersistenceTrackerRuntimeParameters instance.
        """
        return cls(
            persistence_half_life_factor=data["persistence_half_life_factor"],
        )
