"""Hyper-parameters for online error tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Self


@dataclass()
class OnlineErrorTrackerRuntimeParameters:
    """Runtime parameters for online prediction error tracking."""

    error_half_life_factor: Final[float]
    """
    Factor to multiply base half-life by for
    error tracking. Values < 1.0 make error tracking more responsive
    to recent errors, values > 1.0 make it more stable over time.
    """

    def to_dict(self: Self) -> dict[str, Any]:
        """Serializes the instance into a dictionary."""
        return {"error_half_life_factor": self.error_half_life_factor}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineErrorTrackerRuntimeParameters:
        """
        Deserializes a dictionary into an instance of OnlineErrorTrackerRuntimeParameters.

        Args:
            data: Dictionary containing serialized hyper-parameters.
        """
        return cls(
            error_half_life_factor=data["error_half_life_factor"],
        )
