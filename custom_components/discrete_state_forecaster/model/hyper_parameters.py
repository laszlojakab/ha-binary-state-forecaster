from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self


@dataclass()
class HyperParameters:
    half_life: float = 3600.0
    min_prune_interval_factor: float = 5.0
    prune_enabled: bool = True
    persistence_strength: float = 0.5

    @property
    def min_prune_interval(self: Self) -> float:
        return self.half_life * self.min_prune_interval_factor

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "half_life": self.half_life,
            "min_prune_interval_factor": self.min_prune_interval_factor,
            "prune_enabled": self.prune_enabled,
            "persistence_strength": self.persistence_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperParameters:
        parameters = cls(
            half_life=data["half_life"],
            min_prune_interval_factor=data["min_prune_interval_factor"],
            prune_enabled=data["prune_enabled"],
            persistence_strength=data["persistence_strength"],
        )

        return parameters
