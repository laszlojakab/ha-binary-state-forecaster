"""Configuration parameters for hierarchical state statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass()
class HierarchicalStateStatsRuntimeParameters:
    """Runtime parameters for hierarchical state statistics."""

    min_support_factor: Final[float]
    """Scaling factor for minimum support threshold calculation."""
