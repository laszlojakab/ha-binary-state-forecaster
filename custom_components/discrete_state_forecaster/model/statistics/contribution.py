"""Contribution of a temporal level to a prediction.

This module provides `Contribution`, a dataclass that tracks how a specific
temporal level (ancestor in the hierarchy) contributed to a prediction,
including the weight applied and the data available at that level.
"""
from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)


@dataclass(frozen=True)
class Contribution:
    """Contribution of a single temporal level to a hierarchical prediction.

    Records information about how a specific temporal hierarchy level was used
    in forming a prediction, including the TimeKey, confidence weight applied,
    and amount of support data available at that level.

    Attributes:
        key: The TimeKey identifying the temporal level that contributed.
        weight: The confidence weight applied to this level's distribution.
            Weight depends on distance from the specific level: 1.0 for the
            specific level itself, 0.5 for immediate ancestor, 0.33 for
            grandparent, etc. Calculated as 1.0 / (1.0 + level_distance).
        support: The total support (sum of all state weights) available at
            this temporal level. Higher support indicates greater confidence
            in the distribution at this level.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
        >>> key = TimeKey.from_tuple((("hour", 14),))
        >>> contrib = Contribution(key=key, weight=1.0, support=50.0)
        >>> contrib.weight
        1.0
        >>> contrib.support
        50.0
    """
    key: TimeKey
    weight: float
    support: float
