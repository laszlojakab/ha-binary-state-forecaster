"""Immutable prediction result from hierarchical state statistics."""

import math
from dataclasses import dataclass
from typing import Self

from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .confidence import Confidence
from .contribution import Contribution


@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable result of hierarchical state prediction.

    Contains the predicted probability distribution and information about which
    temporal hierarchy levels contributed to the prediction. The contributions
    tuple shows how confident each level was and how much impact it had.

    Attributes:
        key: The TimeKey representing the temporal location of the prediction.
        distribution: The predicted probability distribution over states.
        contributions: Tuple of Contribution objects describing the temporal
            levels that were used to form this prediction, in order of usage
            (specific level first, then ancestors). Each contribution includes
            the source key, weight applied, and support available at that level.
    """

    key: TimeKey
    """TimeKey representing the temporal location of the prediction."""

    distribution: dict[State, float]
    """Predicted probability distribution over states at the given TimeKey."""

    confidence: Confidence
    """Confidence metrics for the prediction."""

    contributions: tuple[Contribution, ...]
    """Ordered tuple of contributions from temporal levels used in the prediction."""

    def __init__(
        self: Self,
        key: TimeKey,
        distribution_stats: DistributionStats,
        contributions: tuple[Contribution, ...],
    ):
        """
        Initializes a instance of PredictionResult class.

        Args:
            key: The TimeKey representing the temporal location of the prediction.
            distribution_stats: The DistributionStats object containing the
                predicted distribution and related statistics.
            contributions: Tuple of Contribution objects describing the temporal
                levels that were used to form this prediction, in order of usage
                (specific level first, then ancestors). Each contribution includes
                the source key, weight applied, and support available at that level.
        """
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "distribution", distribution_stats.distribution)

        n = len(distribution_stats.distribution)
        max_entropy = math.log(n) if n > 1 else 1.0
        entropy_conf = 1.0 - (distribution_stats.entropy / max_entropy)

        object.__setattr__(
            self,
            "confidence",
            Confidence(
                support=distribution_stats.total_support,
                depth=len(contributions) - 1,
                max_probability=distribution_stats.max_probability,
                entropy_confidence=entropy_conf,
            ),
        )

        object.__setattr__(self, "contributions", contributions)
