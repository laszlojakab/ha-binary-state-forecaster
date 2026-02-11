"""
Immutable prediction result from hierarchical state statistics.

This module provides `PredictionResult`, a dataclass that encapsulates the
complete result of a prediction, including the predicted distribution and
information about which temporal levels contributed to the prediction.
"""

from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

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

    Example:
        >>> from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (  # noqa: E501
        ...     DistributionStats,
        ... )
        >>> from custom_components.discrete_state_forecaster.model.temporal.time_key import (
        ...     TimeKey,
        ... )
        >>> from custom_components.discrete_state_forecaster.model.statistics.contribution import (
        ...     Contribution,
        ... )
        >>> dist = DistributionStats()
        >>> dist.update("on", 2.0)
        >>> dist.update("off", 1.0)
        >>> key = TimeKey.from_tuple((("hour", 14),))
        >>> contrib = Contribution(key, weight=1.0, support=3.0)
        >>> result = PredictionResult(key=key, distribution=dist, contributions=(contrib,))
        >>> result.key == key
        True
        >>> result.distribution.max_probability()  # doctest: +SKIP
        0.6667

    """  # noqa: E501

    key: TimeKey
    distribution: DistributionStats
    contributions: tuple[Contribution, ...]
