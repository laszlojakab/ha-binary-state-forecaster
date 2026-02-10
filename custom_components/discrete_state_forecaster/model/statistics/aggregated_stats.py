"""Distribution statistics aggregated at specific temporal locations.

This module provides `AggregatedStats`, which extends `DistributionStats` by
associating it with a specific temporal context represented by a `TimeKey`.
This is useful for building hierarchical temporal models where statistics are
separately maintained for different times, time-of-day values, seasons, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from .distribution_stats import DistributionStats

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.temporal.time_key import (
        TimeKey,
    )


class AggregatedStats(DistributionStats):
    """Distribution statistics for a specific temporal context.

    Extends DistributionStats with an immutable TimeKey that identifies the
    temporal context for which these statistics are aggregated. Enables
    hierarchical models where different temporal contexts maintain independent
    state distributions.

    Attributes:
        key: Immutable TimeKey identifying the temporal location of this
            statistics aggregation (e.g., hour=14, day_of_week=Mon).
        _states: Inherited from DistributionStats. Dictionary mapping states
            to their individual statistics.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
        >>> from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import TemporalFeature
        >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        >>> stats = AggregatedStats(key)
        >>> stats.update("on", weight=1.0)
        >>> stats.key == key
        True
    """

    def __init__(self, key: TimeKey) -> None:
        """Initialize aggregated statistics for a specific temporal context.

        Args:
            key: The TimeKey identifying the temporal location for these
                statistics.
        """
        super().__init__()
        self.key: Final[TimeKey] = key

    @classmethod
    def from_distribution(
        cls,
        distribution: DistributionStats,
        key: TimeKey,
    ) -> AggregatedStats:
        """Create AggregatedStats from an existing DistributionStats.

        Converts support-based statistics into aggregated statistics by copying
        the normalized distribution. This is useful for extracting a snapshot
        of statistics from one context and associating it with a different
        temporal context.

        Args:
            distribution: The DistributionStats to copy from.
            key: The TimeKey to associate with the new aggregated statistics.

        Returns:
            A new AggregatedStats instance containing the states and their
                support values from distribution, associated with the given key.

        Example:
            >>> dist = DistributionStats()
            >>> dist.update("on", 2.0)
            >>> dist.update("off", 1.0)
            >>> from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
            >>> key = TimeKey.from_tuple((("hour", 14),))
            >>> agg = AggregatedStats.from_distribution(dist, key)
            >>> agg.key == key
            True
            >>> agg.states() == {"on", "off"}
            True
        """
        agg = cls(key)

        total_support = distribution.total_support()

        for state, prob in distribution.distribution().items():
            agg.update(state, prob * total_support)

        return agg
