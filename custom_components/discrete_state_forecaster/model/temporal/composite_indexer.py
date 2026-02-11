"""
Composite time indexer for hierarchical temporal contexts.

This module provides `CompositeIndexer`, which combines multiple TimeIndexer
instances into a single hierarchical indexer. It applies each indexer in
sequence, building up a hierarchical TimeKey that captures multiple temporal
dimensions simultaneously.

For example, combining TimeOfDayIndexer, DayOfWeekIndexer, and SeasonIndexer
creates a three-level hierarchy where patterns can be learned separately for
each combination of time-of-day, day-of-week, and season.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Final, Self

from .time_indexer import TimeIndexer
from .time_key import TimeKey


class CompositeIndexer(TimeIndexer):
    """
    Combines multiple time indexers into a single hierarchical indexer.

    A CompositeIndexer applies a sequence of TimeIndexer instances to a given
    timestamp, composing their individual TimeKeys into a single hierarchical
    key. The indexers are applied in order, with each indexer's result nested
    under the previous one.

    This enables modeling of multi-dimensional temporal patterns, where the
    forecaster can learn different behaviors for different combinations of
    temporal contexts (e.g., morning hours on weekends vs weekdays).

    Attributes:
        name: Derived from the names of all component indexers.
        indexers: The list of TimeIndexer instances to apply in sequence.

    Examples:
        >>> indexers = [
        ...     TimeOfDayIndexer(bucket_size=3600),  # 1-hour buckets
        ...     DayOfWeekIndexer(),                  # Day of week
        ...     SeasonIndexer(),                     # Season
        ... ]
        >>> composite = CompositeIndexer(indexers)
        >>> timestamp = datetime(2024, 1, 15, 14, 30)  # 2:30 PM Monday in winter
        >>> key = await composite.get_key(timestamp)
        >>> # key will have 3 features: hour, day_of_week, and season
        >>> len(key)
        3

    """

    def __init__(self: Self, indexers: Iterable[TimeIndexer]) -> None:
        """
        Initialize the composite indexer.

        Args:
            indexers: An iterable of TimeIndexer instances to apply in sequence.
                The indexers are stored in the order provided and applied in
                that order to construct hierarchical keys.

        Examples:
            >>> indexers = [
            ...     TimeOfDayIndexer(bucket_size=3600),
            ...     DayOfWeekIndexer(),
            ... ]
            >>> composite = CompositeIndexer(indexers)

        """
        self.indexers: Final = list(indexers)
        self.name: Final = ", ".join(idx.name for idx in self.indexers)

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Map a timestamp to a hierarchical temporal key.

        Applies each indexer in sequence, composing their results into a
        single hierarchical TimeKey. The first indexer's result becomes the
        root, and each subsequent indexer's features are nested under it.

        Args:
            timestamp: The datetime to map to a hierarchical temporal context.

        Returns:
            A TimeKey representing the hierarchical temporal context. The key
            will have len(indexers) features, one from each indexer.

        Examples:
            >>> composite = CompositeIndexer([
            ...     TimeOfDayIndexer(bucket_size=3600),
            ...     DayOfWeekIndexer(),
            ... ])
            >>> timestamp = datetime(2024, 1, 15, 14, 30)  # 2:30 PM Monday
            >>> key = await composite.get_key(timestamp)
            >>> key.to_tuple()
            (('time_bucket', 14), ('day_of_week', 0))

        """
        current: TimeKey = TimeKey.GLOBAL
        for indexer in self.indexers:
            current += await indexer.get_key(timestamp)

        return current
