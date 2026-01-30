"""
Composite indexer for multi-dimensional temporal pattern analysis.

This module implements a time indexer that combines multiple indexing strategies
to create hierarchical or multi-dimensional bucket keys. This enables modeling
complex patterns that depend on multiple time dimensions (e.g., both time of day
AND day of week, such as "Monday mornings" or "weekend evenings").
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Self

from custom_components.discrete_state_forecaster.model.time_indexers.time_indexer import (
    TimeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import TimeKey


class CompositeIndexer(TimeIndexer):
    """
    Combines multiple time indexers into a composite multi-dimensional key.

    A composite indexer aggregates several time indexing strategies to produce
    keys that capture multiple temporal dimensions simultaneously. This allows
    for fine-grained pattern modeling where behavior depends on the combination
    of multiple time factors.

    The composite key is a tuple of (name, value) pairs, one for each constituent
    indexer. This preserves both the identity and value of each dimension.

    Attributes:
        indexers: List of constituent TimeIndexer instances that define each
            dimension of the composite key.

    Example:
        >>> from time_of_day_indexer import TimeOfDayIndexer
        >>> from day_of_week_indexer import DayOfWeekIndexer
        >>>
        >>> composite = CompositeIndexer([
        ...     DayOfWeekIndexer(),
        ...     TimeOfDayIndexer(bucket_minutes=60)
        ... ])
        >>>
        >>> # Monday at 14:30
        >>> ts = datetime(2026, 1, 26, 14, 30)
        >>> composite.key(ts)
        (('weekday', 0), ('time_bucket', 14))
        >>>
        >>> # Next boundary is the sooner of next day or next hour
        >>> composite.next_boundary(ts)
        datetime(2026, 1, 26, 15, 0)  # Next hour comes first
    """

    def __init__(self: Self, indexers: Iterable[TimeIndexer]):
        """
        Initializes the composite indexer with multiple indexing strategies.

        Args:
            indexers: An iterable of TimeIndexer instances. Each indexer defines
                one dimension of the composite key. The order of indexers is
                preserved in the resulting composite keys.

        Example:
            >>> composite = CompositeIndexer([
            ...     DayOfWeekIndexer(),
            ...     TimeOfDayIndexer(bucket_minutes=30)
            ... ])
        """
        self.indexers = list(indexers)

    def key(self: Self, ts: datetime) -> TimeKey:
        """
        Generates a composite key from all constituent indexers.

        Creates a tuple of (name, value) pairs by calling key() on each
        constituent indexer. The resulting tuple captures all temporal
        dimensions simultaneously.

        Args:
            ts: The timestamp to map to a composite bucket.

        Returns:
            A tuple of (indexer_name, bucket_key) pairs, one for each
            constituent indexer. For example: (('weekday', 0), ('time_bucket', 28))
            represents Monday (0) in the 28th half-hour bucket (14:00-14:29).

        Example:
            >>> composite = CompositeIndexer([
            ...     DayOfWeekIndexer(),
            ...     TimeOfDayIndexer(bucket_minutes=60)
            ... ])
            >>> composite.key(datetime(2026, 1, 26, 14, 30))
            (('weekday', 0), ('time_bucket', 14))
            >>> composite.key(datetime(2026, 1, 31, 14, 30))  # Saturday
            (('weekday', 5), ('time_bucket', 14))
        """
        return tuple((idx.name, idx.key(ts)) for idx in self.indexers)

    def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Returns the earliest next boundary across all constituent indexers.

        Computes the next boundary for each constituent indexer and returns
        the minimum (earliest) one. This represents the soonest time when
        ANY dimension of the composite key will change.

        Args:
            ts: The reference timestamp.

        Returns:
            The earliest timestamp at which any constituent indexer's key
            will change. This is the minimum of all indexers' next_boundary()
            results.

        Example:
            >>> composite = CompositeIndexer([
            ...     DayOfWeekIndexer(),           # Changes at midnight
            ...     TimeOfDayIndexer(bucket_minutes=60)  # Changes every hour
            ... ])
            >>> composite.next_boundary(datetime(2026, 1, 26, 14, 30))
            datetime(2026, 1, 26, 15, 0)  # Next hour (sooner than midnight)
            >>> composite.next_boundary(datetime(2026, 1, 26, 23, 30))
            datetime(2026, 1, 27, 0, 0)  # Midnight (both boundaries coincide)
        """
        return min(idx.next_boundary(ts) for idx in self.indexers)
