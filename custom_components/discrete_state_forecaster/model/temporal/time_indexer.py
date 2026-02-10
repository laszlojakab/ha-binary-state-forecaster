"""
Protocol definition for time indexers.

This module defines `TimeIndexer`, a protocol (interface) that all time indexers
must implement. TimeIndexers convert timestamps into hierarchical temporal keys,
enabling the forecaster to learn and predict patterns within different temporal
contexts (e.g., time of day, day of week, seasons).

Implementations of TimeIndexer can be composed together using CompositeIndexer
to create multi-level temporal hierarchies.
"""

from datetime import datetime
from typing import Protocol, Self

from .time_key import TimeKey


class TimeIndexer(Protocol):
    """
    Protocol for converting timestamps into temporal keys.

    A TimeIndexer maps a given timestamp to a TimeKey, effectively bucketing
    or classifying the timestamp into a temporal context. Different indexers
    can implement different temporal bucketing strategies.

    For example:
    - TimeOfDayIndexer maps to hourly buckets
    - DayOfWeekIndexer maps to days of the week
    - SeasonIndexer maps to meteorological seasons
    - CalendarIndexer maps to calendar event boundaries

    Multiple indexers can be composed via CompositeIndexer to build hierarchical
    temporal structures where patterns are learned separately for each combination
    of temporal contexts.

    Attributes:
        name: A unique identifier for this indexer (used in TimeKey feature names).

    Examples:
        >>> # Typical usage in the forecaster
        >>> indexer = TimeOfDayIndexer(bucket_size=3600)  # 1-hour buckets
        >>> timestamp = datetime(2024, 1, 15, 14, 30)
        >>> key = await indexer.get_key(timestamp)
        >>> key.to_tuple()
        (('time_bucket', 14),)

    """

    name: str

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Maps a timestamp to a temporal key.

        Args:
            timestamp: The datetime to map to a temporal context.

        Returns:
            A TimeKey representing the temporal context of the given timestamp.
            The returned key should have exactly one feature (created using
            TimeKey.from_temporal_feature).

        Examples:
            >>> # For an hourly indexer at 2:30 PM
            >>> timestamp = datetime(2024, 1, 15, 14, 30)
            >>> key = await indexer.get_key(timestamp)
            >>> key.to_tuple()
            (('time_bucket', 14),)  # Hour 14

        """
        ...
