"""
Time-of-day indexer implementation.

This module provides `TimeOfDayIndexer`, a TimeIndexer implementation that
buckets timestamps by the number of seconds elapsed since midnight (local time).
Each bucket represents a discrete time period within a day, enabling the
forecaster to learn time-of-day-specific patterns.

The bucket size is configurable, allowing for different time resolutions:
- 3600 seconds: 1-hour buckets (24 buckets per day)
- 300 seconds: 5-minute buckets (288 buckets per day)
- 60 seconds: 1-minute buckets (1440 buckets per day)
"""

from datetime import datetime
from typing import Final, Self

from .temporal_feature import TemporalFeature
from .time_indexer import (
    TimeIndexer,
)
from .time_key import TimeKey


class TimeOfDayIndexer(TimeIndexer):
    """
    Maps timestamps to time-of-day buckets based on seconds since midnight.

    This indexer divides each day into discrete time buckets of configurable
    size. For example, with 1-hour buckets, all times between 2:00 PM and
    2:59 PM map to the same bucket (index 14).

    The indexer converts the timestamp's hour, minute, and second into total
    seconds since midnight, then divides by the bucket size to get the bucket
    index. This enables learning of daily patterns that repeat across days.

    Attributes:
        name: Always set to "time_bucket" - the name of the temporal feature.
        bucket_size: The number of seconds in each bucket.

    Examples:
        >>> indexer = TimeOfDayIndexer(bucket_size=3600)  # 1-hour buckets
        >>> # 2:30 PM maps to bucket 14 (hour=14)
        >>> timestamp = datetime(2024, 1, 15, 14, 30)
        >>> key = await indexer.get_key(timestamp)
        >>> key.to_tuple()
        (('time_bucket', 14),)
        >>> # 2:00 AM also maps to bucket 2
        >>> early = datetime(2024, 1, 15, 2, 0)
        >>> key2 = await indexer.get_key(early)
        >>> key2.to_tuple()
        (('time_bucket', 2),)

    """

    name: Final = "time_bucket"

    def __init__(self: Self, bucket_size: int):
        """
        Initializes the time-of-day indexer.

        Args:
            bucket_size: The number of seconds per bucket. Common values:
                - 3600: 1-hour buckets (24 per day)
                - 300: 5-minute buckets
                - 60: 1-minute buckets

        Raises:
            ValueError: If bucket_size is not positive.

        Examples:
            >>> indexer = TimeOfDayIndexer(bucket_size=3600)
            >>> indexer.bucket_size
            3600

        """
        if bucket_size <= 0:
            msg = f"bucket_size must be positive, got {bucket_size}"
            raise ValueError(msg)
        self.bucket_size: Final = bucket_size

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Maps a timestamp to its time-of-day bucket.

        Calculates the number of seconds since midnight and divides by
        bucket_size to determine the bucket index.

        Args:
            timestamp: The datetime to map to a time-of-day bucket.

        Returns:
            A TimeKey with a single feature: ("time_bucket", bucket_index).

        Examples:
            >>> indexer = TimeOfDayIndexer(bucket_size=3600)
            >>> # 2:30 PM on any date -> bucket 14
            >>> ts = datetime(2024, 1, 15, 14, 30)
            >>> key = await indexer.get_key(ts)
            >>> key.to_tuple()
            (('time_bucket', 14),)
            >>> # Midnight is bucket 0
            >>> ts2 = datetime(2024, 1, 15, 0, 0)
            >>> key2 = await indexer.get_key(ts2)
            >>> key2.to_tuple()
            (('time_bucket', 0),)

        """
        total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        bucket_index = total_seconds // self.bucket_size

        return TimeKey.from_temporal_feature(TemporalFeature(name=self.name, value=bucket_index))
