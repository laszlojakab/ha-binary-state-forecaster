"""
Time-of-day based indexer for temporal pattern analysis.

This module implements a time indexer that partitions each day into fixed-duration
buckets based on time of day, ignoring the calendar date. This enables modeling
patterns that repeat daily (e.g., morning routines, evening activities).
"""

from datetime import datetime, timedelta
from typing import Self

from custom_components.discrete_state_forecaster.model.time_indexers.time_indexer import (
    TimeIndexer,
)


class TimeOfDayIndexer(TimeIndexer):
    """
    Indexes timestamps by time of day using fixed-duration buckets.

    Divides each 24-hour day into equal-sized time buckets, producing the same
    bucket keys for equivalent times on different dates. This is useful for
    modeling daily recurring patterns where the specific date is irrelevant
    but the time of day matters.

    The bucket size is configurable, allowing various granularities:
    - 60 minutes: hourly buckets (0-23)
    - 30 minutes: half-hourly buckets (0-47)
    - 15 minutes: quarter-hourly buckets (0-95)
    - etc.

    Attributes:
        bucket: The bucket duration in minutes.
        name: Always "time_bucket" for this indexer type.

    Example:
        ```
        >>> indexer = TimeOfDayIndexer(bucket_minutes=30)
        >>> indexer.key(datetime(2026, 1, 29, 14, 15))
        28
        >>> indexer.key(datetime(2025, 12, 25, 14, 15))
        28  # Same key despite different dates
        >>> indexer.next_boundary(datetime(2026, 1, 29, 14, 15))
        datetime(2026, 1, 29, 14, 30)
        ```
    """

    def __init__(self: Self, bucket_minutes: int):
        """
        Initialize the time-of-day indexer.

        Args:
            bucket_minutes: The duration of each bucket in minutes. Must be
                a positive integer. Common values are 15, 30, or 60 minutes.
                The value should ideally divide evenly into 1440 (minutes per day)
                for consistent bucket sizes, though this is not enforced.
        """
        self.bucket = bucket_minutes
        self.name = "time_bucket"

    def key(self: Self, ts: datetime) -> int:
        """
        Calculate the bucket key for a timestamp based on time of day.

        Converts the timestamp's time of day to minutes since midnight,
        then divides by the bucket size to determine the bucket index.
        The date component is ignored.

        Args:
            ts: The timestamp to map to a bucket.

        Returns:
            An integer bucket key (0-based). With 30-minute buckets, keys
            range from 0 (00:00-00:29) to 47 (23:30-23:59). The maximum
            possible key is (1439 // bucket_minutes).

        Example:
            With 30-minute buckets:
            - 00:00 -> 0
            - 00:29 -> 0
            - 00:30 -> 1
            - 14:15 -> 28
            - 23:59 -> 47
        """
        return (ts.hour * 60 + ts.minute) // self.bucket

    def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Calculate the next bucket boundary timestamp.

        Determines when the current time bucket ends and the next one begins.
        Handles wrapping to the next calendar day when crossing midnight.
        Always returns a timestamp with zero seconds and microseconds.

        Args:
            ts: The reference timestamp.

        Returns:
            The timestamp of the next bucket boundary. If the next boundary
            would be at or after midnight, returns a timestamp on the next
            calendar day.

        Example:
            With 30-minute buckets:
            - next_boundary(14:15:30) -> 14:30:00 (same day)
            - next_boundary(14:30:00) -> 15:00:00 (same day)
            - next_boundary(23:45:00) -> 00:00:00 (next day)
            - next_boundary(23:59:59) -> 00:00:00 (next day)
        """
        minutes = ts.hour * 60 + ts.minute
        next_bucket = ((minutes // self.bucket) + 1) * self.bucket

        delta = next_bucket - minutes
        return (ts + timedelta(minutes=delta)).replace(second=0, microsecond=0)
