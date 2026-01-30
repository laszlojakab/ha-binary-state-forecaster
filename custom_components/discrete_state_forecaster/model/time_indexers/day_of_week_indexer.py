"""
Day-of-week based indexer for weekly pattern analysis.

This module implements a time indexer that partitions time by day of the week,
enabling modeling of patterns that repeat weekly (e.g., weekday vs weekend behavior,
Monday morning routines, Friday evening activities).
"""

from datetime import datetime, timedelta
from typing import Self

from custom_components.discrete_state_forecaster.model.time_indexers.time_indexer import (
    TimeIndexer,
)


class DayOfWeekIndexer(TimeIndexer):
    """
    Indexes timestamps by day of the week.

    Maps each timestamp to its day of the week (Monday through Sunday),
    producing consistent bucket keys for the same weekday regardless of
    the calendar date. This is useful for modeling weekly recurring patterns
    where behavior differs by day of week.

    The indexer uses Python's weekday convention:
    - Monday = 0
    - Tuesday = 1
    - Wednesday = 2
    - Thursday = 3
    - Friday = 4
    - Saturday = 5
    - Sunday = 6

    Attributes:
        name: Always "weekday" for this indexer type.

    Example:
        ```
        >>> indexer = DayOfWeekIndexer()
        >>> indexer.key(datetime(2026, 1, 26))  # Monday
        0
        >>> indexer.key(datetime(2026, 2, 2))   # Also Monday
        0
        >>> indexer.key(datetime(2026, 1, 31))  # Friday
        4
        >>> indexer.next_boundary(datetime(2026, 1, 26, 14, 30))
        datetime(2026, 1, 27, 0, 0)  # Next day at midnight
        ```
    """

    name = "weekday"

    def key(self: Self, ts: datetime) -> int:
        """
        Returns the day of week for a timestamp.

        Maps the timestamp to its weekday using Python's weekday() convention
        where Monday=0 and Sunday=6.

        Args:
            ts: The timestamp to map to a weekday.

        Returns:
            An integer from 0 to 6 representing the day of week:
            0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday,
            5=Saturday, 6=Sunday.

        Example:
            ```
            >>> indexer.key(datetime(2026, 1, 26))  # Monday
            0
            >>> indexer.key(datetime(2026, 1, 27))  # Tuesday
            1
            >>> indexer.key(datetime(2026, 2, 1))   # Sunday
            6
            ```
        """
        return ts.weekday()

    def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Returns midnight of the next day.

        Computes the timestamp when the day of week changes, which is
        always midnight (00:00:00) of the following calendar day.

        Args:
            ts: The reference timestamp.

        Returns:
            Midnight of the next calendar day, with time set to 00:00:00.000.

        Example:
            ```
            >>> indexer.next_boundary(datetime(2026, 1, 26, 14, 30))
            datetime(2026, 1, 27, 0, 0, 0, 0)
            >>> indexer.next_boundary(datetime(2026, 1, 26, 23, 59, 59))
            datetime(2026, 1, 27, 0, 0, 0, 0)
            ```
        """
        return ts.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
