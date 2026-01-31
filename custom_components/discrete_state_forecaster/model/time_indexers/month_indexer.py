"""
Month-based indexer for seasonal and monthly pattern analysis.

This module implements a time indexer that partitions time by calendar month,
enabling modeling of patterns that vary by month or season (e.g., heating
behavior in winter months, cooling in summer, holiday patterns in December).
"""

from datetime import datetime
from typing import Self

from custom_components.discrete_state_forecaster.model.time_indexers.time_indexer import (
    TimeIndexer,
)


class MonthIndexer(TimeIndexer):
    """
    Indexes timestamps by calendar month.

    Maps each timestamp to its calendar month (January through December),
    producing consistent bucket keys for the same month regardless of year.
    This is useful for modeling seasonal patterns and monthly recurring
    behaviors.

    The indexer uses standard month numbering:
    - January = 1
    - February = 2
    - March = 3
    - April = 4
    - May = 5
    - June = 6
    - July = 7
    - August = 8
    - September = 9
    - October = 10
    - November = 11
    - December = 12

    Attributes:
        name: Always "month" for this indexer type.

    Example:
        >>> indexer = MonthIndexer()
        >>> indexer.key(datetime(2026, 1, 15))  # January
        1
        >>> indexer.key(datetime(2025, 1, 20))  # Also January
        1
        >>> indexer.key(datetime(2026, 12, 25))  # December
        12
        >>> indexer.next_boundary(datetime(2026, 1, 15, 14, 30))
        datetime(2026, 2, 1, 0, 0)  # First day of next month at midnight
    """

    name = "month"

    async def key(self: Self, ts: datetime) -> int:
        """
        Returns the month number for a timestamp.

        Maps the timestamp to its calendar month number (1-12) where
        January=1 and December=12.

        Args:
            ts: The timestamp to map to a month.

        Returns:
            An integer from 1 to 12 representing the calendar month.

        Example:
        ```
            >>> indexer.key(datetime(2026, 1, 15))
            1
            >>> indexer.key(datetime(2026, 6, 30))
            6
            >>> indexer.key(datetime(2026, 12, 31))
            12
        ```
        """
        return ts.month

    async def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Returns midnight of the first day of the next month.

        Computes the timestamp when the month changes, which is always
        midnight (00:00:00) of the first day of the following calendar month.
        Handles year transitions from December to January.

        Args:
            ts: The reference timestamp.

        Returns:
            Midnight of the first day of the next month. If the current month
            is December, returns January 1st of the following year.

        Example:
        ```
            >>> indexer.next_boundary(datetime(2026, 1, 15, 14, 30))
            datetime(2026, 2, 1, 0, 0, 0, 0)
            >>> indexer.next_boundary(datetime(2026, 12, 31, 23, 59, 59))
            datetime(2027, 1, 1, 0, 0, 0, 0)
        ```
        """
        if ts.month == 12:  # noqa: PLR2004
            return ts.replace(
                year=ts.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        return ts.replace(
            month=ts.month + 1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
