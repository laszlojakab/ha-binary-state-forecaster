"""
Month-based temporal indexer.

This module provides `MonthIndexer`, a `TimeIndexer` that maps a
timestamp to a month-based `TimeKey` (1-12) and computes the
next-month boundary (the first day of the following month).

The indexer preserves the input `datetime`'s `tzinfo` when
computing boundaries.
"""

from datetime import datetime

from .time_indexer import TimeIndexer
from .time_key import TimeKey


class MonthIndexer(TimeIndexer):
    """Indexer that extracts the calendar month from a timestamp."""

    async def get_key(self, timestamp: datetime) -> TimeKey:
        """
        Returns a `TimeKey` for the calendar month of `timestamp`.

        The returned `TimeKey` contains a `TemporalFeature` whose
        `name` is the indexer's `name` and whose `value` is the
        month number (1-12) extracted from `timestamp`.
        """
        month = timestamp.month
        return TimeKey((self.name, month))

    async def next_boundary(self, timestamp: datetime) -> datetime:
        """
        Returns a `datetime` representing the first day of the next month.

        The returned `datetime` preserves the `tzinfo` from `timestamp`.
        If the input month is December, the year rolls over accordingly.
        """
        tz = timestamp.tzinfo
        month = timestamp.month
        year = timestamp.year

        if month == 12:  # noqa: PLR2004
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        return datetime(next_year, next_month, 1, tzinfo=tz)
