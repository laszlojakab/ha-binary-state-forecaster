"""
Day-of-week indexer implementation.

This module provides `DayOfWeekIndexer`, a TimeIndexer implementation that
maps timestamps to their corresponding day of the week. This enables the
forecaster to learn week-specific patterns (e.g., different behavior on
weekdays vs weekends).

Days are represented as integers following Python's datetime convention:
- 0 = Monday
- 1 = Tuesday
- 2 = Wednesday
- 3 = Thursday
- 4 = Friday
- 5 = Saturday
- 6 = Sunday
"""

from datetime import datetime
from typing import Final, Self

from .temporal_feature import TemporalFeature
from .time_indexer import (
    TimeIndexer,
)
from .time_key import TimeKey


class DayOfWeekIndexer(TimeIndexer):
    """
    Maps timestamps to their day of the week.

    This indexer extracts the day of the week from a timestamp and creates
    a temporal feature. Days are represented as integers 0-6 where 0 is
    Monday and 6 is Sunday.

    This enables learning of weekly patterns where behavior may differ by
    day (e.g., weekday vs weekend behavior).

    Attributes:
        name: Always set to "day_of_week" - the name of the temporal feature.

    Examples:
        >>> indexer = DayOfWeekIndexer()
        >>> # Monday, January 15, 2024
        >>> monday = datetime(2024, 1, 15, 10, 30)
        >>> key = await indexer.get_key(monday)
        >>> key.to_tuple()
        (('day_of_week', 0),)  # 0 = Monday
        >>> # Saturday, January 20, 2024
        >>> saturday = datetime(2024, 1, 20, 10, 30)
        >>> key2 = await indexer.get_key(saturday)
        >>> key2.to_tuple()
        (('day_of_week', 5),)  # 5 = Saturday

    """

    name: Final = "day_of_week"

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Map a timestamp to its day of the week.

        Args:
            timestamp: The datetime to map to a day-of-week value.

        Returns:
            A TimeKey with a single feature: ("day_of_week", weekday_number)
            where weekday_number is 0-6 (Monday-Sunday).

        Examples:
            >>> indexer = DayOfWeekIndexer()
            >>> # All times on any Monday map to day_of_week=0
            >>> monday1 = datetime(2024, 1, 8, 10, 30)
            >>> monday2 = datetime(2024, 1, 15, 20, 45)
            >>> key1 = await indexer.get_key(monday1)
            >>> key2 = await indexer.get_key(monday2)
            >>> key1.to_tuple()
            (('day_of_week', 0),)
            >>> key2.to_tuple()
            (('day_of_week', 0),)
            >>> # Friday
            >>> friday = datetime(2024, 1, 19, 10, 30)
            >>> key3 = await indexer.get_key(friday)
            >>> key3.to_tuple()
            (('day_of_week', 4),)

        """
        weekday = timestamp.weekday()  # Monday=0, Sunday=6

        return TimeKey.from_temporal_feature(TemporalFeature(name=self.name, value=weekday))
