"""
Season time indexer.

This module provides `SeasonIndexer`, a `TimeIndexer` implementation that
maps timestamps to meteorological seasons and computes the next season
boundary. Seasons follow Northern Hemisphere meteorological definitions:
- spring: March 1 — May 31
- summer: June 1 — August 31
- autumn: September 1 — November 30
- winter: December 1 — February 28/29

The indexer methods are asynchronous to match the time indexer protocol used
by the forecaster framework.
"""

from datetime import datetime
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey

from .time_indexer import (
    TimeIndexer,
)


class SeasonIndexer(TimeIndexer):
    """
    Maps timestamps to meteorological seasons (Northern Hemisphere).

    This indexer maps each timestamp to one of four meteorological seasons
    based on the month. Meteorological seasons are defined by temperature
    cycles and are more consistent for climate analysis than astronomical
    seasons.

    Seasons enable the forecaster to learn season-specific patterns where
    behavior may differ significantly (e.g., heating patterns differ between
    summer and winter).

    Attributes:
        name: Always set to "season" - the name of the temporal feature.

    Examples:
        >>> indexer = SeasonIndexer()
        >>> # March is spring
        >>> spring = datetime(2024, 3, 15, 10, 30)
        >>> key = await indexer.get_key(spring)
        >>> key.to_tuple()
        (('season', 'spring'),)
        >>> # July is summer
        >>> summer = datetime(2024, 7, 15, 10, 30)
        >>> key2 = await indexer.get_key(summer)
        >>> key2.to_tuple()
        (('season', 'summer'),)

    """

    name: Final = "season"

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Returns a TimeKey with the season name for the given timestamp.

        Uses meteorological seasons (Northern Hemisphere):
        - "spring": March 1 — May 31
        - "summer": June 1 — August 31
        - "autumn": September 1 — November 30
        - "winter": December 1 — February 28/29

        Args:
            timestamp: Timestamp to map to a season.

        Returns:
            A TimeKey with a single feature: ("season", season_name)
            where season_name is one of "spring", "summer", "autumn", "winter".

        Examples:
            >>> indexer = SeasonIndexer()
            >>> # Any date in March is spring
            >>> ts1 = datetime(2024, 3, 1, 0, 0)
            >>> key1 = await indexer.get_key(ts1)
            >>> key1.to_tuple()
            (('season', 'spring'),)
            >>> ts2 = datetime(2024, 12, 15, 12, 0)
            >>> key2 = await indexer.get_key(ts2)
            >>> key2.to_tuple()
            (('season', 'winter'),)

        """
        month = timestamp.month
        if 3 <= month <= 5:  # noqa: PLR2004
            season = "spring"
        elif 6 <= month <= 8:  # noqa: PLR2004
            season = "summer"
        elif 9 <= month <= 11:  # noqa: PLR2004
            season = "autumn"
        else:
            season = "winter"

        return TimeKey.from_tuple(((self.name, season),))

    async def next_boundary(self: Self, timestamp: datetime) -> datetime:
        """
        Returns the next season boundary strictly after `timestamp`.

        The next boundary is the start of the next meteorological season at
        00:00:00 local time. This function preserves tzinfo from `timestamp` when
        constructing the returned datetime.

        Args:
            timestamp: Timestamp to find next season boundary after.

        Returns:
            A `datetime` representing the next season start strictly after
            `timestamp`.

        Examples:
            >>> indexer = SeasonIndexer()
            >>> # From mid-March, next boundary is June 1 (summer)
            >>> ts = datetime(2024, 3, 15, 14, 30)
            >>> boundary = await indexer.next_boundary(ts)
            >>> boundary
            datetime(2024, 6, 1, 0, 0, 0)
            >>> # From December, next boundary is March 1 next year
            >>> ts2 = datetime(2024, 12, 15, 14, 30)
            >>> boundary2 = await indexer.next_boundary(ts2)
            >>> boundary2
            datetime(2025, 3, 1, 0, 0, 0)

        """
        tz = timestamp.tzinfo

        # Determine current season and the month when the next season starts
        month = timestamp.month
        year = timestamp.year

        if 3 <= month <= 5:  # spring -> next is summer starting June 1  # noqa: PLR2004
            next_month = 6
            next_year = year
        elif 6 <= month <= 8:  # summer -> autumn starting Sep 1  # noqa: PLR2004
            next_month = 9
            next_year = year
        elif 9 <= month <= 11:  # autumn -> winter starting Dec 1  # noqa: PLR2004
            next_month = 12
            next_year = year
        else:  # winter -> spring starting Mar 1
            next_month = 3
            # if we're in December, next spring is next year; if Jan/Feb, same year
            next_year = year + 1 if month == 12 else year  # noqa: PLR2004

        candidate = datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=tz)

        # If candidate is not strictly after timestamp (edge cases), advance one season
        if candidate <= timestamp:
            # advance by 3 months
            if next_month == 12:  # noqa: PLR2004
                next_month = 3
                next_year = next_year + 1
            else:
                next_month = next_month + 3
                if next_month > 12:  # noqa: PLR2004
                    next_month -= 12
                    next_year += 1
            candidate = datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=tz)

        return candidate
