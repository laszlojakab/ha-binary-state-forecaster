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
from typing import Self

from custom_components.discrete_state_forecaster.model.time_indexers.time_indexer import (
    TimeIndexer,
)


class SeasonIndexer(TimeIndexer):
    name = "season"

    async def key(self: Self, ts: datetime) -> str:
        """
        Returns the season name for the given timestamp.

        Uses meteorological seasons (Northern Hemisphere):
        - "spring": March 1 — May 31
        - "summer": June 1 — August 31
        - "autumn": September 1 — November 30
        - "winter": December 1 — February 28/29

        Args:
            ts: Timestamp to map to a season.

        Returns:
            A string: one of "spring", "summer", "autumn", "winter".
        """
        month = ts.month
        if 3 <= month <= 5:
            return "spring"
        if 6 <= month <= 8:
            return "summer"
        if 9 <= month <= 11:
            return "autumn"
        return "winter"

    async def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Returns the next season boundary strictly after `ts`.

        The next boundary is the start of the next meteorological season at
        00:00:00 local time. This function preserves tzinfo from `ts` when
        constructing the returned datetime.

        Args:
            ts: Timestamp to find next season boundary after.

        Returns:
            A `datetime` representing the next season start strictly after
            `ts`.
        """
        tz = ts.tzinfo

        # Determine current season and the month when the next season starts
        month = ts.month
        year = ts.year

        if 3 <= month <= 5:  # spring -> next is summer starting June 1
            next_month = 6
            next_year = year
        elif 6 <= month <= 8:  # summer -> autumn starting Sep 1
            next_month = 9
            next_year = year
        elif 9 <= month <= 11:  # autumn -> winter starting Dec 1
            next_month = 12
            next_year = year
        else:  # winter -> spring starting Mar 1
            next_month = 3
            # if we're in December, next spring is next year; if Jan/Feb, same year
            next_year = year + 1 if month == 12 else year

        candidate = datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=tz)

        # If candidate is not strictly after ts (edge cases), advance one season
        if candidate <= ts:
            # advance by 3 months
            if next_month == 12:
                next_month = 3
                next_year = next_year + 1
            else:
                next_month = next_month + 3
                if next_month > 12:
                    next_month -= 12
                    next_year += 1
            candidate = datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=tz)

        return candidate
