from datetime import datetime
from typing import Final, Self

from .temporal_feature import TemporalFeature
from .time_indexer import (
    TimeIndexer,
)
from .time_key import TimeKey


class TimeOfDayIndexer(TimeIndexer):
    name: Final = "day_of_week"

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        weekday = timestamp.weekday()  # Monday=0, Sunday=6

        return TimeKey.from_temporal_feature(
            TemporalFeature(name=self.name, value=weekday)
        )
