from datetime import datetime

from .temporal_feature import TemporalFeature
from .time_indexer import TimeIndexer
from .time_key import TimeKey


class MonthIndexer(TimeIndexer):
    async def get_key(self, timestamp: datetime) -> TimeKey:
        month = timestamp.month
        return TimeKey.from_temporal_feature(
            TemporalFeature(name=self.name, value=month)
        )

    async def next_boundary(self, timestamp: datetime) -> datetime:
        tz = timestamp.tzinfo
        month = timestamp.month
        year = timestamp.year

        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        return datetime(next_year, next_month, 1, tzinfo=tz)
