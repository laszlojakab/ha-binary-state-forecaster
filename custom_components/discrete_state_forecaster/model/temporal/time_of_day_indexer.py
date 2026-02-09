from datetime import datetime
from typing import Final, Self

from .temporal_feature import TemporalFeature
from .time_indexer import (
    TimeIndexer,
)
from .time_key import TimeKey


class TimeOfDayIndexer(TimeIndexer):
    name: Final = "time_bucket"

    def __init__(self: Self, bucket_size: int):
        self.bucket_size: Final = bucket_size

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        bucket_index = total_seconds // self.bucket_size

        return TimeKey.from_temporal_feature(TemporalFeature(name=self.name, value=bucket_index))
