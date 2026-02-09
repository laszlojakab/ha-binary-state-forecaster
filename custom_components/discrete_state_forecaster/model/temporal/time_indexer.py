from datetime import datetime
from typing import Protocol, Self

from .time_key import TimeKey


class TimeIndexer(Protocol):
    name: str

    async def get_key(self: Self, timestamp: datetime) -> TimeKey: ...
