from collections.abc import Iterable
from datetime import datetime
from typing import Final, Self

from .time_indexer import TimeIndexer
from .time_key import TimeKey


class CompositeIndexer(TimeIndexer):
    def __init__(self: Self, indexers: Iterable[TimeIndexer]) -> None:
        self.indexers: Final = list(indexers)

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        current: TimeKey = TimeKey.GLOBAL
        for indexer in self.indexers:
            current += await indexer.get_key(timestamp)

        return current
