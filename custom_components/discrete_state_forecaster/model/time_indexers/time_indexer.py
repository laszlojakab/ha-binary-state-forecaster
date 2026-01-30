"""
Protocol for time-based indexing strategies.

This module defines the interface for time indexers that partition time into
buckets for efficient temporal pattern analysis and forecasting.
"""

from collections.abc import Hashable
from datetime import datetime
from typing import Final, Protocol, Self


class TimeIndexer(Protocol):
    """
    Protocol defining the interface for time-based bucket indexing.

    Time indexers partition continuous time into discrete buckets, enabling
    efficient grouping and analysis of temporal patterns. Implementations
    define how time is divided (e.g., by hour, day, time of day) and can
    compute when bucket boundaries occur.

    Attributes:
        name: A unique identifier for this indexer type. Should be a constant
            string that describes the indexing strategy (e.g., "time_bucket",
            "day_of_week").
    """

    name: Final[str]

    def key(self: Self, ts: datetime) -> Hashable:
        """
        Returns the bucket key for a given timestamp.

        Maps a timestamp to a bucket identifier. Timestamps that fall within
        the same conceptual bucket should return the same key value. The key
        can be any hashable type (int, str, tuple, etc.).

        Args:
            ts: The timestamp to map to a bucket.

        Returns:
            A hashable value representing the bucket containing the timestamp.
            Equal keys indicate timestamps belong to the same bucket.

        Example:
            For a time-of-day indexer with 30-minute buckets:
            - 14:15 -> key 28 (14 * 2 + 0)
            - 14:45 -> key 29 (14 * 2 + 1)
            - 15:00 -> key 30 (15 * 2 + 0)
        """

    def next_boundary(self: Self, ts: datetime) -> datetime:
        """
        Returns the next timestamp where the bucket key may change.

        Computes the earliest future timestamp where the key() value would
        differ from the current timestamp's key. This enables efficient
        scheduling and prediction of when bucket transitions occur.

        Args:
            ts: The reference timestamp.

        Returns:
            The next timestamp at or after which the bucket key will change.
            This should be the earliest time > ts where key(ts) != key(result).

        Example:
            For a 30-minute time-of-day indexer:
            - next_boundary(14:15:30) -> 14:30:00
            - next_boundary(14:30:00) -> 15:00:00
            - next_boundary(23:45:00) -> 00:00:00 (next day)
        """
