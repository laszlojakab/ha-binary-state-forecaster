"""
Type definition for composite time indexer keys.

This module defines the TimeKey type alias used throughout the time indexing system.
A TimeKey represents a multi-dimensional temporal bucket as a tuple of (name, value) pairs,
where each pair corresponds to a specific time dimension (e.g., time of day, day of week, month).

The TimeKey type enables consistent representation of composite time buckets across
different time indexer implementations, supporting both simple and complex temporal
patterns in state prediction models.

Example:
    A TimeKey might represent "Monday at 10:00-10:30 AM" as:
    ```
    (("day_of_week", 0), ("time_of_day", 600))
    ```
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Hashable, Iterator, Optional, Tuple


class TimeKey:
    """
    Immutable, hashable temporal key for hierarchical stats.

    Stores a tuple of (dimension_name, value) pairs, e.g.:
        (("hour", 10), ("weekday", 2), ("month", 1))

    Provides:
    - parent() → returns TimeKey with one fewer level
    - GLOBAL → empty key
    """

    GLOBAL: TimeKey = None  # forward declaration

    def __init__(self, items: Optional[Tuple[Tuple[str, Hashable], ...]] = None):
        self.items: Tuple[Tuple[str, Hashable], ...] = items or ()

    def __hash__(self):
        return hash(self.items)

    def __eq__(self, other):
        if not isinstance(other, TimeKey):
            return False
        return self.items == other.items

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        if self.items:
            return f"TimeKey({self.items})"
        return "TimeKey.GLOBAL"

    def parent(self) -> Optional[TimeKey]:
        """
        Returns a new TimeKey with the last element removed.
        Returns None if already GLOBAL.
        """
        if not self.items:
            return None
        return TimeKey(self.items[:-1])

    def parents(self) -> Iterator[TimeKey]:
        """
        Iterator from self → parent → parent ... → GLOBAL
        """
        current: Optional[TimeKey] = self
        while current is not None:
            yield current
            current = current.parent()
