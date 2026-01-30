"""
Type definition for composite time indexer keys.

This module defines the TimeKey class used throughout the time indexing system.
A TimeKey represents a multi-dimensional temporal bucket as a tuple of (name, value) pairs,
where each pair corresponds to a specific time dimension (e.g., time of day, day of week, month).

The TimeKey type enables consistent representation of composite time buckets across
different time indexer implementations, supporting both simple and complex temporal
patterns in state prediction models.

TimeKeys are immutable and hashable, making them suitable for use as dictionary keys
in caching and lookup structures. They support hierarchical navigation through the
parent() method, allowing traversal from specific to general temporal contexts.

Example:
    A TimeKey might represent "Monday at 10:00-10:30 AM" as:
    ```python
    key = TimeKey((("day_of_week", 0), ("time_of_day", 600)))

    # Navigate to parent (just the day, no specific time)
    day_key = key.parent()  # TimeKey((("day_of_week", 0),))

    # Iterate through hierarchy
    for k in key.parents():
        print(k)  # Prints key → day_key → TimeKey.GLOBAL
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator


class TimeKey:
    """
    Immutable, hashable temporal key for hierarchical time-conditioned statistics.

    A TimeKey represents a point in a multi-dimensional temporal hierarchy, consisting
    of (dimension_name, value) pairs that define a specific temporal context. Each key
    can have a parent key (representing a more general context) by removing the last
    dimension, enabling hierarchical temporal pattern learning.

    Attributes:
        items: Tuple of (dimension_name, value) pairs representing the temporal dimensions.
               Each dimension_name is a string (e.g., "hour", "weekday"), and each value
               is a hashable value identifying the specific temporal bucket.
        GLOBAL: Class-level constant representing the global temporal context (empty key).

    Examples:
        >>> # Create a specific temporal context
        >>> key = TimeKey((("hour", 10), ("weekday", 2)))
        >>> print(key)
        TimeKey((('hour', 10), ('weekday', 2)))

        >>> # Navigate up the hierarchy
        >>> parent = key.parent()
        >>> print(parent)
        TimeKey((('hour', 10),))

        >>> # Global context
        >>> global_key = TimeKey.GLOBAL
        >>> print(global_key)
        TimeKey.GLOBAL
        >>> len(global_key)
        0

        >>> # Iterate through all hierarchical levels
        >>> for k in key.parents():
        ...     print(k)
        TimeKey((('hour', 10), ('weekday', 2)))
        TimeKey((('hour', 10),))
        TimeKey.GLOBAL
    """

    GLOBAL: TimeKey  # forward declaration

    def __init__(self, items: tuple[tuple[str, Hashable], ...] | None = None):
        """
        Initialize a TimeKey with the given dimension items.

        Args:
            items: Optional tuple of (dimension_name, value) pairs. If None or omitted,
                   creates an empty key (global context). Each pair consists of:
                   - dimension_name (str): Name of the temporal dimension (e.g., "hour", "weekday")
                   - value (Hashable): Value identifying the specific bucket in that dimension

                   Dimensions should typically be ordered from most general to most specific,
                   though this is not enforced.

        Examples:
            >>> # Empty key (global context)
            >>> global_key = TimeKey()
            >>> print(global_key)
            TimeKey.GLOBAL

            >>> # Single dimension
            >>> hourly_key = TimeKey((("hour", 15),))
            >>> len(hourly_key)
            1

            >>> # Multiple dimensions
            >>> specific_key = TimeKey((("weekday", 1), ("hour", 15), ("minute_bucket", 2)))
            >>> len(specific_key)
            3
        """
        self.items: tuple[tuple[str, Hashable], ...] = items or ()

    def __hash__(self: Self) -> int:
        """
        Return the hash value for this TimeKey.

        Returns:
            int: Hash value based on the items tuple, suitable for dict/set operations.

        Examples:
            >>> key1 = TimeKey((("hour", 15),))
            >>> key2 = TimeKey((("hour", 15),))
            >>> hash(key1) == hash(key2)
            True
        """
        return hash(self.items)

    def __eq__(self: Self, other: object) -> bool:
        """
        Check equality with another object.

        Two TimeKeys are equal if they have identical items tuples.

        Args:
            other: Object to compare with.

        Returns:
            bool: True if other is a TimeKey with the same items, False otherwise.

        Examples:
            >>> key1 = TimeKey((("hour", 15),))
            >>> key2 = TimeKey((("hour", 15),))
            >>> key3 = TimeKey((("hour", 16),))
            >>> key1 == key2
            True
            >>> key1 == key3
            False
            >>> key1 == "not a timekey"
            False
        """
        if not isinstance(other, TimeKey):
            return False
        return self.items == other.items

    def __len__(self: Self) -> int:
        """
        Return the number of dimensions in this TimeKey.

        Returns:
            int: Number of (dimension_name, value) pairs.

        Examples:
            >>> len(TimeKey.GLOBAL)
            0
            >>> len(TimeKey((("hour", 15),)))
            1
            >>> len(TimeKey((("weekday", 1), ("hour", 15))))
            2
        """
        return len(self.items)

    def __repr__(self: Self) -> str:
        """
        Return a string representation of this TimeKey.

        Returns:
            str: String showing "TimeKey.GLOBAL" for empty keys or "TimeKey(...)" with items.

        Examples:
            >>> repr(TimeKey.GLOBAL)
            'TimeKey.GLOBAL'
            >>> repr(TimeKey((("hour", 15),)))
            "TimeKey((('hour', 15),))"
        """
        if self.items:
            return f"TimeKey({self.items})"
        return "TimeKey.GLOBAL"

    def parent(self: Self) -> Self | None:
        """
        Returns a new TimeKey with the last dimension removed.

        The parent key represents a more general temporal context by removing
        the most specific dimension (the last element in the items tuple).
        This enables hierarchical navigation from specific to general contexts.

        Returns:
            Optional[TimeKey]: Parent key with the last element removed, or None
                              if this is the global key (which has no parent).

        Examples:
            >>> # Multi-dimension key
            >>> key = TimeKey((("weekday", 1), ("hour", 15)))
            >>> parent = key.parent()
            >>> print(parent)
            TimeKey((('weekday', 1),))

            >>> # Single dimension key
            >>> key = TimeKey((("hour", 15),))
            >>> parent = key.parent()
            >>> print(parent)
            TimeKey.GLOBAL

            >>> # Global key has no parent
            >>> TimeKey.GLOBAL.parent() is None
            True
        """
        if not self.items:
            return None
        return TimeKey(self.items[:-1])

    def parents(self: Self) -> Iterator[Self]:
        """
        Iterator yielding this key, then parent, grandparent, etc., up to GLOBAL.

        This method provides a convenient way to traverse the hierarchical chain
        from the most specific context (this key) to the most general context (GLOBAL).
        The iteration includes both endpoints: self and GLOBAL.

        Yields:
            TimeKey: Keys from most specific to most general, ending with GLOBAL.

        Examples:
            >>> # Iterate through all hierarchical levels
            >>> key = TimeKey((("weekday", 1), ("hour", 15)))
            >>> for k in key.parents():
            ...     print(k)
            TimeKey((('weekday', 1), ('hour', 15)))
            TimeKey((('weekday', 1),))
            TimeKey.GLOBAL

            >>> # Single dimension
            >>> key = TimeKey((("hour", 15),))
            >>> list(key.parents())
            [TimeKey((('hour', 15),)), TimeKey.GLOBAL]

            >>> # Global only yields itself
            >>> list(TimeKey.GLOBAL.parents())
            [TimeKey.GLOBAL]
        """
        current: Self | None = self
        while current is not None:
            yield current
            current = current.parent()


# Initialize the GLOBAL constant
TimeKey.GLOBAL = TimeKey()
