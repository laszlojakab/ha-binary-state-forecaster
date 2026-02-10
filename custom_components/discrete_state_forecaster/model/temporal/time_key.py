"""Hierarchical temporal key representation.

This module provides `TimeKey`, a data structure that represents a hierarchical
composition of temporal features. TimeKey instances form parent-child chains,
where each node in the chain adds a TemporalFeature, creating a path from the
global root to a specific temporal context.

A TimeKey can be serialized to/from tuples for storage and transmission, and
supports hashing and equality comparison for use in dictionaries and sets.

Examples:
    >>> from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import TemporalFeature  # noqa: E501
    >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
    >>> key.to_tuple()
    (('hour', 14),)
    >>> deeper = key + TemporalFeature("weekday", 3)
    >>> len(deeper)
    2
    >>> list(deeper.hierarchy())  # Root to leaf
    [TimeKey(...), TimeKey(...), TimeKey.GLOBAL]

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

from .temporal_feature import TemporalFeature, TemporalFeatureName, TemporalFeatureValue

if TYPE_CHECKING:
    from collections.abc import Iterator


class TimeKey:
    """A hierarchical key representing a path through temporal features.

    A TimeKey is an immutable linked-list-like structure where each node
    contains a single TemporalFeature and a reference to its parent TimeKey.
    This creates a hierarchy from a root GLOBAL node down to specific temporal
    contexts.

    The root node `TimeKey.GLOBAL` represents no temporal context and is used
    as the starting point for building hierarchical keys. Each subsequent node
    adds a single temporal feature, building up a context path.

    TimeKeys are hashable and comparable, making them suitable for use as
    dictionary keys or set members. The hash and equality are based on the
    complete tuple representation (all features from root to current node).

    Attributes:
        GLOBAL: The root TimeKey node representing no temporal context.
        parent: Reference to the parent TimeKey (or None for GLOBAL).
        feature: The TemporalFeature at this node (or None for GLOBAL).
        is_root: Property indicating if this is the GLOBAL root node.

    Examples:
        >>> # Create a hierarchical key
        >>> hour_feature = TemporalFeature("hour", 14)
        >>> day_feature = TemporalFeature("day_of_week", 3)
        >>> key = TimeKey.GLOBAL + hour_feature + day_feature
        >>> len(key)
        2
        >>> key.to_tuple()
        (('hour', 14), ('day_of_week', 3))

        >>> # Use as dict key
        >>> patterns = {key: "afternoon on Wednesday"}
        >>> patterns[key]
        'afternoon on Wednesday'

    """

    GLOBAL: TimeKey

    def __init__(
        self: Self,
        parent: TimeKey | None = None,
        feature: TemporalFeature | None = None,
    ):
        """Initialize a TimeKey node.

        Args:
            parent: The parent TimeKey (or None for root). Defaults to None.
            feature: The TemporalFeature at this node (or None for root).
                Defaults to None.

        Note:
            Use `TimeKey.GLOBAL` to access the root node, or use the `+`
            operator to create new keys by adding features.

        Examples:
            >>> # Usually created via TimeKey.GLOBAL + feature
            >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14)

        """
        self.parent: Final[TimeKey | None] = parent
        self.feature: Final[TemporalFeature | None] = feature

    @property
    def is_root(self) -> bool:
        """Check if this is the root GLOBAL node.

        Returns:
            True if this is TimeKey.GLOBAL, False otherwise.

        Examples:
            >>> TimeKey.GLOBAL.is_root
            True
            >>> (TimeKey.GLOBAL + TemporalFeature("hour", 14)).is_root
            False

        """
        return self == TimeKey.GLOBAL

    def ancestors(self) -> Iterator[TimeKey]:
        """Iterate through ancestor nodes from immediate parent to root.

        Yields nodes starting with the immediate parent and ending at the root
        (but not including the current node).

        Yields:
            TimeKey nodes in the parent chain, from immediate parent to root.

        Examples:
            >>> key = TimeKey.GLOBAL + TemporalFeature("a", 1) + TemporalFeature("b", 2)
            >>> list(key.ancestors())  # [parent, grandparent]
            [TimeKey(...), TimeKey(...)]

        """
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def hierarchy(self) -> Iterator[TimeKey]:
        """Iterate through the complete hierarchy from self to root.

        Yields the current node first, then all ancestors in order from child
        to root.

        Yields:
            TimeKey nodes from current node to root inclusive.

        Examples:
            >>> key = TimeKey.GLOBAL + TemporalFeature("a", 1) + TemporalFeature("b", 2)
            >>> hierarchy = list(key.hierarchy())
            >>> len(hierarchy)
            3  # self, parent, GLOBAL

        """
        yield self
        yield from self.ancestors()

    def to_tuple(
        self: Self,
    ) -> tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]:
        """Convert the key to a tuple of feature tuples.

        Collects all features from the current node up to (but not including)
        the root, then reverses them to produce root-to-leaf order.

        Returns:
            A tuple of (name, value) pairs ordered from root to leaf.

        Examples:
            >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14) + TemporalFeature("day", 3)
            >>> key.to_tuple()
            (('hour', 14), ('day', 3))

        """
        items: list[tuple[TemporalFeatureName, TemporalFeatureValue]] = []
        current: TimeKey | None = self
        while current is not None and current.feature is not None:
            items.append(current.feature.to_tuple())
            current = current.parent

        items.reverse()
        return tuple(items)

    @classmethod
    def from_tuple(cls, data: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]) -> Self:
        """Construct a TimeKey from a tuple of feature tuples.

        Args:
            data: A tuple of (name, value) pairs ordered from root to leaf.
                Empty tuple constructs the GLOBAL root node.

        Returns:
            A TimeKey representing the given hierarchy.

        Examples:
            >>> key = TimeKey.from_tuple((("hour", 14), ("day", 3)))
            >>> len(key)
            2
            >>> key.to_tuple()
            (('hour', 14), ('day', 3))

        """
        if not data:
            return cls.GLOBAL

        current: TimeKey = cls.GLOBAL
        for feature in data:
            current = cls(current, TemporalFeature.from_tuple(feature))

        return current

    @classmethod
    def from_temporal_feature(cls, feature: TemporalFeature) -> Self:
        """Construct a single-feature TimeKey from a TemporalFeature.

        Args:
            feature: A TemporalFeature to wrap in a TimeKey.

        Returns:
            A TimeKey with the given feature as the only non-root element.

        Examples:
            >>> feature = TemporalFeature("hour", 14)
            >>> key = TimeKey.from_temporal_feature(feature)
            >>> key.to_tuple()
            (('hour', 14),)

        """
        return cls(cls.GLOBAL, feature)

    def __hash__(self: Self) -> int:
        """Generate a hash based on the complete tuple representation.

        Returns:
            An integer hash value.

        Note:
            Equal TimeKeys have equal hashes. The hash is stable across
            multiple calls.

        """
        return hash(self.to_tuple())

    def __eq__(self: Self, other: object) -> bool:
        """Check equality based on tuple representation.

        Args:
            other: Another object to compare with.

        Returns:
            True if both are TimeKeys with identical tuple representations.

        Examples:
            >>> k1 = TimeKey.from_tuple((("hour", 14),))
            >>> k2 = TimeKey.from_tuple((("hour", 14),))
            >>> k1 == k2
            True
            >>> k1 is k2
            False

        """
        return isinstance(other, TimeKey) and self.to_tuple() == other.to_tuple()

    def __len__(self: Self) -> int:
        """Return the number of features in the hierarchy.

        The root GLOBAL node has length 0.

        Returns:
            The count of TemporalFeatures (all nodes except root).

        Examples:
            >>> TimeKey.GLOBAL.__len__()
            0
            >>> key = TimeKey.from_tuple((("a", 1), ("b", 2)))
            >>> len(key)
            2

        """
        count = 0
        current: TimeKey | None = self
        while current is not None and current.feature is not None:
            count += 1
            current = current.parent

        return count

    def __repr__(self: Self) -> str:
        """Return a human-readable string representation.

        Shows feature names and values in root-to-leaf order separated by
        commas. The root GLOBAL node is shown as "GLOBAL".

        Returns:
            A readable string like "hour = 14, day = 3".

        Examples:
            >>> TimeKey.GLOBAL.__repr__()
            'GLOBAL'
            >>> key = TimeKey.from_tuple((("hour", 14), ("day", 3)))
            >>> repr(key)
            'hour = 14, day = 3'

        """
        if self.parent is None and self.feature is None:
            return "GLOBAL"

        parts: list[str] = []
        current: TimeKey | None = self
        while current is not None and current.feature is not None:
            parts.append(repr(current.feature))
            current = current.parent

        parts.reverse()
        return ", ".join(parts)

    def __add__(self: Self, other: TemporalFeature | TimeKey) -> Self:
        """Create a new TimeKey by appending a feature or another TimeKey.

        Can append either a single TemporalFeature or the features of another
        TimeKey. The result is a new TimeKey with the added feature(s).

        Args:
            other: A TemporalFeature or TimeKey to append.

        Returns:
            A new TimeKey with the appended feature(s).

        Raises:
            TypeError: If other is neither TemporalFeature nor TimeKey.

        Examples:
            >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
            >>> deeper = key + TemporalFeature("day", 3)
            >>> deeper.to_tuple()
            (('hour', 14), ('day', 3))

            >>> # Appending a TimeKey appends all its features
            >>> other_key = TimeKey.from_tuple((("season", "summer"),))
            >>> combined = key + other_key
            >>> combined.to_tuple()
            (('hour', 14), ('season', 'summer'))

        """
        if isinstance(other, TemporalFeature):
            return self.__class__(self, other)

        if isinstance(other, TimeKey):
            # Find the root of the other TimeKey
            features = []
            current: TimeKey | None = other
            while current is not None and current.feature is not None:
                features.append(current.feature)
                current = current.parent
            features.reverse()

            result: TimeKey = self
            for feature in features:
                result = self.__class__(result, feature)
            return result

        raise TypeError(f"Cannot add TimeKey and {type(other).__name__}")


# Initialize the GLOBAL constant
TimeKey.GLOBAL = TimeKey()
