"""
Utilities for hierarchical temporal keys used by the forecaster.

This module defines the `TimeKey` class, an immutable, ordered
collection of temporal feature name/value pairs that form a hierarchical
key (for example, `("day", 3) + ("hour", 14)`).

The `TimeKey` helpers provide convenient navigation of the
hierarchy (parent, ancestors, hierarchy), combination of keys, and
basic introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .temporal_feature import TemporalFeatureName, TemporalFeatureValue


@dataclass(frozen=True)
class TimeKey:
    """
    Immutable hierarchical key composed of ordered temporal feature parts.

    Each part is a `(feature_name, feature_value)` tuple. The `TimeKey`
    represents a specific point in the temporal hierarchy (for example,
    day -> hour -> minute). The class is immutable and hashable.

    A global root key is available as :attr:`GLOBAL` and is represented by an
    empty parts tuple.
    """

    GLOBAL: ClassVar[TimeKey]

    parts: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...] = ()

    def __init__(
        self: Self, *parts: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]
    ) -> None:
        """
        Creates a new `TimeKey`.

        Accepts a variable number of `(feature_name, feature_value)` tuples
        as positional arguments. When no parts are provided the created key is
        the global root.
        """
        object.__setattr__(self, "parts", parts)

    @property
    def is_root(self) -> bool:
        """
        Returns `True` when this key is the global root key.

        The global root is represented by an empty parts tuple and is
        available as :attr:`GLOBAL`.
        """
        return self == TimeKey.GLOBAL

    @property
    def parent(self) -> TimeKey | None:
        """
        Returns the parent `TimeKey` or `None` if this is root.

        The parent is constructed by dropping the last part from `parts`.
        """
        if self.is_root:
            return None

        return TimeKey(*self.parts[:-1])

    def ancestors(self) -> Iterator[TimeKey]:
        """
        Yields ancestor keys from the immediate parent up to the root.

        Ancestors are yielded in order from nearest (immediate parent) to
        furthest (the global root). The sequence does not include `self`.
        """
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def hierarchy(self) -> Iterator[TimeKey]:
        """
        Yields `self` followed by its ancestors (nearest first).

        This yields a convenient traversal of the full hierarchy from the
        current key up to the global root.
        """
        yield self
        yield from self.ancestors()

    def __len__(self: Self) -> int:
        """Returns the number of parts in the key (hierarchy depth)."""
        return len(self.parts)

    def __repr__(self: Self) -> str:
        """
        Returns a human-readable representation of the key.

        The global root is represented as `'GLOBAL'`. Otherwise returns a
        comma-separated list of `name: value` pairs.
        """
        if len(self.parts) == 0:
            return "GLOBAL"

        return ", ".join(f"{part[0]}: {part[1]}" for part in self.parts)

    def __add__(
        self: Self, other: tuple[TemporalFeatureName, TemporalFeatureValue] | TimeKey
    ) -> Self:
        """
        Combines this key with another part or `TimeKey`.

        - If `other` is a `(name, value)` tuple, return a new `TimeKey`
          with that part appended.
        - If `other` is another `TimeKey`, return a new key that is the
          concatenation of both key parts (`self` followed by `other`).

        Raises:
          TypeError: if `other` is not a 2-tuple or `TimeKey`.
        """
        if isinstance(other, tuple) and len(other) == 2:  # noqa: PLR2004
            return self.__class__(*(*self.parts, other))

        if isinstance(other, TimeKey):
            return self.__class__(*(*self.parts, *other.parts))

        raise TypeError(f"Cannot add TimeKey and {type(other).__name__}")


# Initialize the GLOBAL constant
TimeKey.GLOBAL = TimeKey()
