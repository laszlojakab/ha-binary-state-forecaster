"""
Immutable temporal feature representation.

This module provides `TemporalFeature`, an immutable data structure representing
a single temporal feature as a name-value pair. Temporal features are the atomic
building blocks of hierarchical temporal keys used throughout the forecaster.

Examples:
    >>> feature = TemporalFeature("hour", 14)
    >>> feature.name
    'hour'
    >>> feature.value
    14
    >>> repr(feature)
    'hour = 14'
    >>> feature.to_tuple()
    ('hour', 14)

"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

TemporalFeatureName = str
TemporalFeatureValue = Hashable


@dataclass(frozen=True)
class TemporalFeature:
    """
    An immutable name-value pair representing a single temporal feature.

    TemporalFeature instances are frozen (immutable) and hashable, making them
    suitable for use as dictionary keys and in sets. They serve as the atomic
    units composed into hierarchical TimeKey structures.

    Attributes:
        name: A string identifier for the feature (e.g., "hour", "weekday").
        value: A hashable value associated with the feature. Can be any hashable
            type (int, str, bool, tuple, etc.).

    Examples:
        >>> feature = TemporalFeature("day_of_week", "Monday")
        >>> feature in {feature}  # Can be used in sets
        True
        >>> {feature: "weekday"}[feature]  # Can be used as dict key
        'weekday'

    """

    name: TemporalFeatureName
    value: TemporalFeatureValue

    def __repr__(self: Self) -> str:
        """Returns a readable string representation of the feature."""
        return f"{self.name} = {self.value}"

    def to_tuple(self: Self) -> tuple[TemporalFeatureName, TemporalFeatureValue]:
        """
        Converts the feature to a tuple for serialization.

        Returns:
            A 2-tuple of (name, value).

        Examples:
            >>> TemporalFeature("hour", 14).to_tuple()
            ('hour', 14)

        """
        return (self.name, self.value)

    @classmethod
    def from_tuple(cls, data: tuple[TemporalFeatureName, TemporalFeatureValue]) -> Self:
        """
        Constructs a TemporalFeature from a tuple.

        Args:
            data: A 2-tuple of (name, value) to reconstruct the feature.

        Returns:
            A new TemporalFeature instance.

        Raises:
            TypeError: If data is not a tuple.
            ValueError: If data does not have exactly 2 elements.

        Examples:
            >>> feature = TemporalFeature.from_tuple(("hour", 14))
            >>> feature.name
            'hour'
            >>> feature.value
            14

        """
        if not isinstance(data, tuple):
            msg = f"Expected tuple, got {type(data).__name__}"
            raise TypeError(msg)
        if len(data) != 2:  # noqa: PLR2004
            msg = f"Expected 2-tuple, got {len(data)}-tuple"
            raise ValueError(msg)
        return cls(name=data[0], value=data[1])
