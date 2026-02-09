from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

from .temporal_feature import TemporalFeature, TemporalFeatureName, TemporalFeatureValue

if TYPE_CHECKING:
    from collections.abc import Iterator


class TimeKey:
    GLOBAL: TimeKey

    def __init__(
        self: Self,
        parent: TimeKey | None = None,
        feature: TemporalFeature | None = None,
    ):
        self.parent: Final[TimeKey | None] = parent
        self.feature: Final[TemporalFeature | None] = feature

    @property
    def is_root(self) -> bool:
        return self == TimeKey.GLOBAL

    def ancestors(self) -> Iterator[TimeKey]:
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def hierarchy(self) -> Iterator[TimeKey]:
        yield self
        yield from self.ancestors()

    def to_tuple(
        self: Self,
    ) -> tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]:
        items: list[tuple[TemporalFeatureName, TemporalFeatureValue]] = []
        current: TimeKey | None = self
        while current is not None and current.feature is not None:
            items.append(current.feature.to_tuple())
            current = current.parent

        items.reverse()
        return tuple(items)

    @classmethod
    def from_tuple(cls, data: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]) -> Self:
        if not data:
            return cls.GLOBAL

        current: TimeKey = cls.GLOBAL
        for feature in data:
            current = cls(current, TemporalFeature.from_tuple(feature))

        return current

    @classmethod
    def from_temporal_feature(cls, feature: TemporalFeature) -> Self:
        return cls(cls.GLOBAL, feature)

    def __hash__(self: Self) -> int:
        return hash(self.to_tuple())

    def __eq__(self: Self, other: object) -> bool:
        return isinstance(other, TimeKey) and self.to_tuple() == other.to_tuple()

    def __len__(self: Self) -> int:
        count = 0
        current: TimeKey | None = self
        while current is not None and current.feature is not None:
            count += 1
            current = current.parent

        return count

    def __repr__(self: Self) -> str:
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
