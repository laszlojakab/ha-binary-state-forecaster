from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .temporal_feature import TemporalFeatureName, TemporalFeatureValue


@dataclass(frozen=True)
class TimeKey:
    GLOBAL: ClassVar[TimeKey]

    parts: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...] = ()

    def __init__(
        self: Self, *parts: tuple[tuple[TemporalFeatureName, TemporalFeatureValue], ...]
    ) -> None:
        object.__setattr__(self, "parts", parts)

    @property
    def is_root(self) -> bool:
        return self == TimeKey.GLOBAL

    @property
    def parent(self) -> TimeKey | None:
        if self.is_root:
            return None

        return TimeKey(*self.parts[:-1])

    def ancestors(self) -> Iterator[TimeKey]:
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def hierarchy(self) -> Iterator[TimeKey]:
        yield self
        yield from self.ancestors()

    def __len__(self: Self) -> int:
        return len(self.parts)

    def __repr__(self: Self) -> str:
        if len(self.parts) == 0:
            return "GLOBAL"

        return ", ".join(f"{part[0]}: {part[1]}" for part in self.parts)

    def __add__(
        self: Self, other: tuple[TemporalFeatureName, TemporalFeatureValue] | TimeKey
    ) -> Self:
        if isinstance(other, tuple) and len(other) == 2:
            return self.__class__(*(*self.parts, other))

        if isinstance(other, TimeKey):
            # Find the root of the other TimeKey
            return self.__class__(*(*self.parts, *other.parts))

        raise TypeError(f"Cannot add TimeKey and {type(other).__name__}")


# Initialize the GLOBAL constant
TimeKey.GLOBAL = TimeKey()
