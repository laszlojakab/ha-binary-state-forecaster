from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

TemporalFeatureName = str
TemporalFeatureValue = Hashable


@dataclass(frozen=True)
class TemporalFeature:
    name: TemporalFeatureName
    value: TemporalFeatureValue

    def __repr__(self: Self) -> str:
        return f"{self.name} = {self.value}"

    def to_tuple(self: Self) -> tuple[TemporalFeatureName, TemporalFeatureValue]:
        return (self.name, self.value)

    @classmethod
    def from_tuple(cls, data: tuple[TemporalFeatureName, TemporalFeatureValue]) -> Self:
        if not isinstance(data, tuple):
            msg = f"Expected tuple, got {type(data).__name__}"
            raise TypeError(msg)
        if len(data) != 2:
            msg = f"Expected 2-tuple, got {len(data)}-tuple"
            raise ValueError(msg)
        return cls(name=data[0], value=data[1])
