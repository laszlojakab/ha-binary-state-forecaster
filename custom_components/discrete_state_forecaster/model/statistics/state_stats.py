from dataclasses import dataclass
from typing import Self


@dataclass
class StateStats:
    _support: float = 0.0

    def update(self: Self, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("weight must be non negative")

        self._support += weight

    def apply_decay(self: Self, factor: float) -> None:
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"decay factor must be in (0, 1]. Got: {factor}")

        self._support *= factor

        if self._support < 1e-12:
            self._support = 0.0

    def support(self: Self) -> float:
        return self._support

    def is_active(self: Self, min_support: float) -> bool:
        return self._support >= min_support
