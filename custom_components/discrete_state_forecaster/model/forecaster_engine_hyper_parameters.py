from __future__ import annotations

from typing import Any, Self


class ForecasterEngineHyperParameters:
    def __init__(
        self: Self,
        half_life: float,
        min_prune_interval_factor: float,
        prune_enabled: bool,
        persistence_strength: float,
    ) -> None:
        self._half_life = half_life
        self._min_prune_interval_factor = min_prune_interval_factor
        self._prune_enabled = prune_enabled
        self._persistence_strength = persistence_strength

        self._base_half_life = half_life
        self._base_min_prune_interval_factor = min_prune_interval_factor
        self._base_prune_enabled = prune_enabled
        self._base_persistence_strength = persistence_strength

    def reset(self: Self) -> None:
        self._half_life = self._base_half_life
        self._min_prune_interval_factor = self._base_min_prune_interval_factor
        self._prune_enabled = self._base_prune_enabled
        self._persistence_strength = self._base_persistence_strength

    def update(
        self: Self,
        *,
        half_life: float | None = None,
        min_prune_interval_factor: float | None = None,
        prune_enabled: bool | None = None,
        persistence_strength: float | None = None,
    ) -> None:
        if half_life is not None:
            self._half_life = half_life
        if min_prune_interval_factor is not None:
            self._min_prune_interval_factor = min_prune_interval_factor
        if prune_enabled is not None:
            self._prune_enabled = prune_enabled
        if persistence_strength is not None:
            self._persistence_strength = persistence_strength

    @property
    def half_life(self: Self) -> float:
        return self._half_life

    @property
    def prune_enabled(self: Self) -> bool:
        return self._prune_enabled

    @property
    def persistence_strength(self: Self) -> float:
        return self._persistence_strength

    @property
    def min_prune_interval_factor(self: Self) -> float:
        return self._min_prune_interval_factor

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "half_life": self._half_life,
            "min_prune_interval_factor": self._min_prune_interval_factor,
            "prune_enabled": self._prune_enabled,
            "persistence_strength": self._persistence_strength,
            "base_half_life": self._base_half_life,
            "base_min_prune_interval_factor": self._base_min_prune_interval_factor,
            "base_prune_enabled": self._base_prune_enabled,
            "base_persistence_strength": self._base_persistence_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecasterEngineHyperParameters:
        parameters = cls(
            half_life=data["half_life"],
            min_prune_interval_factor=data["min_prune_interval_factor"],
            prune_enabled=data["prune_enabled"],
            persistence_strength=data["persistence_strength"],
        )
        parameters._base_half_life = data["base_half_life"]
        parameters._base_min_prune_interval_factor = data[
            "base_min_prune_interval_factor"
        ]
        parameters._base_prune_enabled = data["base_prune_enabled"]
        parameters._base_persistence_strength = data["base_persistence_strength"]

        return parameters
