from typing import Self


class HyperParameters:
    def __init__(
        self: Self,
        half_life: float,
        min_prune_interval: float,
        prune_enabled: bool,
        persistence_strength: float,
    ) -> None:
        self._half_life = half_life
        self._min_prune_interval = min_prune_interval
        self._prune_enabled = prune_enabled
        self._persistence_strength = persistence_strength

        self._base_half_life = half_life
        self._base_min_prune_interval = min_prune_interval
        self._base_prune_enabled = prune_enabled
        self._base_persistence_strength = persistence_strength

    def reset(self: Self) -> None:
        self._half_life = self._base_half_life
        self._min_prune_interval = self._base_min_prune_interval
        self._prune_enabled = self._base_prune_enabled
        self._persistence_strength = self._base_persistence_strength

    def update(
        self: Self,
        *,
        half_life: float | None = None,
        min_prune_interval: float | None = None,
        prune_enabled: bool | None = None,
        persistence_strength: float | None = None,
    ) -> None:
        if half_life is not None:
            self._half_life = half_life
        if min_prune_interval is not None:
            self._min_prune_interval = min_prune_interval
        if prune_enabled is not None:
            self._prune_enabled = prune_enabled
        if persistence_strength is not None:
            self._persistence_strength = persistence_strength

    @property
    def half_life(self: Self) -> float:
        return self._half_life

    @property
    def min_prune_interval(self: Self) -> float:
        return self._min_prune_interval

    @property
    def prune_enabled(self: Self) -> bool:
        return self._prune_enabled

    @property
    def persistence_strength(self: Self) -> float:
        return self._persistence_strength
