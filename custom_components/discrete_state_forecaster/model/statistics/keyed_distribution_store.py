from collections.abc import Hashable, Iterable
from typing import Self

from .distribution_stats import DistributionStats


class KeyedDistributionStore:
    def __init__(self: Self) -> None:
        self._store: dict[Hashable, DistributionStats] = {}

    def update(
        self: Self,
        key: Hashable,
        state: Hashable,
        weight: float = 1.0,
    ) -> None:
        stats = self._store.get(key)
        if stats is None:
            stats = DistributionStats()
            self._store[key] = stats

        stats.update(state, weight)

    def apply_decay(self: Self, factor: float) -> None:
        for stats in self._store.values():
            stats.apply_decay(factor)

    def prune(
        self: Self,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
    ) -> None:
        for dist in list(self._store.values()):
            dist.prune_adaptive(epsilon, absolute_min)

        self._store = {k: d for k, d in self._store.items() if not d.is_empty()}

    def get_distribution(self, key: Hashable) -> DistributionStats | None:
        return self._store.get(key)

    def aggregate(
        self: Self,
        keys: Iterable[Hashable],
        min_support: float,
    ) -> tuple[DistributionStats, Hashable] | None:
        aggregated = DistributionStats()

        for key in keys:
            stats = self._store.get(key)
            if not stats:
                continue

            for state, prob in stats.distribution().items():
                aggregated.update(
                    state,
                    prob * stats.total_support(),
                )

            if aggregated.is_confident(min_support):
                return aggregated, key

        return (aggregated, None) if aggregated.total_support() > 0 else None
