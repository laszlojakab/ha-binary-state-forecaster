"""Indexed storage of multiple distribution statistics.

This module provides `KeyedDistributionStore`, a container that manages multiple
`DistributionStats` instances indexed by arbitrary hashable keys. It enables
bulk operations like decay and pruning, as well as aggregation across multiple
distributions for hierarchical prediction models.

The store is useful in hierarchical temporal models where different temporal
contexts (e.g., different times of day) need independent statistics that can
still be aggregated when making predictions.
"""
from collections.abc import Hashable, Iterable
from typing import Self

from .distribution_stats import DistributionStats


class KeyedDistributionStore:
    """Storage and management of multiple indexed distribution statistics.

    Maintains a dictionary mapping arbitrary hashable keys to DistributionStats
    objects. Supports bulk operations across all stored distributions and
    provides aggregation across multiple distributions with confidence thresholds.

    Attributes:
        _store: Dictionary mapping hashable keys to their associated
            DistributionStats objects.

    Example:
        >>> store = KeyedDistributionStore()
        >>> store.update("breakfast", "on", weight=2.0)
        >>> store.update("breakfast", "off", weight=1.0)
        >>> store.update("evening", "on", weight=1.0)
        >>> dist = store.get_distribution("breakfast")
        >>> dist.distribution()
        {'on': 0.6667, 'off': 0.3333}

    """

    def __init__(self: Self) -> None:
        """Initialize an empty distribution store."""
        self._store: dict[Hashable, DistributionStats] = {}

    def update(
        self: Self,
        key: Hashable,
        state: Hashable,
        weight: float = 1.0,
    ) -> None:
        """Update state support in the distribution for a given key.

        Creates a new DistributionStats for the key if it doesn't exist yet.
        Then updates the specified state with the given weight.

        Args:
            key: The hashable key identifying which distribution to update.
            state: The state to update.
            weight: The weight to add to the state. Defaults to 1.0.

        """
        stats = self._store.get(key)
        if stats is None:
            stats = DistributionStats()
            self._store[key] = stats

        stats.update(state, weight)

    def apply_decay(self: Self, factor: float) -> None:
        """Apply exponential decay to all distributions in the store.

        Multiplies all state support values in all distributions by the decay
        factor, giving recent observations more weight than older ones.

        Args:
            factor: Decay factor in range (0, 1]. Values closer to 0 produce
                stronger decay.

        """
        for stats in self._store.values():
            stats.apply_decay(factor)

    def prune(
        self: Self,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
    ) -> None:
        """Remove infrequent states from all distributions and empty entries.

        First applies adaptive pruning to each distribution using the given
        thresholds, then removes distributions that have become empty after
        pruning.

        Args:
            epsilon: Relative threshold factor (default 0.003). States with
                support < epsilon * total_support are candidates for removal.
            absolute_min: Absolute minimum support (default 20.0). Ensures
                infrequently observed states are removed even if total support
                is high.

        """
        for dist in list(self._store.values()):
            dist.prune_adaptive(epsilon, absolute_min)

        self._store = {k: d for k, d in self._store.items() if not d.is_empty()}

    def get_distribution(self, key: Hashable) -> DistributionStats | None:
        """Get the distribution for a specific key.

        Args:
            key: The hashable key identifying the distribution.

        Returns:
            The DistributionStats for the key, or None if the key doesn't
                exist or was removed by pruning.

        """
        return self._store.get(key)

    def aggregate(
        self: Self,
        keys: Iterable[Hashable],
        min_support: float,
    ) -> tuple[DistributionStats, Hashable] | None:
        """Aggregate statistics across multiple distributions until confident.

        Iterates through the provided keys in order, combining their
        distributions until the aggregated total support meets or exceeds
        min_support. Returns the aggregated distribution and the last key
        processed. If no key reaches the threshold, returns the aggregated
        distribution with the last key if any support exists.

        This is useful in hierarchical models where you want to fall back to
        higher-level (broader) temporal contexts when lower-level contexts
        don't have enough confidence.

        Args:
            keys: Iterable of hashable keys to aggregate in order.
            min_support: Minimum required total support for confidence.

        Returns:
            Tuple of (aggregated_distribution, last_key_processed) if any
                statistics were found, or None if no keys exist in the store.
                The aggregated distribution accumulates support from all
                processed keys.

        Example:
            >>> store = KeyedDistributionStore()
            >>> store.update("14:00", "on", 5.0)
            >>> store.update("14:00", "off", 2.0)
            >>> store.update("afternoon", "on", 20.0)
            >>> agg, key = store.aggregate([("14:00", "afternoon")], min_support=15.0)
            >>> key
            ('14:00', 'afternoon')
            >>> agg.total_support()  # doctest: +SKIP
            27.0

        """
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
