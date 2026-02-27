"""
Indexed storage of multiple distribution statistics.

This module provides `KeyedDistributionStore`, a container that manages multiple
`DistributionStats` instances indexed by arbitrary hashable keys. It enables
bulk operations like decay and pruning, as well as aggregation across multiple
distributions for hierarchical prediction models.

The store is useful in hierarchical temporal models where different temporal
contexts (e.g., different times of day) need independent statistics that can
still be aggregated when making predictions.
"""

from __future__ import annotations

from dataclasses import astuple
from typing import TYPE_CHECKING, Any, Self

from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .distribution_stats import DistributionStats

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable


class KeyedDistributionStore:
    """
    Storage and management of multiple indexed distribution statistics.

    Maintains a dictionary mapping arbitrary hashable keys to DistributionStats
    objects. Supports bulk operations across all stored distributions and
    provides aggregation across multiple distributions with confidence thresholds.

    Example:
        >>> store = KeyedDistributionStore()
        >>> store.update("breakfast", "on", weight=2.0)
        >>> store.update("breakfast", "off", weight=1.0)
        >>> store.update("evening", "on", weight=1.0)
        >>> dist = store.get_distribution("breakfast")
        >>> dist.distribution()
        {'on': 0.6667, 'off': 0.3333}

    """

    _store: dict[TimeKey, DistributionStats]
    """Dictionary mapping TimeKey keys to their associated DistributionStats."""

    def __init__(self: Self) -> None:
        """Initializes an empty distribution store."""
        self._store: dict[TimeKey, DistributionStats] = {}

    def update(
        self: Self,
        key: TimeKey,
        state: Hashable,
        weight: float = 1.0,
        decay_factor: float | None = None,
    ) -> None:
        """
        Updates state support in the distribution for a given key.

        Creates a new DistributionStats for the key if it doesn't exist yet.
        If `decay_factor` is provided, applies it to the existing distribution
        **before** adding the new observation. This implements per-key
        observation-weighted decay: a key only decays when it receives a new
        observation, so dormant keys (e.g. a ``season=winter`` key observed
        during summer) are frozen in place rather than silently eroded.

        Args:
            key: The TimeKey identifying which distribution to update.
            state: The state to update.
            weight: The weight to add to the state. Defaults to 1.0.
            decay_factor: Optional decay factor in range (0, 1] to apply to
                this key's distribution before writing the new observation.
                When ``None`` (default) no decay is applied, preserving the
                previous bulk-decay behaviour.

        """
        stats = self._store.get(key)
        if stats is None:
            stats = DistributionStats()
            self._store[key] = stats
        elif decay_factor is not None:
            stats.apply_decay(decay_factor)

        stats.update(state, weight)

    def apply_decay(self: Self, factor: float) -> None:
        """
        Applies exponential decay to all distributions in the store.

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
        absolute_minimum_support: float = 20.0,
    ) -> None:
        """
        Removes infrequent states from all distributions and empty entries.

        First applies adaptive pruning to each distribution using the given
        thresholds, then removes distributions that have become empty after
        pruning.

        Args:
            epsilon: Relative threshold factor (default 0.003). States with
                support < epsilon * total_support are candidates for removal.
            absolute_minimum_support: Absolute minimum support (default 20.0). Ensures
                infrequently observed states are removed even if total support
                is high.

        """
        for dist in list(self._store.values()):
            dist.prune_adaptive(epsilon, absolute_minimum_support)

        self._store = {k: d for k, d in self._store.items() if not d.is_empty}

    def get_distribution(self, key: TimeKey) -> DistributionStats | None:
        """
        Gets the distribution for a specific key.

        Args:
            key: The TimeKey identifying the distribution.

        Returns:
            The DistributionStats for the key, or None if the key doesn't
                exist or was removed by pruning.

        """
        return self._store.get(key)

    # TODO: not used!
    def aggregate(
        self: Self,
        keys: Iterable[TimeKey],
        min_support: float,
    ) -> tuple[DistributionStats, TimeKey] | None:
        """
        Aggregates statistics across multiple distributions until confident.

        Iterates through the provided keys in order, combining their
        distributions until the aggregated total support meets or exceeds
        min_support. Returns the aggregated distribution and the last key
        processed. If no key reaches the threshold, returns the aggregated
        distribution with the last key if any support exists.

        This is useful in hierarchical models where you want to fall back to
        higher-level (broader) temporal contexts when lower-level contexts
        don't have enough confidence.

        Args:
            keys: Iterable of TimeKey keys to aggregate in order.
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
            >>> agg.total_support
            27.0

        """
        aggregated = DistributionStats()

        for key in keys:
            stats = self._store.get(key)
            if not stats:
                continue

            for state, prob in stats.distribution.items():
                aggregated.update(
                    state,
                    prob * stats.total_support,
                )

            if aggregated.is_confident(min_support):
                return aggregated, key

        return (aggregated, None) if aggregated.total_support > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the store to a dictionary for persistence.

        Returns:
          A dictionary representation of the store, where each key maps to the serialized
          form of its DistributionStats.
        """
        return {
            "store": [
                {
                    "key": astuple(k)[0],
                    "stats": v.to_dict(),
                }
                for k, v in self._store.items()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyedDistributionStore:
        """
        Deserializes a KeyedDistributionStore from a dictionary.

        Args:
          data: A dictionary where each key maps to the serialized form of a DistributionStats.

        Returns:
          A KeyedDistributionStore instance reconstructed from the provided dictionary.
        """
        store = cls()
        for item in data.get("store", []):
            key = TimeKey(*tuple(tuple(k) for k in item["key"]))
            stats = DistributionStats.from_dict(item["stats"])
            store._store[key] = stats

        return store
