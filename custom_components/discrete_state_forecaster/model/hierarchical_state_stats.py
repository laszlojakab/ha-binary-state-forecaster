"""
Hierarchical state statistics model for multi-level temporal pattern learning.

This module provides the HierarchicalStateStats class, which manages state statistics
across multiple temporal granularities (e.g., specific hour + weekday, just weekday, global).
It enables sophisticated state prediction by combining fine-grained and coarse-grained
patterns with support-weighted blending, automatic pruning, and exponential decay.
"""

from typing import Final, Self

from custom_components.discrete_state_forecaster.model.aggregated_stats import (
    AggregatedStats,
)
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.state_stats import StateStats
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)

# Minimum support time (seconds) required for a TimeKey to contribute to distribution
# TODO: Make this domain-specific or configurable
MIN_SUPPORT: Final[float] = 30.0  # seconds


class HierarchicalStateStats:
    """
    Manages state statistics across hierarchical temporal keys with automatic decay and pruning.

    This class maintains a dictionary of StateStats objects indexed by TimeKey, where each
    TimeKey represents a specific temporal context (e.g., "Monday at 10 AM"). It provides
    hierarchical blending of statistics, exponential decay for temporal relevance, and
    automatic pruning to manage memory usage.

    Attributes:
        stats: Dictionary mapping TimeKey objects to StateStats.
        half_life: Time in seconds for exponential decay. Default is 3600.0 (1 hour).
        last_prune_ts: Timestamp of the last pruning operation.
        prune_interval: Minimum time in seconds between pruning operations.
                        Default is 21600 (6 hours).
        max_keys: Maximum number of TimeKey entries to retain. Default is 50,000.
    """

    def __init__(self: Self, half_life: float = 3600.0) -> None:
        """Initialize an empty hierarchical state statistics tracker."""
        self.stats: dict[TimeKey, StateStats] = {}
        self.half_life: float = half_life
        self.last_prune_ts: float = 0.0
        self.prune_interval: float = 6 * 3600
        self.max_keys: int = 50_000

    def distribution(self: Self, key: TimeKey) -> AggregatedStats:
        """
        Computes state probability distribution using hierarchical blending.

        Traverses from the specific TimeKey through its parent hierarchy,
        collecting statistics at each level that has sufficient support.
        Blends these distributions weighted by their total support.

        Args:
            key: The TimeKey representing the temporal context.

        Returns:
            AggregatedStats with three fields:
            - distribution: Dict mapping states to probabilities (summing to 1.0)
            - support_time: Total accumulated support time across all levels used
            - depth: Number of hierarchy levels that contributed (had >= MIN_SUPPORT)
            Returns empty distribution with 0.0 support and depth count if no level
            has sufficient support (>= MIN_SUPPORT).
        """
        weighted: dict[State, float] = {}
        total_support = 0.0
        depth = 0

        # Traverse from exact → parent → GLOBAL hierarchy
        for k in key.parents():
            stats = self.stats.get(k)
            if not stats:
                continue

            support = stats.total()
            if support < MIN_SUPPORT:
                continue

            dist = stats.distribution()

            for state, prob in dist.items():
                weighted[state] = weighted.get(state, 0.0) + prob * support

            total_support += support
            depth += 1

        if total_support == 0.0:
            return AggregatedStats(distribution={}, support_time=0.0, depth=depth)

        # normalize
        norm_dist = {state: w / total_support for state, w in weighted.items()}

        return AggregatedStats(
            distribution=norm_dist, support_time=total_support, depth=depth
        )

    def prune(
        self: Self,
        now_ts: float,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
        min_total: float = 60.0,
    ) -> None:
        """
        Prunes insignificant statistics and applies decay to manage memory and relevance.

        Performs maintenance by:
        1. Checking if enough time has passed since last pruning
        2. Applying exponential decay to all statistics
        3. Removing insignificant states within each StateStats
        4. Removing entire TimeKey entries with insufficient total support

        Args:
            now_ts: Current timestamp in seconds.
            epsilon: Relative threshold for pruning states. Default is 0.003 (0.3%).
            absolute_min: Absolute minimum duration (seconds) for state retention. Default is 20.0.
            min_total: Minimum total duration (seconds) for TimeKey retention. Default is 60.0.
        """
        if now_ts - self.last_prune_ts < self.prune_interval:
            return

        self.last_prune_ts = now_ts

        keys_to_delete: list[TimeKey] = []

        for key, stats in self.stats.items():

            # Decay all stats first
            stats.apply_decay(now_ts, self.half_life)

            # prune states inside the stat
            stats.prune_adaptive(
                epsilon=epsilon,
                absolute_min=absolute_min,
            )

            # remove empty / near-empty stats
            if stats.total() < min_total:
                keys_to_delete.append(key)

        # delete after iteration
        for key in keys_to_delete:
            del self.stats[key]

    def update(
        self: Self,
        key: TimeKey,
        state: State,
        duration: float,
        ts: float,
    ) -> None:
        """
        Records a state observation at a specific temporal context and all parent levels.

        Updates the statistics for the given TimeKey AND all its parent keys in the hierarchy
        by adding the duration to the specified state. This ensures that statistics are
        automatically aggregated at all levels (specific → general → global), enabling
        effective hierarchical blending in the distribution method.

        For example, updating key=(("hour", 10), ("weekday", 1)) will also update:
        - The parent key (("hour", 10),)
        - The global key (empty tuple)

        If a TimeKey doesn't exist, creates a new StateStats entry. Applies decay before
        updating to maintain temporal relevance.

        Args:
            key: The TimeKey representing the temporal context.
            state: The state value that was observed (e.g., "on", "off").
            duration: The time in seconds spent in this state.
            ts: The timestamp in seconds when this observation occurred.

        Side Effects:
            - Creates StateStats entries for key and all parent keys if they don't exist
            - Applies decay to all affected StateStats before updating
            - Updates durations for the state across all hierarchy levels
            - May trigger key limit enforcement if max_keys threshold is exceeded
        """
        for k in key.parents():
            stats = self.stats.get(k)
            if stats is None:
                stats = StateStats()
                self.stats[k] = stats

            stats.apply_decay(ts, self.half_life)
            stats.update_duration(state, duration)

        if len(self.stats) > self.max_keys * 1.1:
            self.enforce_key_limit()

    def enforce_key_limit(self: Self) -> None:
        """
        Enforces the maximum key limit by removing least-supported entries.

        When the number of TimeKey entries exceeds max_keys, this method removes
        the entries with the smallest total support (lowest cumulative duration)
        until the dictionary size returns to max_keys.
        """
        if len(self.stats) <= self.max_keys:
            return

        overflow = len(self.stats) - self.max_keys

        # smallest totals first
        sorted_keys = sorted(self.stats.items(), key=lambda kv: kv[1].total())

        for key, _ in sorted_keys[:overflow]:
            del self.stats[key]
