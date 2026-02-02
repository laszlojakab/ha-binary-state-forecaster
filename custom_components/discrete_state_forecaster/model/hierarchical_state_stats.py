"""
Hierarchical state statistics model for multi-level temporal pattern learning.

This module provides the HierarchicalStateStats class, which manages state statistics
across multiple temporal granularities (e.g., specific hour + weekday, just weekday, global).
It enables sophisticated state prediction by combining fine-grained and coarse-grained
patterns with support-weighted blending, automatic pruning, and exponential decay.
"""

import time
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
    hierarchical blending of statistics, exponential decay for temporal relevance, automatic
    pruning to manage memory usage, and concept drift detection to adapt to pattern changes.

    Features:
    - Hierarchical temporal pattern tracking across multiple granularity levels
    - Support-weighted blending of distributions from specific to general patterns
    - Exponential decay for temporal relevance with configurable half-life
    - Automatic pruning of insignificant statistics
    - Concept drift detection with accelerated adaptation on pattern shifts
    - Memory management through key limit enforcement

    Attributes:
        stats: Dictionary mapping TimeKey objects to StateStats.
        half_life: Time in seconds for exponential decay. Default is 3600.0 (1 hour).
        last_prune_ts: Timestamp of the last pruning operation.
        prune_interval: Minimum time in seconds between pruning operations.
                        Default is 21600 (6 hours).
        prune_every_n_updates: Prune after this many updates (if set). If None (default),
                               uses only time-based pruning. When set, pruning occurs
                               after N updates OR when time interval elapses (whichever
                               comes first).
        update_count: Counter tracking number of updates since last prune.
        max_keys: Maximum number of TimeKey entries to retain. Default is 50,000.
    """

    def __init__(
        self: Self,
        half_life: float = 3600.0,
        prune_interval: float = 21600.0,
        prune_every_n_updates: int | None = None,
    ) -> None:
        """
        Initializes an empty hierarchical state statistics tracker.

        Args:
            half_life: Time in seconds for exponential decay. Default is 3600.0 (1 hour).
            prune_interval: Minimum time in seconds between pruning operations.
                           Default is 21600.0 (6 hours).
            prune_every_n_updates: If set, prune after this many updates in addition to
                                   time-based pruning. If None (default), uses only
                                   time-based pruning. When set, pruning occurs when
                                   either N updates have occurred OR the time interval
                                   has elapsed (whichever comes first).
        """
        self.stats: dict[TimeKey, StateStats] = {}
        self.half_life: float = half_life
        self.last_prune_ts: float = 0.0
        self.prune_interval: float = prune_interval
        self.prune_every_n_updates: int | None = prune_every_n_updates
        self.update_count: int = 0
        self.max_keys: int = 50_000

    def distribution(
        self: Self, key: TimeKey, timestamp: float | None = None
    ) -> AggregatedStats:
        """
        Computes state probability distribution using hierarchical blending.

        First checks if the specific TimeKey has sufficient data (>= MIN_SUPPORT).
        If yes, returns ONLY that level's distribution without blending parents.
        If no, traverses through parent hierarchy and blends distributions weighted
        by their total support.

        This approach prioritizes specific temporal patterns when they are well-established,
        and only falls back to more general patterns when specific data is insufficient.

        Args:
            key: The TimeKey representing the temporal context.
            timestamp: The timestamp in seconds for applying decay before computing
                distribution. If None (default), uses current time from time.time().

        Returns:
            AggregatedStats with three fields:
            - distribution: Dict mapping states to probabilities (summing to 1.0)
            - support_time: Total accumulated support time across all levels used
            - depth: Number of hierarchy levels that contributed (had >= MIN_SUPPORT)
            Returns empty distribution with 0.0 support and depth count if no level
            has sufficient support (>= MIN_SUPPORT).

        Example:
            >>> stats = HierarchicalStateStats()
            >>> key = TimeKey((("hour", 10),))
            >>> stats.update(key, "on", 100.0, timestamp=1000.0)
            >>> # Use current time
            >>> result = stats.distribution(key)
            >>> # Or specify explicit timestamp for determinism
            >>> result = stats.distribution(key, timestamp=1500.0)
        """
        if timestamp is None:
            timestamp = time.time()

        # Check if the specific key has sufficient data
        specific_stats = self.stats.get(key)
        if specific_stats:
            specific_stats.apply_decay(timestamp, self.half_life)
            specific_support = specific_stats.total()

            # If specific key has enough support, use ONLY it (no blending)
            if specific_support >= MIN_SUPPORT:
                return AggregatedStats(
                    distribution=specific_stats.distribution(),
                    support_time=specific_support,
                    key=key,
                )

        # Specific key lacks sufficient data, blend across hierarchy
        weighted: dict[State, float] = {}
        total_support = 0.0

        # Track the parent key that provides the most support (for traceability)
        best_key: TimeKey | None = None
        best_support = 0.0

        # Traverse from exact → parent → GLOBAL hierarchy
        for k in key.parents():
            stats = self.stats.get(k)
            if not stats:
                continue

            stats.apply_decay(timestamp, self.half_life)

            support = stats.total()
            if support < MIN_SUPPORT:
                continue

            dist = stats.distribution()

            for state, prob in dist.items():
                weighted[state] = weighted.get(state, 0.0) + prob * support

            total_support += support

            # remember which key had the largest raw support
            if support > best_support:
                best_support = support
                best_key = k

        if total_support == 0.0:
            return AggregatedStats(
                distribution={}, support_time=0.0, key=TimeKey.GLOBAL
            )

        # normalize
        norm_dist = {state: w / total_support for state, w in weighted.items()}

        return AggregatedStats(
            distribution=norm_dist,
            support_time=total_support,
            key=best_key or TimeKey.GLOBAL,
        )

    def prune(
        self: Self,
        timestamp: float | None = None,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
        min_total: float = 60.0,
    ) -> None:
        """
        Prunes insignificant statistics and applies decay to manage memory and relevance.

        Performs maintenance by:
        1. Checking if enough time has passed since last pruning OR if enough updates have occurred
        2. Applying exponential decay to all statistics
        3. Removing insignificant states within each StateStats
        4. Removing entire TimeKey entries with insufficient total support

        Pruning is triggered when EITHER:
        - Time interval has elapsed (prune_interval), OR
        - Update count threshold is reached (prune_every_n_updates, if configured)

        Args:
            timestamp: Current timestamp in seconds. If None (default), uses current
                time from time.time().
            epsilon: Relative threshold for pruning states. Default is 0.003 (0.3%).
            absolute_min: Absolute minimum duration (seconds) for state retention. Default is 20.0.
            min_total: Minimum total duration (seconds) for TimeKey retention. Default is 60.0.

        Example:
            >>> stats = HierarchicalStateStats()
            >>> # Use current time for pruning
            >>> stats.prune()
            >>> # Or specify explicit timestamp
            >>> stats.prune(timestamp=1000.0)
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize last_prune_ts on first call to prevent immediate pruning
        if self.last_prune_ts == 0.0:
            self.last_prune_ts = timestamp
            return

        # Check both time-based and update-based pruning conditions
        time_based_prune = timestamp - self.last_prune_ts >= self.prune_interval
        update_based_prune = (
            self.prune_every_n_updates is not None
            and self.update_count >= self.prune_every_n_updates
        )

        if not (time_based_prune or update_based_prune):
            return

        self.last_prune_ts = timestamp
        self.update_count = 0

        keys_to_delete: list[TimeKey] = []

        for key, stats in self.stats.items():
            # Decay all stats first
            stats.apply_decay(timestamp, self.half_life)

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
        timestamp: float | None = None,
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

        Concept Drift Detection:
        At each hierarchy level, checks for concept drift (significant distribution shifts)
        using the StateStats.check_drift() method. When drift is detected:
        - Sets fast_decay_updates to 15 for that level
        - Triggers accelerated adaptation to new patterns
        - Helps the model quickly adjust to behavioral changes

        The fast_decay_updates counter decrements on each subsequent update, providing
        a brief period of enhanced learning after pattern shifts are detected.

        Args:
            key: The TimeKey representing the temporal context.
            state: The state value that was observed (e.g., "on", "off").
            duration: The time in seconds spent in this state.
            timestamp: The timestamp in seconds when this observation occurred. If None
                (default), uses current time from time.time().

        Side Effects:
            - Creates StateStats entries for key and all parent keys if they don't exist
            - Applies decay to all affected StateStats before updating
            - Updates durations for the state across all hierarchy levels
            - Checks for concept drift at each level
            - Sets fast_decay_updates=15 when drift is detected
            - Decrements fast_decay_updates counter if > 0
            - May trigger key limit enforcement if max_keys threshold is exceeded
            - Automatically triggers pruning when conditions are met (based on prune_interval
              or prune_every_n_updates), removing insignificant entries to manage memory

        Example:
            >>> stats = HierarchicalStateStats()
            >>> key = TimeKey((("hour", 10), ("weekday", 1)))
            >>> # Use current time
            >>> stats.update(key, "on", 100.0)
            >>> # Or specify explicit timestamp for determinism
            >>> stats.update(key, "on", 100.0, timestamp=1000.0)
            >>> # If pattern changes significantly, drift is detected automatically
            >>> # and fast_decay_updates is set for adaptive learning
        """
        if timestamp is None:
            timestamp = time.time()

        self.update_count += 1

        for k in key.parents():
            stats = self.stats.get(k)
            if stats is None:
                stats = StateStats()
                self.stats[k] = stats

            stats.apply_decay(timestamp, self.half_life)
            stats.update_duration(state, duration)

            if stats.check_drift(timestamp):
                stats.fast_decay_updates = 15

            if stats.fast_decay_updates > 0:
                stats.fast_decay_updates -= 1

        if len(self.stats) > self.max_keys * 1.1:
            self.enforce_key_limit()

        # Auto-prune when conditions are met (respects prune_interval and prune_every_n_updates)
        self.prune(timestamp)

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

    def to_dict(self: Self) -> dict[str, any]:
        """
        Serializes the HierarchicalStateStats to a dictionary.

        Returns:
            Dictionary containing all hierarchical state statistics data.
        """
        return {
            "stats": [
                # Convert TimeKey to JSON-serializable format
                # Store as list of [key_data, stats_data] pairs
                [key.to_tuple(), stats.to_dict()]
                for key, stats in self.stats.items()
            ],
            "half_life": self.half_life,
            "last_prune_ts": self.last_prune_ts,
            "prune_interval": self.prune_interval,
            "prune_every_n_updates": self.prune_every_n_updates,
            "update_count": self.update_count,
            "max_keys": self.max_keys,
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Self:
        """
        Deserializes a HierarchicalStateStats from a dictionary.

        Args:
            data: Dictionary containing serialized HierarchicalStateStats data.

        Returns:
            Restored HierarchicalStateStats instance.
        """
        instance = cls(
            half_life=data.get("half_life", 3600.0),
            prune_interval=data.get("prune_interval", 21600.0),
            prune_every_n_updates=data.get("prune_every_n_updates"),
        )

        # Restore stats dictionary with TimeKey conversion
        # Data is stored as list of [key_data, stats_data] pairs
        stats_data = data.get("stats", [])
        for key_data, stats_dict in stats_data:
            # Convert list back to TimeKey
            key = TimeKey.from_tuple(key_data)
            instance.stats[key] = StateStats.from_dict(stats_dict)

        instance.last_prune_ts = data.get("last_prune_ts", 0.0)
        instance.update_count = data.get("update_count", 0)
        instance.max_keys = data.get("max_keys", 50_000)

        return instance
