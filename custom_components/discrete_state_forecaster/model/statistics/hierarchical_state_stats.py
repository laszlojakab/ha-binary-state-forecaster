"""
Hierarchical temporal state statistics and prediction engine.

This module provides `HierarchicalStateStats`, the core prediction engine that
combines statistics from multiple temporal levels to make predictions. It uses
a hierarchical fallback mechanism where predictions are first attempted at
specific temporal contexts, then fall back to broader contexts (ancestors) if
confidence is insufficient.

The hierarchy allows the model to learn distinct patterns for specific
times (e.g., "14:30 on Wednesday in spring") while still generalizing
when data is
sparse by falling back to broader patterns (e.g., "afternoon" or "spring").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Self

from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)

from .contribution import Contribution
from .hierarchical_state_stats_parameters import HierarchicalStateStatsParameters
from .keyed_distribution_store import KeyedDistributionStore
from .prediction_result import PredictionResult

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.state import State
    from custom_components.discrete_state_forecaster.model.temporal.time_key import (
        TimeKey,
    )

    from .hierarchical_state_stats_hyper_parameters import (
        HierarchicalStateStatsHyperParameters,
    )
    from .hierarchical_state_stats_runtime_parameters import (
        HierarchicalStateStatsRuntimeParameters,
    )


class HierarchicalStateStats:
    """
    Hierarchical temporal state statistics with multi-level prediction fallback.

    Maintains state statistics at multiple temporal hierarchy levels and makes
    predictions using a confidence-aware fallback strategy. When making a
    prediction at a specific temporal location, it first checks if that level
    has sufficient confidence, then falls back to parent contexts with
    decreasing confidence weights.

    The hierarchy levels correspond to different temporal granularities maintained
    by the TimeKey structure (e.g., global → season → day_of_week → hour).

    Example:
        >>> hp = HierarchicalStateStatsHyperParameters(min_support=10.0)
        >>> rp = HierarchicalStateStatsRuntimeParameters(min_support_factor=1.0)
        >>> stats = HierarchicalStateStats(hp, rp)
        >>> key = TimeKey(("hour", 14))
        >>> stats.update(key, "on", weight=1.0)
        >>> result = stats.predict(key)
        >>> result is not None
        True

    """

    _stats: Final[KeyedDistributionStore]
    """Store of distributions for all temporal keys in the hierarchy."""

    def __init__(
        self: Self,
        hyper_parameters: HierarchicalStateStatsHyperParameters,
        runtime_parameters: HierarchicalStateStatsRuntimeParameters,
    ):
        """
        Initializes the hierarchical state statistics engine.

        Args:
            hyper_parameters: Configuration including minimum support thresholds
                and other prediction parameters.
            runtime_parameters: Runtime parameters that can be adjusted during
                execution, such as scaling factors for thresholds.
        """
        self._stats = KeyedDistributionStore()
        self.parameters = HierarchicalStateStatsParameters(
            hyper_parameters, runtime_parameters
        )

    def update(
        self: Self,
        key: TimeKey,
        state: State,
        weight: float,
        decay_factor: float | None = None,
    ) -> None:
        """
        Updates state statistics at all levels of the temporal hierarchy.

        For a given specific TimeKey, this updates the statistics for that key
        and all of its ancestors in the hierarchy. For example, if the key is
        "hour=14, day_of_week=Mon, season=spring", this updates statistics for:
        - (hour=14, day_of_week=Mon, season=spring)
        - (day_of_week=Mon, season=spring)
        - (season=spring)
        - GLOBAL

        This allows the model to learn different patterns at different levels
        of temporal granularity.

        If `decay_factor` is provided it is forwarded to the underlying store
        so that each ancestor key is decayed **in-place** just before the new
        observation is written. Keys that are *not* part of the current
        hierarchy (e.g. ``season=winter`` keys during summer) are not touched
        and therefore retain their accumulated statistics intact.

        Args:
            key: The specific TimeKey at which the state was observed.
            state: The state that was observed.
            weight: The weight to increment for this observation.
            decay_factor: Optional decay factor in range (0, 1] to apply to
                each ancestor key's distribution before writing the new
                observation. When ``None`` (default) no per-key decay is
                applied.

        """
        for ancestor in key.hierarchy():
            self._stats.update(ancestor, state, weight, decay_factor=decay_factor)

    def predict(self: Self, key: TimeKey) -> PredictionResult | None:
        """
        Predicts state distribution at a given temporal location.

        Uses a confidence-aware hierarchical fallback strategy:
        1. First tries the specific key with min_support threshold defined
           in hyper parameters.
        2. If insufficient, iterates through ancestors with decaying weight
        3. Returns first level with sufficient confidence, or None if no level
           reaches the threshold

        The confidence-weighted fallback ensures that if a specific time has
        limited data but a broader time (ancestor) has abundant data, the
        broader pattern is used but weighted less than specific data would be.

        Args:
            key: The TimeKey representing the temporal location to predict.

        Returns:
            A PredictionResult containing the probability distribution, confidence
                and contribution sources if confident enough, or None if no level
                reaches the minimum support threshold.

        """
        specific = self._stats.get_distribution(key)

        if specific and specific.is_confident(self.parameters.min_support):
            return PredictionResult(
                key=key,
                distribution_stats=specific,
                contributions=(
                    Contribution(
                        key=key,
                        weight=1.0,
                        support=specific.total_support,
                    ),
                ),
            )

        # fallback
        aggregated = DistributionStats()
        contributions: list[Contribution] = []

        for level, source_key in enumerate(key.ancestors()):
            stats = self._stats.get_distribution(source_key)
            if not stats:
                continue

            weight = 1.0 / (1.0 + level)

            for state, prob in stats.distribution.items():
                aggregated.update(state, prob * stats.total_support * weight)

            contributions.append(Contribution(source_key, weight, stats.total_support))

            if aggregated.is_confident(self.parameters.min_support):
                break

        if not aggregated.is_confident(self.parameters.min_support):
            return None

        return PredictionResult(
            key=key,
            distribution_stats=aggregated,
            contributions=tuple(contributions),
        )

    def apply_decay(self: Self, factor: float) -> None:
        """
        Applies exponential decay to all statistics in the hierarchy.

        Multiplies all state support values by the decay factor, giving recent
        observations more weight than older ones. This is typically called
        periodically to implement time-decay of observations.

        Args:
            factor: Decay factor in range (0, 1]. Values closer to 0 produce
                stronger decay. For example, 0.95 retains 95% of previous support.

        """
        self._stats.apply_decay(factor)

    def prune(
        self: Self, epsilon: float = 0.003, absolute_minimum_support: float = 20.0
    ) -> None:
        """
        Removes infrequent states and empty distributions from the hierarchy.

        Uses adaptive pruning to remove states that fall below dynamic thresholds,
        then removes any distributions that become empty. This reduces memory
        usage and focuses the model on frequently-observed patterns.

        Args:
            epsilon: Relative threshold factor. States with
                support < epsilon * total_support are candidates for removal.
            absolute_minimum_support: Absolute minimum support. Ensures
                frequently observed states are not removed even if total
                support is very high.

        """
        self._stats.prune(epsilon, absolute_minimum_support)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serializes the instance into a dictionary.

        Returns:
          A dictionary representation of the hierarchical state statistics
        """
        return {
            "stats": self._stats.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        hyper_parameters: HierarchicalStateStatsHyperParameters,
        runtime_parameters: HierarchicalStateStatsRuntimeParameters,
    ) -> HierarchicalStateStats:
        """
        Deserializes a dictionary into an instance of HierarchicalStateStats.

        Args:
            data: Dictionary containing serialized hierarchical state statistics.
            hyper_parameters: Hyper parameters needed to reconstruct the instance.
            runtime_parameters: Runtime parameters needed to reconstruct the instance.
        """
        stats = cls(
            hyper_parameters=hyper_parameters, runtime_parameters=runtime_parameters
        )

        stats._stats = KeyedDistributionStore.from_dict(data["stats"])

        return stats
