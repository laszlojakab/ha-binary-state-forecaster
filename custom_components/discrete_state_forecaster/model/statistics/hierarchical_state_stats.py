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

from typing import Any, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (  # noqa: E501
    HierarchicalStateStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .contribution import Contribution
from .keyed_distribution_store import KeyedDistributionStore
from .prediction_result import PredictionResult


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

    Attributes:
        _stats: KeyedDistributionStore that maps TimeKeys to their state
            distributions at all hierarchy levels.
        _hyper_parameters: Configuration parameters including minimum support
            thresholds for predictions.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.temporal.time_key import (
        ...     TimeKey,
        ... )
        >>> from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (  # noqa: E501
        ...     TemporalFeature,
        ... )
        >>> hp = HierarchicalStateStatsHyperParameters(min_support=10.0)
        >>> stats = HierarchicalStateStats(hp)
        >>> key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        >>> stats.update(key, "on", weight=1.0)
        >>> result = stats.predict(key)
        >>> result is not None
        True

    """  # noqa: E501

    def __init__(self: Self, hyper_parameters: HierarchicalStateStatsHyperParameters):
        """
        Initializes the hierarchical state statistics engine.

        Args:
            hyper_parameters: Configuration including minimum support thresholds
                and other prediction parameters.

        """
        self._stats = KeyedDistributionStore()
        self._hyper_parameters = hyper_parameters

    def update(
        self: Self,
        key: TimeKey,
        state: State,
        weight: float,
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

        Args:
            key: The specific TimeKey at which the state was observed.
            state: The state that was observed.
            weight: The weight to increment for this observation.

        """
        for ancestor in key.hierarchy():
            self._stats.update(ancestor, state, weight)

    def predict(self: Self, key: TimeKey) -> PredictionResult | None:
        """
        Predicts state distribution at a given temporal location.

        Uses a confidence-aware hierarchical fallback strategy:
        1. First tries the specific key with min_support threshold
        2. If insufficient, iterates through ancestors with decaying weight
        3. Returns first level with sufficient confidence, or None if no level
           reaches the threshold

        The confidence-weighted fallback ensures that if a specific time has
        limited data but a broader time (ancestor) has abundant data, the
        broader pattern is used but weighted less than specific data would be.

        Args:
            key: The TimeKey representing the temporal location to predict.

        Returns:
            A PredictionResult containing the probability distribution and
                contribution sources if confident enough, or None if no level
                reaches the minimum support threshold.

        """
        specific = self._stats.get_distribution(key)

        if specific and specific.is_confident(self._hyper_parameters.min_support):
            return PredictionResult(
                key=key,
                distribution=specific,
                contributions=(
                    Contribution(
                        key=key,
                        weight=1.0,
                        support=specific.total_support(),
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

            for state, prob in stats.distribution().items():
                aggregated.update(state, prob * stats.total_support() * weight)

            contributions.append(
                Contribution(source_key, weight, stats.total_support())
            )

            if aggregated.is_confident(self._hyper_parameters.min_support):
                break

        if not aggregated.is_confident(self._hyper_parameters.min_support):
            return None

        return PredictionResult(
            key=key,
            distribution=aggregated,
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

    def prune(self: Self, epsilon: float = 0.003, absolute_min: float = 20.0) -> None:
        """
        Removes infrequent states and empty distributions from the hierarchy.

        Uses adaptive pruning to remove states that fall below dynamic thresholds,
        then removes any distributions that become empty. This reduces memory
        usage and focuses the model on frequently-observed patterns.

        Args:
            epsilon: Relative threshold factor (default 0.003). States with
                support < epsilon * total_support are candidates for removal.
            absolute_min: Absolute minimum support (default 20.0). Ensures
                frequently observed states are not removed even if total
                support is very high.

        """
        self._stats.prune(epsilon, absolute_min)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serializes the hyper parameters to a dictionary.

        Returns:
          A dictionary representation of the hierarchical state statistics
        """
        return {
            "hyper_parameters": self._hyper_parameters.to_dict(),
            "stats": self._stats.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> HierarchicalStateStats:
        return cls(
            hyper_parameters=HierarchicalStateStatsHyperParameters.from_dict(
                data["hyper_parameters"], hyper_parameters
            ),
            stats=KeyedDistributionStore.from_dict(data["stats"]),
        )
