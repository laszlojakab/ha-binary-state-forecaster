from typing import Self

from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (
    HierarchicalStateStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .contribution import Contribution
from .keyed_distribution_store import KeyedDistributionStore
from .prediction_result import PredictionResult


class HierarchicalStateStats:
    def __init__(self: Self, hyper_parameters: HierarchicalStateStatsHyperParameters):
        self._stats = KeyedDistributionStore()
        self._hyper_parameters = hyper_parameters

    def update(
        self: Self,
        key: TimeKey,
        state: State,
        weight: float,
    ) -> None:
        for ancestor in key.hierarchy():
            self._stats.update(ancestor, state, weight)

    def predict(self: Self, key: TimeKey) -> PredictionResult | None:
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
        self._stats.apply_decay(factor)

    def prune(self: Self, epsilon: float = 0.003, absolute_min: float = 20.0) -> None:
        self._stats.prune(epsilon, absolute_min)
