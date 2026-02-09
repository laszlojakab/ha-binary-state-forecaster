from __future__ import annotations

from typing import TYPE_CHECKING, Final

from .distribution_stats import DistributionStats

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.temporal.time_key import (
        TimeKey,
    )


class AggregatedStats(DistributionStats):
    def __init__(self, key: TimeKey) -> None:
        super().__init__()
        self.key: Final[TimeKey] = key

    @classmethod
    def from_distribution(
        cls,
        distribution: DistributionStats,
        key: TimeKey,
    ) -> AggregatedStats:
        agg = cls(key)

        total_support = distribution.total_support()

        for state, prob in distribution.distribution().items():
            agg.update(state, prob * total_support)

        return agg
