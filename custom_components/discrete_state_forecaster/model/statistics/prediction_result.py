from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .contribution import Contribution


@dataclass(frozen=True)
class PredictionResult:
    key: TimeKey
    distribution: DistributionStats
    contributions: tuple[Contribution, ...]
