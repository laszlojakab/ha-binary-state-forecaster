from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)


@dataclass(frozen=True)
class Contribution:
    key: TimeKey
    weight: float
    support: float
