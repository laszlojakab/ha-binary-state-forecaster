from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.state import State


@dataclass
class AggregatedStats:
    distribution: dict[State, float]
    support_time: float
    depth: int