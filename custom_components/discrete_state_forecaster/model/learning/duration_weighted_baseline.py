import math
from typing import Final

from custom_components.discrete_state_forecaster.model.state import (
    State,
)

from .duration_weighted_baseline_hyper_parameters import (
    DurationWeightedBaselineHyperParameters,
)


class DurationWeightedBaseline:
    """
    Baseline distribution that integrates distributions over time.

    Each update represents a distribution that was valid for dt seconds.
    Older evidence decays exponentially based on half-life.
    """

    def __init__(
        self,
        hyper_parameters: DurationWeightedBaselineHyperParameters,
    ):
        self._hyper_parameters: Final = hyper_parameters

        self._mass: dict[State, float] = {}
        self._last_ts: float | None = None

    def update(
        self,
        dist: dict[State, float],
        timestamp: float,
    ) -> None:
        if self._last_ts is None:
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        # 1. decay old mass
        lambda_ = math.log(2.0) / (self._hyper_parameters.baseline_half_life)
        decay = math.exp(-lambda_ * dt)
        for s in list(self._mass.keys()):
            self._mass[s] *= decay
            if self._mass[s] < self._hyper_parameters.prune_threshold:
                del self._mass[s]

        # 2. integrate new distribution over dt
        for s, p in dist.items():
            if p > 0.0:
                self._mass[s] = self._mass.get(s, 0.0) + p * dt

        self._last_ts = timestamp

    def distribution(self) -> dict[State, float]:
        total = sum(self._mass.values())
        if total <= 0.0:
            return {}

        K = len(self._mass)
        denom = total + self._hyper_parameters.epsilon * K

        return {
            s: (m + self._hyper_parameters.epsilon) / denom
            for s, m in self._mass.items()
        }

    def total_mass(self) -> float:
        return sum(self._mass.values())
