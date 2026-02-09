import math
from typing import Final

from custom_components.discrete_state_forecaster.model.state import (
    State,
)


class BaselineDistribution:
    def __init__(
        self,
        half_life: float,
        epsilon: float = 1e-9,
        prune_threshold: float = 1e-12,
    ):
        self._states: dict[State, float] = {}
        self._epsilon: Final = epsilon
        self._prune_threshold: Final = prune_threshold
        self._last_ts: float | None = None
        self._lambda = math.log(2.0) / half_life

    def update(
        self,
        dist: dict[State, float],
        timestamp: float,
    ):
        if self._last_ts is None:
            # First state, we just copy it
            # and set the timestamp, no decay needed
            self._states = dict(dist)
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        decay = math.exp(-self._lambda * dt)
        # We apply the decay to all existing states, and prune those that are too small
        for k in list(self._states.keys()):
            self._states[k] *= decay
            if self._states[k] < self._prune_threshold:
                del self._states[k]

        # Then we mix in the new distribution with the decayed one, using a simple weighted average
        mix = 1.0 - decay
        for k, p in dist.items():
            if p > 0:
                self._states[k] = self._states.get(k, 0.0) + mix * p

        self._last_ts = timestamp

    def distribution(self) -> dict[State, float]:
        total = sum(self._states.values())
        if total <= 0:
            return {}

        K = len(self._states)
        denom = total + self._epsilon * K

        return {k: (v + self._epsilon) / denom for k, v in self._states.items()}
