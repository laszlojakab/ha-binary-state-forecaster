from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self

from custom_components.discrete_state_forecaster.model.statistics.state_stats import (
    StateStats,
)

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.state import State


class DistributionStats:
    def __init__(self: Self) -> None:
        self._states: dict[State, StateStats] = {}

    def update(self: Self, state: State, weight: float = 1.0) -> None:
        if state not in self._states:
            self._states[state] = StateStats()

        self._states[state].update(weight)

    def total_support(self: Self) -> float:
        return sum(stats.support() for stats in self._states.values())

    def support(self, state: State) -> float:
        stats = self._states.get(state)
        return 0.0 if stats is None else stats.support()

    def distribution(self: Self) -> dict[State, float]:
        total = self.total_support()
        if total <= 0.0:
            return {}

        return {state: stats.support() / total for state, stats in self._states.items()}

    def is_confident(self: Self, min_support: float) -> bool:
        return self.total_support() >= min_support

    def active_states(self: Self, min_support: float) -> set[State]:
        return {
            state
            for state, stats in self._states.items()
            if stats.is_active(min_support)
        }

    def entropy(self: Self) -> float:
        dist = self.distribution()
        if not dist:
            return 0.0

        return -sum(p * math.log(p) for p in dist.values() if p > 0.0)

    def max_probability(self: Self) -> float:
        dist = self.distribution()
        return max(dist.values(), default=0.0)

    def apply_decay(self: Self, factor: float) -> None:
        for stats in self._states.values():
            stats.apply_decay(factor)

    def states(self: Self) -> set[State]:
        return set(self._states.keys())

    def is_empty(self: Self) -> bool:
        return not self._states

    def prune(
        self: Self,
        min_state_duration: float,
    ) -> None:
        self._states = {
            s: d for s, d in self._states.items() if d.is_active(min_state_duration)
        }

    def prune_adaptive(
        self: Self,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
    ) -> None:
        threshold = max(self.total_support() * epsilon, absolute_min)
        self.prune(threshold)
