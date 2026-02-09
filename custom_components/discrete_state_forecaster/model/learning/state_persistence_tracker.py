import math
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.state import (
    State,
)

from .state_persistence_tracker_hyper_parameters import (
    StatePersistenceTrackerHyperParameters,
)


class StatePersistenceTracker:
    def __init__(
        self: Self, hyper_parameters: StatePersistenceTrackerHyperParameters
    ) -> None:
        self._hyper_parameters: Final = hyper_parameters
        self._mean_duration: dict[State, float] = {}
        self._last_ts: float | None = None
        self._current_state: State | None = None
        self._current_state_start: float | None = None

    def update(self: Self, state: State, timestamp: float) -> None:
        if self._current_state is None:
            self._current_state = state
            self._current_state_start = timestamp
            self._last_ts = timestamp
            return

        if state != self._current_state:
            duration = timestamp - self._current_state_start

            prev = self._mean_duration.get(self._current_state, duration)
            dt = timestamp - self._last_ts if self._last_ts else 0.0
            _lambda = math.log(2.0) / self._hyper_parameters.persistence_half_life
            decay = math.exp(-_lambda * dt)

            self._mean_duration[self._current_state] = (
                decay * prev + (1 - decay) * duration
            )

            self._current_state = state
            self._current_state_start = timestamp
            self._last_ts = timestamp

    def current_state(self: Self) -> State | None:
        return self._current_state

    def current_duration(self: Self, timestamp: float) -> float:
        if self._current_state_start is None:
            return 0.0
        return max(0.0, timestamp - self._current_state_start)

    def expected_duration(self: Self, state: State, default: float = 60.0) -> float:
        return self._mean_duration.get(state, default)

    def persistence_boost(
        self: Self,
        state: State,
        timestamp: float,
        default_expected: float = 60.0,
    ) -> float:
        """
        Hazard-style persistence boost.

        Returns multiplier in [0, 1].
        """
        if state != self._current_state:
            return 0.0

        expected = self.expected_duration(state, default_expected)
        current = self.current_duration(timestamp)

        # hazard-style decay
        ratio = current / max(expected, 1e-6)

        # exp(-ratio) → strong when fresh, weak when overstaying
        return math.exp(-ratio)
