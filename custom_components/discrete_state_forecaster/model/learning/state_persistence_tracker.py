"""
State persistence tracking using exponentially weighted duration statistics.

This module provides StatePersistenceTracker, which tracks how long states
typically persist and computes persistence boosts based on hazard-style decay.
This helps predictions favor the current state when it's expected to persist.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Self

from .state_persistence_tracker_hyper_parameters import (
    StatePersistenceTrackerHyperParameters,
)

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )
    from custom_components.discrete_state_forecaster.model.state import (
        State,
    )


class StatePersistenceTracker:
    """
    Tracks state persistence and computes persistence boosts.

    Maintains exponentially weighted mean durations for each state and uses
    hazard-style decay to compute persistence boosts. The boost is highest
    when a state has just started and decays as the state duration approaches
    and exceeds its expected duration.

    Example:
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> hp = StatePersistenceTrackerHyperParameters(
        ...     hyper_parameters=base_hp,
        ...     persistence_half_life_factor=1.0,
        ... )
        >>> tracker = StatePersistenceTracker(hp)
        >>> tracker.update("on", 100.0)
        >>> tracker.current_state
        'on'
    """

    _hyper_parameters: Final[StatePersistenceTrackerHyperParameters]
    """Configuration controlling decay behavior."""

    _mean_duration: Final[dict[State, float]]
    """Exponentially weighted mean duration for each observed state."""

    _last_ts: float | None
    """Timestamp of last update, or None if never updated."""

    _current_state: State | None
    """Currently active state, or None if no state observed yet."""

    _current_state_start: float | None
    """Timestamp when current state started, or None if no state active."""

    def __init__(
        self: Self, hyper_parameters: StatePersistenceTrackerHyperParameters
    ) -> None:
        """
        Initialize state persistence tracker.

        Args:
            hyper_parameters: Configuration controlling decay behavior.
        """
        self._hyper_parameters: Final = hyper_parameters
        self._mean_duration: dict[State, float] = {}
        self._last_ts: float | None = None
        self._current_state: State | None = None
        self._current_state_start: float | None = None

    def update(self: Self, state: State, timestamp: float) -> None:
        """
        Updates tracker with new state observation.

        When state changes, records the duration of the previous state and
        updates its mean duration using exponential weighting. The first
        observation just sets the current state.

        Args:
            state: The observed state.
            timestamp: Current timestamp.
        """
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

    @property
    def current_state(self: Self) -> State | None:
        """
        Get the currently active state.

        Returns:
            The state that has been most recently observed, or None if
                no state has been observed yet.

        """
        return self._current_state

    def current_duration(self: Self, timestamp: float) -> float:
        """
        Get duration of current state.

        Args:
            timestamp: Current timestamp.

        Returns:
            Time elapsed since current state started, or 0.0 if no state active.

        """
        if self._current_state_start is None:
            return 0.0
        return max(0.0, timestamp - self._current_state_start)

    def expected_duration(self: Self, state: State, default: float = 60.0) -> float:
        """
        Get expected duration for a state.

        Args:
            state: The state to query.
            default: Default duration if state has not been observed (default 60.0).

        Returns:
            Exponentially weighted mean duration for the state.

        """
        return self._mean_duration.get(state, default)

    def persistence_boost(
        self: Self,
        state: State,
        timestamp: float,
        default_expected: float = 60.0,
    ) -> float:
        """
        Compute hazard-style persistence boost.

        Returns a multiplier in [0, 1] that decays as the current duration
        approaches and exceeds the expected duration. Only non-zero when
        the queried state matches the current state.

        Args:
            state: The state to compute boost for.
            timestamp: Current timestamp.
            default_expected: Default expected duration if state not observed.

        Returns:
            Persistence boost multiplier (0 if state != current_state,
                exp(-ratio) otherwise where ratio = current/expected).

        """
        if state != self._current_state:
            return 0.0

        expected = self.expected_duration(state, default_expected)
        current = self.current_duration(timestamp)

        # hazard-style decay
        ratio = current / max(expected, 1e-6)

        # exp(-ratio) → strong when fresh, weak when overstaying
        return math.exp(-ratio)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the instance into a dictionary.

        Returns:
            A dictionary representation of the instance, including hyper-parameters,
            mean durations for all observed states, last update timestamp,
            current state, and current state start time.
        """
        return {
            "hyper_parameters": self._hyper_parameters.to_dict(),
            "mean_duration": dict(self._mean_duration),
            "last_ts": self._last_ts,
            "current_state": self._current_state,
            "current_state_start": self._current_state_start,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> StatePersistenceTracker:
        """
        Deserialize an instance from a dictionary.

        Args:
            data: Dictionary containing serialized instance data including
                hyper_parameters, mean_duration, last_ts, current_state,
                and current_state_start.
            hyper_parameters: Base hyper-parameters needed to reconstruct the
                StatePersistenceTrackerHyperParameters.

        Returns:
            A new StatePersistenceTracker instance initialized from the provided
            data with all internal state restored.
        """
        tracker_hp = StatePersistenceTrackerHyperParameters.from_dict(
            data["hyper_parameters"], hyper_parameters
        )

        tracker = cls(hyper_parameters=tracker_hp)
        tracker._mean_duration = dict(data["mean_duration"])
        tracker._last_ts = data["last_ts"]
        tracker._current_state = data["current_state"]
        tracker._current_state_start = data["current_state_start"]

        return tracker
