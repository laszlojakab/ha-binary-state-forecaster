"""
Exponentially weighted baseline probability distribution.

This module provides BaselineDistribution, which maintains a probability distribution
over states that adapts to changes using exponential decay. Old observations are
gradually forgotten, allowing the baseline to track shifting patterns over time.
"""

import math
from typing import Final

from custom_components.discrete_state_forecaster.model.state import (
    State,
)
from custom_components.discrete_state_forecaster.model.utils import (
    deserialize_state,
    serialize_state,
)


class BaselineDistribution:
    """
    Maintains an exponentially weighted probability distribution over states.

    Tracks state probabilities using exponential decay, where older observations
    have less weight than recent ones. Applies Laplace smoothing to avoid zero
    probabilities and automatically prunes near-zero probability states.

    Attributes:
        _states: Dictionary mapping states to their accumulated weights.
        _epsilon: Laplace smoothing parameter to avoid zero probabilities.
        _prune_threshold: States with weight below this are removed.
        _last_ts: Timestamp of last update, or None if never updated.
        _lambda: Decay rate computed from half-life.

    Example:
        >>> baseline = BaselineDistribution(half_life=50.0)
        >>> dist = {"on": 0.7, "off": 0.3}
        >>> baseline.update(dist, 100.0)
        >>> result = baseline.distribution()
        >>> abs(result["on"] - 0.7) < 0.01
        True

    """

    def __init__(
        self,
        half_life: float,
        epsilon: float = 1e-9,
        prune_threshold: float = 1e-12,
    ):
        """
        Initializes the baseline distribution tracker.

        Args:
            half_life: Time after which old observations have half their weight.
            epsilon: Laplace smoothing parameter (default 1e-9).
            prune_threshold: Minimum weight to retain a state (default 1e-12).

        """
        self._states: dict[State, float] = {}
        self._epsilon: Final = epsilon
        self._prune_threshold: Final = prune_threshold
        self._last_ts: float | None = None
        self._half_life: Final = half_life
        self._lambda = math.log(2.0) / half_life

    # State serialization/deserialization delegated to model.utils

    def to_dict(self) -> dict:
        """
        Returns a JSON-serializable representation of this BaselineDistribution.

        The representation includes half-life, epsilon, prune_threshold, last
        timestamp, and a list of serialized (state, weight) pairs.
        """
        states_list = []
        for s, w in self._states.items():
            states_list.append({"state": serialize_state(s), "weight": w})

        return {
            "half_life": self._half_life,
            "epsilon": float(self._epsilon),
            "prune_threshold": float(self._prune_threshold),
            "last_ts": self._last_ts,
            "states": states_list,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineDistribution":
        """
        Reconstructs a BaselineDistribution from its JSON-serializable dict.

        Note: states serialized with unknown types (repr fallback) may not be
        reconstructed to their original Python objects.
        """
        half_life = float(data.get("half_life"))
        epsilon = float(data.get("epsilon", 1e-9))
        prune_threshold = float(data.get("prune_threshold", 1e-12))

        inst = cls(
            half_life=half_life, epsilon=epsilon, prune_threshold=prune_threshold
        )

        inst._last_ts = data.get("last_ts")

        states = data.get("states", []) or []
        for item in states:
            s = deserialize_state(item["state"]) if isinstance(item, dict) else None
            w = float(item.get("weight", 0.0)) if isinstance(item, dict) else 0.0
            if s is not None:
                inst._states[s] = w

        return inst

    def update(
        self,
        dist: dict[State, float],
        timestamp: float,
    ) -> None:
        """
        Updates baseline with new probability distribution.

        Applies exponential decay to existing state weights, prunes near-zero
        weights, then mixes in the new distribution.

        Args:
            dist: New probability distribution (should sum to ~1.0).
            timestamp: Current timestamp for computing decay.

        """
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
        """
        Gets normalized probability distribution with Laplace smoothing.

        Returns:
            Dictionary mapping states to probabilities (sums to 1.0), or empty
                dict if no states have been observed.

        """
        total = sum(self._states.values())
        if total <= 0:
            return {}

        num_states = len(self._states)
        denom = total + self._epsilon * num_states

        return {k: (v + self._epsilon) / denom for k, v in self._states.items()}
