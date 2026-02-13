"""
Duration-weighted baseline distribution with exponential decay.

This module provides DurationWeightedBaseline, which maintains a probability
distribution by integrating distribution observations over time. Each update
represents a distribution that was valid for dt seconds, and older evidence
decays exponentially.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
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
    Older evidence decays exponentially based on half-life. This allows
    the baseline to adapt to changing patterns while giving appropriate
    weight to how long each distribution persisted.

    Example:
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
        >>> hp = DurationWeightedBaselineHyperParameters(
        ...     hyper_parameters=drift_hp,
        ...     half_life_factor=1.0,
        ... )
        >>> baseline = DurationWeightedBaseline(hp)
        >>> dist = {"on": 0.7, "off": 0.3}
        >>> baseline.update(dist, 100.0)
        >>> baseline.update(dist, 105.0)
        >>> result = baseline.distribution()
        >>> abs(result["on"] - 0.7) < 0.01
        True
    """

    _hyper_parameters: Final[DurationWeightedBaselineHyperParameters]
    """Configuration controlling decay and smoothing."""

    _mass: Final[dict[State, float]]
    """Accumulated mass for each state."""

    _last_ts: float | None
    """Timestamp of last update, or None if never updated."""

    def __init__(
        self: Self,
        hyper_parameters: DurationWeightedBaselineHyperParameters,
    ) -> None:
        """
        Initialize duration-weighted baseline tracker.

        Args:
            hyper_parameters: Configuration controlling decay and smoothing.

        """
        self._hyper_parameters = hyper_parameters
        self._mass = {}
        self._last_ts = None

    def update(
        self: Self,
        dist: dict[State, float],
        timestamp: float,
    ) -> None:
        """
        Update baseline with new distribution observation.

        Applies exponential decay to existing mass, prunes small values,
        then integrates the new distribution weighted by time elapsed.
        The first update just sets the timestamp without accumulating mass.

        Args:
            dist: Probability distribution to integrate (should sum to ~1.0).
            timestamp: Current timestamp for computing decay.

        """
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

    def distribution(self: Self) -> dict[State, float]:
        """
        Get normalized probability distribution with Laplace smoothing.

        Returns:
            Dictionary mapping states to probabilities (sums to ~1.0),
                or empty dict if no mass has been accumulated.

        """
        total = sum(self._mass.values())
        if total <= 0.0:
            return {}

        num_states = len(self._mass)
        denom = total + self._hyper_parameters.epsilon * num_states

        return {
            s: (m + self._hyper_parameters.epsilon) / denom
            for s, m in self._mass.items()
        }

    def total_mass(self: Self) -> float:
        """
        Get total accumulated mass across all states.

        Returns:
            Sum of all state masses before normalization.

        """
        return sum(self._mass.values())

    def to_dict(self: Self) -> dict:
        """
        Serialize baseline state to a dictionary.

        Returns:
            Dictionary containing hyper-parameters and current mass state.

        """
        return {
            "hyper_parameters": self._hyper_parameters.to_dict(),
            "mass": dict(self._mass),
            "last_ts": self._last_ts,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        hyper_parameters: DurationWeightedBaselineHyperParameters,
    ) -> DurationWeightedBaseline:
        """
        Deserialize baseline state from a dictionary.

        Args:
            data: Dictionary containing serialized state and hyper-parameters.
            hyper_parameters: Hyper-parameters to use for the instance.

        Returns:
            A DurationWeightedBaseline instance initialized with the provided data.
        """
        instance = cls(hyper_parameters=hyper_parameters)
        instance._mass = dict(data["mass"])
        instance._last_ts = data["last_ts"]
        return instance
