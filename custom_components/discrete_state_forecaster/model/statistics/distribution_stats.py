"""
Statistical distributions over discrete states with support-based weighting.

This module provides `DistributionStats`, a class that manages a distribution
of states and their associated statistics. It aggregates support (weight) across
multiple states, calculates probability distributions, measures entropy, and
supports temporal decay and pruning operations.

The class uses `StateStats` internally to track support for individual states,
enabling features like decay weighting and minimum support thresholds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Self, cast

from custom_components.discrete_state_forecaster.model.statistics.state_stats import (
    StateStats,
)

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.state import State


class DistributionStats:
    """
    Distribution statistics over discrete states with support aggregation.

    Manages a collection of states and their individual statistics, providing
    aggregated views like probability distributions and entropy. Supports temporal
    operations like decay weighting and prune operations.

    Example:
        >>> dist = DistributionStats()
        >>> dist.update("on", weight=2.0)
        >>> dist.update("off", weight=1.0)
        >>> dist.distribution
        {'on': 0.6667, 'off': 0.3333}
        >>> dist.entropy
        0.637
        >>> dist.max_probability
        0.6667
    """

    _states: Final[dict[State, StateStats]] = {}
    """Dictionary mapping each state to its cumulative support statistics."""

    def __init__(self: Self) -> None:
        """Initializes a new instance of `DistributionStats` class."""
        self._states: dict[State, StateStats] = {}

    @property
    def total_support(self: Self) -> float:
        """
        Calculates the total support across all states.

        Returns:
            The sum of support values for all states in the distribution.
                Returns 0.0 if no states have been observed.

        """
        return sum(stats.support for stats in self._states.values())

    @property
    def distribution(self: Self) -> dict[State, float]:
        """
        Calculates probability distribution by normalizing support values.

        The probability for each state is its support divided by the total
        support across all states.

        Returns:
            Dictionary mapping each observed state to its probability in the
                range [0.0, 1.0]. Returns empty dict if no observations exist
                or total support is zero.

        """
        total = self.total_support
        if total == 0.0:
            return {}

        return {state: stats.support / total for state, stats in self._states.items()}

    @property
    def entropy(self: Self) -> float:
        """
        Calculates Shannon entropy of the probability distribution.

        Entropy measures the uncertainty in the distribution. Higher entropy
        indicates more uniform distribution (less predictability), while lower
        entropy indicates concentration on fewer states (more predictability).

        Returns:
            Non-negative entropy value in nats. Returns 0.0 if distribution is empty.

        """
        dist = self.distribution
        if not dist:
            return 0.0

        return -sum(p * math.log(p) for p in dist.values() if p > 0.0)

    @property
    def max_probability(self: Self) -> float:
        """
        Gets the maximum probability in the distribution.

        Returns:
            The highest probability value across all states. Returns 0.0 if
                no states have been observed.

        """
        return max(self.distribution.values(), default=0.0)

    @property
    def is_empty(self: Self) -> bool:
        """
        Checks if the distribution has no observed states.

        Returns:
            True if no states have been updated yet, False otherwise.

        """
        return not self._states

    def is_confident(self: Self, min_support: float) -> bool:
        """
        Checks if distribution has sufficient total support.

        Args:
            min_support: The minimum total support threshold.

        Returns:
            True if total_support >= min_support, False otherwise.

        """
        return self.total_support >= min_support

    def update(self: Self, state: State, weight: float) -> None:
        """
        Updates support for a state by adding the given weight.

        Creates a new StateStats entry if this state has not been observed
        before. If the state already exists, adds the weight to its existing
        support.

        Args:
            state: The state to update.
            weight: The weight (support) to add. Must be non-negative.
        """
        if state not in self._states:
            self._states[state] = StateStats()

        self._states[state].update(weight)

    def apply_decay(self: Self, factor: float) -> None:
        """
        Applies exponential decay to all state support values.

        Multiplies the support of each state by the decay factor, giving
        recent observations more weight than older ones.

        Args:
            factor: Decay factor in range (0, 1]. Values closer to 0 produce
                stronger decay. For example, 0.95 retains 95% of previous
                support.

        """
        for stats in self._states.values():
            stats.apply_decay(factor)

    def prune(
        self: Self,
        min_support: float,
    ) -> None:
        """
        Removes states with support below the minimum threshold.

        Uses is_active() check on each state to determine which to keep,
        allowing removal of infrequently observed states.

        Args:
            min_support: The minimum support threshold. States with
                support < min_support are removed.

        """
        for key in [
            state
            for state, stats in self._states.items()
            if not stats.is_active(min_support)
        ]:
            del self._states[key]

    def prune_adaptive(
        self: Self,
        epsilon: float = 0.003,
        absolute_minimum_support: float = 20.0,
    ) -> None:
        """
        Adaptively prunes states using relative and absolute thresholds.

        Removes states using a dynamic threshold that is the maximum of:
        - Relative threshold: epsilon * total_support
        - Absolute threshold: absolute_minimum_support

        This allows the algorithm to automatically adjust pruning aggressiveness
        based on total accumulated support while maintaining a floor.

        Args:
            epsilon: Relative threshold factor (default 0.003). States with
                support < epsilon * total_support are candidates for removal.
            absolute_minimum_support: Absolute minimum support (default 20.0).
                Ensures pruning doesn't remove frequently observed states even
                if total support is very high.

        """
        threshold = max(self.total_support * epsilon, absolute_minimum_support)

        self.prune(threshold)

    def to_dict(self: Self) -> dict[str, Any]:
        """Returns a JSON-serializable representation of this DistributionStats."""
        return {
            "states": {s: stats.to_dict() for s, stats in self._states.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributionStats:
        """
        Reconstructs DistributionStats from a dict representation.

        Args:
          data: Dictionary containing distribution statistics.

        Returns:
          A new DistributionStats instance with states initialized from data.
        """
        stats = cls()
        stats._states = {
            s: StateStats.from_dict(stats_data)
            for s, stats_data in cast(
                "dict[str, dict[str, Any]]", data["states"]
            ).items()
        }

        return stats
