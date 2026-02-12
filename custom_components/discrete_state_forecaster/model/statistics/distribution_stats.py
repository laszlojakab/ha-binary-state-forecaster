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
from typing import TYPE_CHECKING, Any, Self

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

    Attributes:
        _states: Dictionary mapping each state to its associated StateStats,
            tracking cumulative support weight for that state.

    Example:
        >>> dist = DistributionStats()
        >>> dist.update("on", weight=2.0)
        >>> dist.update("off", weight=1.0)
        >>> dist.distribution()
        {'on': 0.6667, 'off': 0.3333}
        >>> dist.entropy()  # doctest: +SKIP
        0.637

    """

    def __init__(self: Self, states: dict[State, StateStats] | None = None) -> None:
        """
        Initializes a distribution.

        Args:
          states: Optional initial state statistics. If None, starts with an empty distribution.
        """
        self._states: dict[State, StateStats] = {} if states is None else states

    def update(self: Self, state: State, weight: float = 1.0) -> None:
        """
        Updates support for a state by adding the given weight.

        Creates a new StateStats entry if this state has not been observed
        before. If the state already exists, adds the weight to its existing
        support.

        Args:
            state: The state to update.
            weight: The weight (support) to add. Defaults to 1.0. Must be
                positive or zero.

        """
        if state not in self._states:
            self._states[state] = StateStats()

        self._states[state].update(weight)

    def total_support(self: Self) -> float:
        """
        Calculates the total support across all states.

        Returns:
            The sum of support values for all states in the distribution.
                Returns 0.0 if no states have been observed.

        """
        return sum(stats.support() for stats in self._states.values())

    def support(self, state: State) -> float:
        """
        Gets the support for a specific state.

        Args:
            state: The state to query.

        Returns:
            The support value for the state. Returns 0.0 if the state has
                not been observed.

        """
        stats = self._states.get(state)
        return 0.0 if stats is None else stats.support()

    def distribution(self: Self) -> dict[State, float]:
        """
        Calculates probability distribution by normalizing support values.

        The probability for each state is its support divided by the total
        support across all states.

        Returns:
            Dictionary mapping each observed state to its probability in the
                range [0.0, 1.0]. Returns empty dict if no observations exist
                or total support is zero or negative.

        """
        total = self.total_support()
        if total <= 0.0:
            return {}

        return {state: stats.support() / total for state, stats in self._states.items()}

    def is_confident(self: Self, min_support: float) -> bool:
        """
        Checks if distribution has sufficient total support.

        Args:
            min_support: The minimum total support threshold.

        Returns:
            True if total_support() >= min_support, False otherwise.

        """
        return self.total_support() >= min_support

    def active_states(self: Self, min_support: float) -> set[State]:
        """
        Gets states that meet or exceed the minimum support threshold.

        Args:
            min_support: The minimum support threshold for a state to be
                considered active.

        Returns:
            Set of states where individual support >= min_support. Returns
                empty set if no states meet the threshold.

        """
        return {
            state
            for state, stats in self._states.items()
            if stats.is_active(min_support)
        }

    def entropy(self: Self) -> float:
        """
        Calculates Shannon entropy of the probability distribution.

        Entropy measures the uncertainty in the distribution. Higher entropy
        indicates more uniform distribution (less predictability), while lower
        entropy indicates concentration on fewer states (more predictability).

        Returns:
            Non-negative entropy value in nats. Returns 0.0 if distribution is
                empty or only one state has non-zero probability.

        """
        dist = self.distribution()
        if not dist:
            return 0.0

        return -sum(p * math.log(p) for p in dist.values() if p > 0.0)

    def max_probability(self: Self) -> float:
        """
        Gets the maximum probability in the distribution.

        Returns:
            The highest probability value across all states. Returns 0.0 if
                no states have been observed.

        """
        dist = self.distribution()
        return max(dist.values(), default=0.0)

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

    def states(self: Self) -> set[State]:
        """
        Gets all observed states in the distribution.

        Returns:
            Set of all states that have been updated at least once.

        """
        return set(self._states.keys())

    def is_empty(self: Self) -> bool:
        """
        Checks if the distribution has no observed states.

        Returns:
            True if no states have been updated yet, False otherwise.

        """
        return not self._states

    def prune(
        self: Self,
        min_state_duration: float,
    ) -> None:
        """
        Removes states with support below the minimum threshold.

        Uses is_active() check on each state to determine which to keep,
        allowing removal of infrequently observed states.

        Args:
            min_state_duration: The minimum support threshold. States with
                support < min_state_duration are removed.

        """
        self._states = {
            s: d for s, d in self._states.items() if d.is_active(min_state_duration)
        }

    def prune_adaptive(
        self: Self,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
    ) -> None:
        """
        Adaptively prunes states using relative and absolute thresholds.

        Removes states using a dynamic threshold that is the maximum of:
        - Relative threshold: epsilon * total_support()
        - Absolute threshold: absolute_min

        This allows the algorithm to automatically adjust pruning aggressiveness
        based on total accumulated support while maintaining a floor.

        Args:
            epsilon: Relative threshold factor (default 0.003). States with
                support < epsilon * total_support are candidates for removal.
            absolute_min: Absolute minimum support (default 20.0). Ensures
                pruning doesn't remove frequently observed states even if total
                support is very high.

        """
        threshold = max(self.total_support() * epsilon, absolute_min)
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
        return cls(
            states={
                s: StateStats.from_dict(stats)
                for s, stats in data.get("states", {}).items()
            }
        )
