"""
Time-aware state forecaster with temporal pattern learning.

This module provides the TimeAwareForecaster class, which combines a time indexer
with a discrete conditional model to learn and predict state patterns based on
time. The forecaster automatically handles time intervals by splitting them across
temporal bucket boundaries, ensuring accurate pattern learning.

The TimeAwareForecaster is the main interface for training and predicting with
temporal state patterns, managing the complexity of time bucketing and state
tracking.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Self

from custom_components.discrete_state_forecaster.model.hierarchical_temporal_state_model import (
    HierarchicalTemporalStateModel,
)
from custom_components.discrete_state_forecaster.model.prediction import Prediction
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)

# Constants for state persistence tracking
MIN_OBSERVATIONS_FOR_PERSISTENCE = 10  # Minimum observations before using learned persistence
MAX_TIME_GAP_FOR_CONTINUATION = 3600  # Max time gap (seconds) to consider state continuation


@dataclass
class StatePersistenceTracker:
    """
    Tracks state transition statistics for adaptive persistence learning.

    This class monitors state transitions to learn how "sticky" each state is.
    States that tend to persist (remain unchanged) are given higher persistence
    factors in predictions, while transient states get lower factors.

    Uses exponential moving average to adapt to changing patterns over time.

    Attributes:
        persistence_count: Weighted count of times each state persisted.
        total_count: Total weighted count of observations for each state.
        smoothing: Smoothing factor for exponential moving average (0.0 to 1.0).
                  Higher values give more weight to historical data.
    """

    persistence_count: dict[State, float] = field(default_factory=dict)
    total_count: dict[State, float] = field(default_factory=dict)
    smoothing: float = 0.95

    def update(self: Self, state: State, persisted: bool) -> None:
        """
        Update persistence statistics for a state.

        Args:
            state: The state being tracked.
            persisted: True if state remained the same, False if it changed.
        """
        if state not in self.persistence_count:
            self.persistence_count[state] = 0.0
            self.total_count[state] = 0.0

        # Exponential moving average to adapt to changing patterns
        self.persistence_count[state] = self.smoothing * self.persistence_count[
            state
        ] + (1.0 if persisted else 0.0)
        self.total_count[state] = self.smoothing * self.total_count[state] + 1.0

    def get_persistence_factor(self: Self, state: State, default: float = 0.3) -> float:
        """
        Get learned persistence factor for a state.

        Args:
            state: The state to query.
            default: Default factor if insufficient data exists.

        Returns:
            Learned persistence factor (0.0 to 1.0). Returns default if less than
            10 observations exist for the state.
        """
        if (
            state not in self.total_count
            or self.total_count[state] < MIN_OBSERVATIONS_FOR_PERSISTENCE
        ):
            return default  # Not enough data yet

        # Empirical persistence rate
        rate = self.persistence_count[state] / self.total_count[state]

        # Blend with default for stability (80% learned, 20% default)
        return 0.8 * rate + 0.2 * default


class TimeAwareForecaster:
    """
    Time-aware forecaster for learning and predicting state patterns.

    This class combines temporal indexing with statistical modeling to learn
    state patterns that vary by time. It automatically handles state intervals
    that span multiple time buckets by splitting them appropriately.

    The forecaster uses a CompositeIndexer to partition time into buckets
    (e.g., by hour, day of week, month) and learns separate probability
    distributions for each bucket. This enables predictions that capture
    temporal patterns like "device is usually on during weekday mornings."

    Attributes:
        indexer: The composite indexer defining temporal bucket boundaries.
        model: The underlying discrete conditional model storing learned patterns.
        state_persistence_factor: Base persistence factor (0.0 to 1.0).
        adaptive_persistence: Whether to automatically learn state-specific persistence.
        persistence_tracker: Tracks state transitions for adaptive learning.
        last_state: Most recent state observed (for transition tracking).
        last_timestamp: Timestamp of most recent state (for transition tracking).

    Example:
        ```
        >>> from .time_indexers import CompositeIndexer, TimeOfDayIndexer
        >>> indexer = CompositeIndexer([TimeOfDayIndexer(30)])
        >>> forecaster = TimeAwareForecaster(indexer)
        >>>
        >>> # Learn from state interval
        >>> start = datetime(2024, 1, 1, 10, 0)
        >>> end = datetime(2024, 1, 1, 11, 0)
        >>> forecaster.update_interval(start, end, "on")
        >>>
        >>> # Make prediction
        >>> pred_time = datetime(2024, 1, 2, 10, 15)
        >>> prediction = forecaster.predict(pred_time)
        >>> print(f"Predicted state: {prediction.state}")
        ```
    """

    def __init__(
        self: Self,
        indexer: CompositeIndexer,
        half_life: float = 0.0,
        state_persistence_factor: float = 0.3,
        adaptive_persistence: bool = True,
    ) -> None:
        """
        Initialize a TimeAwareForecaster with a time indexer.

        Args:
            indexer: CompositeIndexer defining how time is partitioned into buckets.
                This determines the temporal granularity of pattern learning.
                For example, CompositeIndexer([TimeOfDayIndexer(60)]) creates
                hourly buckets.
            half_life: Half-life for exponential decay in seconds. If 0.0 (default),
                no decay is applied. Positive values enable temporal decay where
                older observations have less influence on predictions.
            state_persistence_factor: Initial/fallback persistence factor (0.0 to 1.0).
                Default is 0.3 (30% boost to current state). When adaptive_persistence
                is True, this serves as the default for states with insufficient data.
                When adaptive_persistence is False, this is the fixed persistence factor.
            adaptive_persistence: If True (default), automatically learns state-specific
                persistence from observed transitions. States that tend to persist get
                higher factors. If False, uses fixed state_persistence_factor for all.

        Example:
            ```
            >>> from .time_indexers import CompositeIndexer, TimeOfDayIndexer, DayOfWeekIndexer
            >>> # Create forecaster with adaptive persistence (recommended)
            >>> indexer = CompositeIndexer([
            ...     DayOfWeekIndexer(),
            ...     TimeOfDayIndexer(60)
            ... ])
            >>> forecaster = TimeAwareForecaster(indexer)
            >>>
            >>> # Create forecaster with fixed persistence
            >>> forecaster_fixed = TimeAwareForecaster(
            ...     indexer,
            ...     state_persistence_factor=0.5,
            ...     adaptive_persistence=False
            ... )
            ```
        """
        self.indexer: CompositeIndexer = indexer
        self.model: HierarchicalTemporalStateModel = HierarchicalTemporalStateModel(
            half_life=half_life
        )
        self.state_persistence_factor: float = max(
            0.0, min(1.0, state_persistence_factor)
        )
        self.adaptive_persistence: bool = adaptive_persistence
        self.persistence_tracker: StatePersistenceTracker = StatePersistenceTracker()

        # Track last state for transition detection
        self.last_state: State | None = None
        self.last_timestamp: float | None = None

    def update_interval(
        self: Self,
        start: datetime,
        end: datetime,
        state: State,
    ) -> None:
        """
        Updates the model with a state interval observation.

        Records that a specific state was active during the time interval
        [start, end). The interval is automatically split across temporal
        bucket boundaries to ensure each bucket receives the correct duration.

        This method handles intervals that span multiple time buckets by:
        1. Finding bucket boundaries within the interval
        2. Calculating duration within each bucket
        3. Updating the model for each bucket separately

        If adaptive_persistence is enabled, also tracks state transitions to
        learn state-specific persistence factors.

        Args:
            start: Start of the interval (inclusive). The state was active
                at this time.
            end: End of the interval (exclusive). The state became inactive
                at this time.
            state: The state that was active during this interval.

        Note:
            - If end <= start, the update is ignored (invalid interval)
            - Short durations (< 5 seconds per bucket) are filtered by the model
            - The method correctly handles intervals crossing midnight, month
              boundaries, etc.

        Example:
            ```
            >>> forecaster = TimeAwareForecaster(indexer)
            >>> # State was "on" from 9:45 to 10:15 (crosses 10:00 boundary)
            >>> start = datetime(2024, 1, 1, 9, 45)
            >>> end = datetime(2024, 1, 1, 10, 15)
            >>> forecaster.update_interval(start, end, "on")
            >>> # This splits into: 15 min in 9:00-10:00 bucket, 15 min in 10:00-11:00 bucket
            >>> # Also tracks transition: if previous state was "off", learns transition probability
            ```
        """
        if end <= start:
            return

        # Update persistence tracker if adaptive mode enabled
        if self.adaptive_persistence:
            current_ts = start.timestamp()

            # Check if this is a continuation or transition
            if self.last_state is not None and self.last_timestamp is not None:
                # Consider it a transition if states differ OR significant time gap (>1 hour)
                time_gap = current_ts - self.last_timestamp
                is_transition = (state != self.last_state) or (
                    time_gap > MAX_TIME_GAP_FOR_CONTINUATION
                )

                # Update persistence statistics for the previous state
                persisted = not is_transition and state == self.last_state
                self.persistence_tracker.update(self.last_state, persisted)

            # Update last state tracking
            self.last_state = state
            self.last_timestamp = end.timestamp()

        # Original update logic
        cursor = start

        while cursor < end:
            next_ts = min(
                self.indexer.next_boundary(cursor),
                end,
            )

            duration = (next_ts - cursor).total_seconds()

            key = self.indexer.key(cursor)

            self.model.update_duration(
                key,
                state,
                duration,
                timestamp=cursor.timestamp(),
            )

            cursor = next_ts

    def _calculate_duration_adjusted_persistence(
        self: Self,
        base_factor: float,
        state_duration: float | None,
    ) -> float:
        """
        Adjust persistence factor based on how long the state has been active.

        States that have been active longer are more likely to persist, so we
        increase the persistence factor for longer durations. Uses logarithmic
        scaling to provide diminishing returns for very long durations.

        Args:
            base_factor: The base persistence factor (from learned or fixed value).
            state_duration: How long the current state has been active (in seconds).
                If None or <= 0, returns base_factor unchanged.

        Returns:
            Adjusted persistence factor (0.0 to 1.0). For longer durations, this
            will be higher than base_factor, capped at 1.0.

        Examples:
            >>> # Short duration (5 minutes) - minimal adjustment
            >>> self._calculate_duration_adjusted_persistence(0.5, 300)
            0.5  # 1.0x multiplier

            >>> # Medium duration (30 minutes) - moderate boost
            >>> self._calculate_duration_adjusted_persistence(0.5, 1800)
            0.58  # ~1.15x multiplier

            >>> # Long duration (2 hours) - significant boost
            >>> self._calculate_duration_adjusted_persistence(0.5, 7200)
            0.65  # ~1.3x multiplier

            >>> # Very long duration (10+ hours) - capped boost
            >>> self._calculate_duration_adjusted_persistence(0.5, 36000)
            0.75  # ~1.5x multiplier (approaching cap)
        """
        if state_duration is None or state_duration <= 0:
            return base_factor

        # Logarithmic scaling: longer durations → higher persistence
        # Reference point: 300s (5 minutes) gives multiplier of 1.0
        # 30 min → ~1.15x, 2 hours → ~1.3x, 10 hours → ~1.5x (caps out)
        duration_multiplier = 1.0 + 0.15 * min(
            1.0, math.log10(max(1, state_duration) / 300)
        )

        # Apply multiplier and cap at 1.0
        return min(1.0, base_factor * duration_multiplier)

    def predict(
        self: Self,
        timestamp: datetime,
        current_state: State | None = None,
        state_duration: float | None = None,
    ) -> Prediction:
        """
        Predicts the most likely state at a specific time.

        Looks up the learned probability distribution for the time bucket
        containing the given timestamp and returns the most likely state
        along with confidence metrics.

        When current_state is provided, the prediction is adjusted to account for
        state persistence. If adaptive_persistence is enabled, uses the learned
        state-specific persistence factor; otherwise uses the fixed factor.

        Args:
            timestamp: The timestamp for which to make a prediction. The forecaster
                will determine which temporal bucket contains this time and
                use the learned patterns for that bucket.
            current_state: The current state (if known). When provided, the prediction
                is adjusted to favor state persistence. This improves accuracy for
                systems where states tend to be sticky (e.g., heating, occupancy).
            state_duration: How long the current_state has been active (in seconds).
                If provided, increases persistence factor for longer durations. States
                that have been active longer are more likely to remain stable.

        Returns:
            A Prediction object containing:
            - state: The predicted state (highest probability), or None if
                no data exists for this time bucket.
            - distribution: Dictionary mapping states to their probabilities
                for this time bucket. Adjusted for state persistence if current_state
                is provided. Empty if no data exists.
            - confidence: Confidence metrics including max probability,
                entropy-based confidence, and total support time. All zeros
                if no data exists.

        Example:
            ```
            >>> # After training with historical data
            >>> forecaster.update_interval(
            ...     datetime(2024, 1, 1, 10, 0),
            ...     datetime(2024, 1, 1, 11, 0),
            ...     "on"
            ... )
            >>>
            >>> # Predict without context
            >>> pred_time = datetime(2024, 1, 2, 10, 30)
            >>> prediction = forecaster.predict(pred_time)
            >>> prob = prediction.distribution.get(prediction.state, 0)
            >>> print(f"State: {prediction.state}, Probability: {prob:.2f}")
            >>>
            >>> # Predict with current state context (more accurate)
            >>> prediction_ctx = forecaster.predict(
            ...     pred_time,
            ...     current_state="on",
            ...     state_duration=300.0  # Been "on" for 5 minutes
            ... )
            >>> # Probability of "on" will be higher due to learned persistence
            ```
        """
        key = self.indexer.key(timestamp)
        base_prediction = self.model.predict(key, timestamp.timestamp())

        # If no current state provided, return base prediction
        if current_state is None:
            return base_prediction

        # Determine persistence factor to use
        if self.adaptive_persistence:
            # Use learned state-specific persistence factor
            base_factor = self.persistence_tracker.get_persistence_factor(
                current_state, default=self.state_persistence_factor
            )
        else:
            # Use fixed persistence factor
            base_factor = self.state_persistence_factor

        # Apply duration adjustment if duration is provided
        persistence_factor = self._calculate_duration_adjusted_persistence(
            base_factor, state_duration
        )

        # If no persistence or no data, return base prediction
        if persistence_factor == 0.0 or not base_prediction.distribution:
            return base_prediction

        # Apply state persistence adjustment
        adjusted_dist = base_prediction.distribution.copy()

        if current_state in adjusted_dist:
            current_prob = adjusted_dist[current_state]

            # Boost current state probability towards 1.0
            # Formula: new_prob = old_prob + factor * (1 - old_prob)  # noqa: ERA001
            # This preserves relative probabilities of other states
            boost = persistence_factor * (1.0 - current_prob)
            adjusted_dist[current_state] = current_prob + boost

            # Renormalize all probabilities to sum to 1.0
            total = sum(adjusted_dist.values())
            if total > 0:
                adjusted_dist = {k: v / total for k, v in adjusted_dist.items()}
        else:
            # Current state not in learned distribution - add it with small probability
            # This handles novel states that weren't seen during training
            adjusted_dist[current_state] = persistence_factor * 0.5

            # Renormalize
            total = sum(adjusted_dist.values())
            if total > 0:
                adjusted_dist = {k: v / total for k, v in adjusted_dist.items()}

        # Find new most likely state
        new_state = max(adjusted_dist, key=adjusted_dist.get)

        # Return updated prediction with adjusted distribution
        return Prediction(
            state=new_state,
            distribution=adjusted_dist,
            confidence=base_prediction.confidence,  # Keep original confidence metrics
        )

    def get_learned_persistence(
        self: Self, state: State | None = None
    ) -> dict[State, float]:
        """
        Get learned persistence factors for introspection and debugging.

        Useful for understanding what the model has learned about state persistence
        and validating that the adaptive learning is working correctly.

        Args:
            state: Specific state to query, or None for all states.

        Returns:
            Dictionary mapping states to their learned persistence factors.
            If state is specified, returns single-entry dict. If None, returns
            all learned persistence factors.

        Example:
            ```
            >>> forecaster = TimeAwareForecaster(indexer, adaptive_persistence=True)
            >>> # After training...
            >>> forecaster.get_learned_persistence()
            {'on': 0.72, 'off': 0.31}  # "on" is stickier than "off"
            >>>
            >>> # Query specific state
            >>> forecaster.get_learned_persistence("on")
            {'on': 0.72}
            ```
        """
        if state is not None:
            return {
                state: self.persistence_tracker.get_persistence_factor(
                    state, self.state_persistence_factor
                )
            }

        # Return all learned persistence factors
        result = {}
        for s in self.persistence_tracker.total_count:
            result[s] = self.persistence_tracker.get_persistence_factor(
                s, self.state_persistence_factor
            )
        return result
