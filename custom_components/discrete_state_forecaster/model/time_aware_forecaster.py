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

from datetime import datetime
from typing import Self

from custom_components.discrete_state_forecaster.model.discrete_conditional_model import (
    DiscreteConditionalModel,
)
from custom_components.discrete_state_forecaster.model.prediction import Prediction
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)


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

    def __init__(self: Self, indexer: CompositeIndexer, half_life: float = 0.0) -> None:
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

        Example:
            ```
            >>> from .time_indexers import CompositeIndexer, TimeOfDayIndexer, DayOfWeekIndexer
            >>> # Create forecaster with hour-of-day and day-of-week indexing
            >>> indexer = CompositeIndexer([
            ...     DayOfWeekIndexer(),
            ...     TimeOfDayIndexer(60)
            ... ])
            >>> forecaster = TimeAwareForecaster(indexer)
            ```
        """
        self.indexer: CompositeIndexer = indexer
        self.model: DiscreteConditionalModel = DiscreteConditionalModel(half_life=half_life)

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
            ```
        """
        if end <= start:
            return

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

    def predict(self: Self, timestamp: datetime) -> Prediction:
        """
        Predicts the most likely state at a specific time.

        Looks up the learned probability distribution for the time bucket
        containing the given timestamp and returns the most likely state
        along with confidence metrics.

        Args:
            timestamp: The timestamp for which to make a prediction. The forecaster
                will determine which temporal bucket contains this time and
                use the learned patterns for that bucket.

        Returns:
            A Prediction object containing:
            - state: The predicted state (highest probability), or None if
                no data exists for this time bucket.
            - distribution: Dictionary mapping states to their probabilities
                for this time bucket. Empty if no data exists.
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
            >>> # Predict for a similar time
            >>> pred_time = datetime(2024, 1, 2, 10, 30)
            >>> prediction = forecaster.predict(pred_time)
            >>> state_prob = prediction.distribution.get(prediction.state, 0)
            >>> print(f"State: {prediction.state}, Probability: {state_prob:.2f}")
            >>> print(f"Confidence: {prediction.confidence.max_probability:.2f}")
            ```
        """
        key = self.indexer.key(timestamp)
        return self.model.predict(key, timestamp.timestamp())
