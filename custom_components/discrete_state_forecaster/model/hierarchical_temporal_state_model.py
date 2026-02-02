"""
Discrete conditional probability model for state prediction.

This module implements a time-conditioned discrete state prediction model that
learns probability distributions of states based on temporal patterns. The model
partitions time into buckets and tracks state durations within each bucket to
predict future states based on historical patterns.

The model uses temporal keys (TimeKey) to organize state statistics, enabling
predictions that vary by time of day, day of week, month, or any combination
of temporal features.

Example:
    ```
    >>> from .time_indexers import CompositeIndexer, TimeOfDayIndexer, DayOfWeekIndexer
    >>> model = HierarchicalTemporalStateModel()
    >>> model.update_duration((("time_of_day", 600),), "on", 150.0)
    >>> model.update_duration((("time_of_day", 600),), "off", 50.0)
    >>> model.distribution((("time_of_day", 600),))
    {'on': 0.75, 'off': 0.25}
    ```
"""

import math
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.aggregated_stats import (
    AggregatedStats,
)
from custom_components.discrete_state_forecaster.model.confidence import Confidence
from custom_components.discrete_state_forecaster.model.hierarchical_state_stats import (
    HierarchicalStateStats,
)
from custom_components.discrete_state_forecaster.model.prediction import Prediction
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)

MIN_DURATION_THRESHOLD: Final = 5.0  # Filter durations below this (seconds)


# Name may should be changed to Hierarchical Temporal Bayesian?
class HierarchicalTemporalStateModel:
    """
    A time-conditioned discrete state prediction model.

    This model learns probability distributions of states conditioned on temporal
    features. It maintains separate statistics for each unique time bucket,
    allowing it to capture patterns that vary by time (e.g., a device being on
    more often during weekday mornings than weekend nights).

    The model accumulates state durations over time and uses them to compute
    probability distributions for each time bucket. Short-duration states
    (< 5 seconds) are filtered as noise to improve prediction quality.

    Attributes:
        _states: Internal dictionary mapping time keys to their state statistics.
            Each TimeKey represents a unique temporal bucket (e.g., "Monday 10:00-10:30").

    Example:
        ```
        >>> model = HierarchicalTemporalStateModel()
        >>> # Record that during time bucket 600, the state was "on" for 100 seconds
        >>> model.update_duration((("time_of_day", 600),), "on", 100.0)
        >>> model.update_duration((("time_of_day", 600),), "off", 200.0)
        >>> # Get probability distribution for this time bucket
        >>> dist = model.distribution((("time_of_day", 600),))
        >>> print(dist)
        {'on': 0.3333333333333333, 'off': 0.6666666666666666}
        ```
    """

    def __init__(self, half_life: float = 0.0):
        """
        Initializes a new HierarchicalTemporalStateModel.

        Creates an empty model with no learned state statistics. Statistics are
        accumulated as durations are updated via the update_duration method.

        The model supports adaptive decay rates that accelerate after detecting
        concept drift, enabling rapid adaptation to new behavioral patterns.

        Args:
            half_life: Half-life in seconds for exponential decay of historical
                observations. When > 0, older observations are exponentially
                down-weighted over time, losing 50% of their weight after the
                specified duration. Default is 0.0 (no decay). Typical values:
                - 3600 (1 hour): quick adaptation to recent patterns
                - 86400 (1 day): balanced short/long-term memory
                - 604800 (1 week): slow adaptation, stable long-term patterns

                The model maintains two decay rates:
                - **half_life_normal**: Normal decay rate (= half_life)
                - **half_life_fast**: Accelerated decay rate (= half_life / 10)

                After detecting concept drift, the model automatically switches
                to fast decay for 15 updates, then returns to normal decay.
                This enables quick forgetting of old patterns while maintaining
                stability during normal operation.

        Example:
            ```
            >>> # No decay: all observations weighted equally
            >>> model = HierarchicalTemporalStateModel(half_life=0.0)
            >>> # 1 day half-life: observations lose 50% weight after 24 hours
            >>> # After drift: 10x faster decay (2.4 hour half-life) for 15 updates
            >>> model = HierarchicalTemporalStateModel(half_life=86400.0)
            ```
        """
        self._stats = HierarchicalStateStats(half_life=half_life)
        self.half_life: Final = half_life
        self.half_life_normal: Final = half_life
        self.half_life_fast: Final = half_life / 10

    def update_duration(
        self: Self,
        key: TimeKey,
        state: State,
        duration: float,
        timestamp: float,
    ) -> None:
        """
        Update the model with a state duration observation.

        Records that a particular state was active for a given duration during
        the specified time bucket. This incrementally updates the model's
        learned statistics. Durations less than 5 seconds are filtered as
        noise to improve prediction quality.

        The method automatically detects concept drift (significant changes in
        state distributions) and triggers accelerated decay to quickly adapt
        to new behavioral patterns.

        Args:
            key: The temporal key identifying the time bucket (e.g., a specific
                hour of the day or day of the week). Typically produced by a
                TimeIndexer's key() method.
            state: The state value that was active (e.g., "on", "off", "heating").
                Must be hashable.
            duration: The duration in seconds that the state was active. Durations
                less than 5 seconds are ignored as noise.
            timestamp: Unix timestamp when this observation occurred. Required
                for decay to function properly; observations are decayed based
                on elapsed time since last update.

        Behavior:
            1. Filters durations < 5 seconds as noise
            2. Retrieves or creates StateStats for the time bucket
            3. Applies exponential decay (normal or fast rate)
            4. Adds duration to state statistics
            5. Checks for concept drift using Jensen-Shannon divergence
            6. If drift detected: sets fast_decay_updates = 15
            7. Decrements fast_decay_updates counter (if > 0)

        Concept Drift Adaptation:
            When drift is detected, the model enters "fast decay mode" for the
            next 15 updates:
            - Uses half_life_fast (= half_life / 10) for accelerated forgetting
            - Quickly reduces weight of old observations
            - Adapts rapidly to new behavioral patterns
            - Automatically returns to normal decay after 15 updates

        Example:
            ```
            >>> model = HierarchicalTemporalStateModel(half_life=86400.0)
            >>> # Normal operation: 1-day half-life
            >>> model.update_duration((("time_of_day", 600),), "on", 150.0)
            >>> # Pattern changes significantly (drift detected)
            >>> model.update_duration((("time_of_day", 600),), "off", 3000.0)
            >>> # Next 15 updates use fast decay (2.4-hour half-life)
            >>> # Old "on"-dominated pattern quickly forgotten
            >>> # After 15 updates: returns to normal 1-day half-life
            ```
        """
        if duration < MIN_DURATION_THRESHOLD:  # Filter noise
            return

        self._stats.update(
            key=key,
            state=state,
            duration=duration,
            timestamp=timestamp,
        )

    def distribution(self: Self, key: TimeKey, timestamp: float) -> AggregatedStats:
        """
        Gets the aggregated probability distribution for a specific time bucket.

        Returns the learned probability distribution over states for the given
        time bucket, blended hierarchically across all parent temporal contexts.
        Uses support-weighted blending to combine statistics from specific to
        general temporal patterns.

        When the model has decay enabled (half_life > 0), this method applies
        exponential decay to the statistics based on the elapsed time since the
        last update, ensuring that recent observations are weighted more heavily
        than older ones.

        Args:
            key: The temporal key identifying the time bucket for which to
                retrieve the distribution.
            timestamp: Unix timestamp for the distribution calculation. Used to
                apply exponential decay when half_life > 0. The decay factor is
                calculated based on the elapsed time since the last update.

        Returns:
            AggregatedStats containing:
            - distribution: Dictionary mapping state values to their probabilities
              (values between 0 and 1 that sum to 1). Empty dict if no sufficient
              observations have been recorded.
            - support_time: Total accumulated support time across all hierarchy
              levels that contributed to the distribution.
            - depth: Number of hierarchy levels that had sufficient support to
              contribute to the blended distribution.

        Example:
            ```
            >>> model = HierarchicalTemporalStateModel()
            >>> key = TimeKey((("time_of_day", 600),))
            >>> model.update_duration(key, "on", 100.0, timestamp=1000.0)
            >>> model.update_duration(key, "off", 200.0, timestamp=1000.0)
            >>> agg_stats = model.distribution(key, timestamp=1000.0)
            >>> print(agg_stats.distribution)
            {'on': 0.3333333333333333, 'off': 0.6666666666666666}
            >>> print(agg_stats.support_time)
            300.0
            >>> # Unknown time buckets return empty distribution
            >>> model.distribution(TimeKey((("time_of_day", 700),)), timestamp=1000.0)
            AggregatedStats(distribution={}, support_time=0.0, depth=0)
            ```
        """
        return self._stats.distribution(key, timestamp=timestamp)

    def predict(self: Self, key: TimeKey, timestamp: float) -> Prediction:
        """
        Predicts the most likely state for a specific time bucket.

        Analyzes the learned probability distribution for the given time bucket
        and returns the state with the highest probability, along with the full
        distribution and confidence metrics.

        When the model has decay enabled (half_life > 0), this method automatically
        applies exponential decay to the bucket's state statistics before making
        the prediction. This ensures predictions reflect recent patterns more
        heavily than older observations.

        Args:
            key: The temporal key identifying the time bucket for which to
                make a prediction.
            timestamp: Unix timestamp for the prediction. Used to apply exponential
                decay when half_life > 0. When decay is enabled, this timestamp is used to
                calculate elapsed time since the last update, which determines
                the decay factor applied to historical observations. For models
                without decay, this parameter has no effect.

        Returns:
            A Prediction object containing:
            - state: The predicted state (the one with highest probability),
                or None if no data exists for this time bucket.
            - distribution: Dictionary mapping states to their probabilities.
                Empty dict if no data exists.
            - confidence: Confidence metrics assessing prediction reliability,
                including max probability, entropy-based confidence, and total
                support time. All metrics are 0 if no data exists.

        Behavior:
            - Retrieves state statistics for the given time bucket
            - If decay enabled: applies exponential decay based on elapsed time
              since last update, down-weighting older observations
            - Calculates probability distribution from (possibly decayed) durations
            - Returns state with maximum probability along with confidence metrics
            - Returns empty Prediction if no data exists for the bucket

        Examples:
            ```
            >>> # Basic prediction without decay
            >>> model = HierarchicalTemporalStateModel()
            >>> model.update_duration((('time_of_day', 600),), 'on', 300.0)
            >>> model.update_duration((('time_of_day', 600),), 'off', 100.0)
            >>> prediction = model.predict((('time_of_day', 600),))
            >>> print(f"Predicted: {prediction.state}")
            Predicted: on
            >>> print(f"Probability: {prediction.distribution[prediction.state]:.2f}")
            Probability: 0.75
            >>> print(f"Max probability: {prediction.confidence.max_probability:.2f}")
            Max probability: 0.75

            >>> # Prediction with decay - recent patterns matter more
            >>> model = HierarchicalTemporalStateModel(half_life=3600.0)  # 1 hour half-life
            >>> key = (('hour', 10),)
            >>> # Old observation: mostly 'off'
            >>> model.update_duration(key, 'off', 1000.0, timestamp=0.0)
            >>> model.update_duration(key, 'on', 100.0, timestamp=0.0)
            >>> # After 2 half-lives (2 hours), old data decayed to ~25%
            >>> # Recent observation: mostly 'on'
            >>> model.update_duration(key, 'on', 400.0, timestamp=7200.0)
            >>> # Prediction at t=7200 reflects recent pattern
            >>> pred = model.predict(key, timestamp=7200.0)
            >>> pred.state
            'on'
            >>> # Decayed: off ~250, on ~25 (old) + 400 (new) = ~425
            >>> # Distribution: on ~63%, off ~37%

            >>> # Unknown time bucket returns empty prediction
            >>> model.predict((('time_of_day', 700),))
            Prediction(
              state=None,
              distribution={},
              confidence=Confidence(max_probability=0, entropy_confidence=0, support_time=0)
            )

            >>> # Explicit timestamp for decay calculation
            >>> model = HierarchicalTemporalStateModel(half_life=86400.0)  # 1 day
            >>> key = (('hour', 14),)
            >>> model.update_duration(key, 'heating', 5000.0, timestamp=0.0)
            >>> # Predict 3 days later - heavy decay applied
            >>> pred = model.predict(key, timestamp=3*86400.0)
            >>> # After 3 half-lives: heating duration ~625s (5000 x 0.5^3)
            >>> pred.confidence.support_time
            625.0
            ```

        Note:
            When decay is enabled, each call to predict() modifies the internal
            state statistics by applying decay. This ensures the model's internal
            state always reflects the current timestamp. Subsequent predictions
            or operations will use these decayed values unless new observations
            are added.
        """
        stats = self._stats.distribution(key, timestamp)

        # Handle empty distribution (no data for this bucket)
        if not stats.distribution:
            return Prediction(
                state=None,
                distribution={},
                key=None,
                confidence=Confidence(
                    max_probability=0.0, entropy_confidence=0.0, support_time=0.0, depth=0
                ),
            )

        state = max(stats.distribution, key=stats.distribution.get)

        max_p = max(stats.distribution.values())

        ent = self._entropy(stats.distribution)
        max_ent = math.log2(len(stats.distribution)) if len(stats.distribution) > 1 else 0
        entropy_conf = 1 - ent / max_ent if max_ent > 0 else 1.0

        return Prediction(
            state=state,
            distribution=stats.distribution,
            key=stats.key,
            confidence=Confidence(
                max_probability=max_p,
                entropy_confidence=entropy_conf,
                support_time=stats.support_time,
                depth=stats.depth,
            ),
        )

    def _entropy(self: Self, dist: dict[State, float]) -> float:
        """
        Calculates Shannon entropy of a probability distribution.

        Entropy measures the uncertainty or "spread" of a probability distribution.
        Lower entropy indicates more certainty (probability mass concentrated on
        fewer states), while higher entropy indicates more uncertainty (probability
        mass spread across many states).

        Args:
            dist: Dictionary mapping states to their probabilities (values sum to 1).

        Returns:
            Entropy in bits (using log base 2). Range is [0, log2(n_states)].
            - 0 bits: all probability on one state (complete certainty)
            - log2(n): uniform distribution over n states (maximum uncertainty)

        Example:
            ```
            >>> # Low entropy: certain prediction
            >>> model._entropy({'on': 0.99, 'off': 0.01})
            0.08...
            >>> # High entropy: uncertain prediction
            >>> model._entropy({'on': 0.5, 'off': 0.5})
            1.0
            >>> # Maximum entropy for 4 states
            >>> model._entropy({'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25})
            2.0
            ```
        """
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def prune(
        self: Self,
        now_ts: float,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
        min_total: float = 60.0,
    ) -> None:
        """
        Removes low-weight states and time buckets with insufficient data.

        Performs two-level pruning to reduce model size and eliminate noise:
        1. State-level: Removes states with durations below the threshold within
           each time bucket (via StateStats.prune)
        2. Bucket-level: Removes entire time buckets whose total duration falls
           below the support threshold after state pruning

        This is especially useful after decay operations, where old observations
        may have decayed to insignificant weights. Regular pruning helps:
        - Reduce memory footprint by removing rarely-seen states
        - Eliminate noise from insufficient observations
        - Maintain model quality by focusing on well-supported patterns
        - Clean up time buckets that are no longer relevant

        Args:
            now_ts: Current timestamp used for decay calculations.
            epsilon: Relative threshold for pruning states. States with relative
                support below this fraction are removed.
            absolute_min: Absolute minimum duration threshold in seconds for
                individual states within each bucket. States with durations
                strictly less than this value are removed from their bucket.
                Typical values: 10-60 seconds.
            min_total: Minimum total duration threshold in seconds for time buckets.
                After state pruning, buckets with total duration less than this
                value are removed entirely. This ensures predictions are only made
                for buckets with sufficient data. Typical values: 60-300 seconds.

        Behavior:
            - First applies state-level pruning to all buckets
            - Then removes buckets below support threshold
            - Empty buckets (after state pruning) are always removed
            - Order of operations: prune states → check bucket support → delete buckets

        Examples:
            ```
            >>> model = HierarchicalTemporalStateModel()
            >>> model.update_duration((('hour', 10),), 'on', 100.0)
            >>> model.update_duration((('hour', 10),), 'off', 5.0)
            >>> model.update_duration((('hour', 11),), 'idle', 20.0)
            >>> # Remove states < 10s, buckets with support < 50s
            >>> model.prune(min_state_duration=10.0, min_bucket_support=50.0)
            >>> # hour=10 bucket: 'off' removed (5.0 < 10), bucket kept (100 >= 50)
            >>> # hour=11 bucket: removed entirely (20 < 50)
            >>> model._states[('hour', 10)].durations
            {'on': 100.0}
            >>> ('hour', 11) in model._states
            False

            >>> # Typical usage after decay
            >>> model = HierarchicalTemporalStateModel(half_life=86400.0)  # 1 day
            >>> # ... record observations over time ...
            >>> # After 1 week, prune states that decayed below threshold
            >>> model.prune(min_state_duration=10.0, min_bucket_support=60.0)
            ```
        """
        self._stats.prune(now_ts, epsilon, absolute_min, min_total)

    def to_dict(self: Self) -> dict[str, any]:
        """
        Serializes the HierarchicalTemporalStateModel to a dictionary.

        Returns:
            Dictionary containing all model data including learned statistics.
        """
        return {
            "stats": self._stats.to_dict(),
            "half_life": self.half_life,
            "half_life_normal": self.half_life_normal,
            "half_life_fast": self.half_life_fast,
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Self:
        """
        Deserializes a HierarchicalTemporalStateModel from a dictionary.

        Args:
            data: Dictionary containing serialized model data.

        Returns:
            Restored HierarchicalTemporalStateModel instance.
        """
        half_life = data.get("half_life", 0.0)
        instance = cls(half_life=half_life)

        # Restore the internal stats object
        stats_data = data.get("stats", {})
        instance._stats = HierarchicalStateStats.from_dict(stats_data)

        return instance
