"""
State tracking for temporal pattern learning.

This module provides the StateTracker class for tracking state transitions
over time and updating a forecasting model with observed state intervals.
"""

from datetime import datetime
from typing import Self

from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)


class StateTracker:
    """
    Tracks state transitions and updates a time-aware forecaster.

    The StateTracker maintains the current state and timestamp, automatically
    calculating state intervals and feeding them to a TimeAwareForecaster for
    pattern learning. Each time a new state is observed, the tracker records
    the interval between the previous state and the current timestamp, then
    updates its internal state.

    This class simplifies the process of building temporal models from state
    observations by handling interval calculation and forecaster updates
    automatically.

    Attributes:
        forecaster: The TimeAwareForecaster to update with state intervals.
        last_state: The most recently observed state, or None if no states
            have been observed yet.
        last_ts: The timestamp of the most recent state observation, or None
            if no states have been observed yet.

    Example:
        ```
        >>> from datetime import datetime
        >>> from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
        ...     TimeAwareForecaster,
        ... )
        >>> from custom_components.discrete_state_forecaster.model.time_indexers import (
        ...     TimeOfDayIndexer,
        ...     CompositeIndexer,
        ... )
        >>>
        >>> # Create a forecaster with time-of-day indexing
        >>> indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        >>> forecaster = TimeAwareForecaster(indexer)
        >>> tracker = StateTracker(forecaster)
        >>>
        >>> # Track state observations
        >>> # First observation - no interval recorded
        >>> tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        >>> # Records "on" for 1 hour at 10:00
        >>> tracker.update(datetime(2024, 1, 1, 11, 0), "off")
        >>> # Records "off" for 2 hours at 11:00
        >>> tracker.update(datetime(2024, 1, 1, 13, 0), "on")
        >>>
        >>> # Now the forecaster has learned the patterns
        >>> prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        >>> print(prediction.state)  # Likely "on" based on learned patterns
        ```
    """

    def __init__(self: Self, forecaster: TimeAwareForecaster) -> None:
        """
        Initializes a StateTracker with a forecaster.

        Args:
            forecaster: The TimeAwareForecaster to update with observed
                state intervals. This forecaster will learn temporal patterns
                from the state transitions tracked by this tracker.
        """
        self.forecaster: TimeAwareForecaster = forecaster
        self.last_state: State | None = None
        self.last_ts: datetime | None = None

    async def update(self: Self, timestamp: datetime, new_state: State) -> None:
        """
        Update the tracker with a new state observation.

        Records the interval from the previous state observation to the current
        timestamp, then updates the internal state to the new observation. The
        first call to update() only records the initial state and timestamp
        without updating the forecaster (since there's no previous interval).

        Subsequent calls calculate the time interval between the previous
        timestamp and the current timestamp, associate it with the previous
        state, and pass this information to the forecaster for learning.

        Args:
            timestamp: The timestamp when the new state was observed. Must be greater
                than or equal to the previous timestamp for meaningful intervals.
            new_state: The newly observed state. Can be any hashable value
                representing a discrete state.

        Example:
            ```
            >>> tracker = StateTracker(forecaster)
            >>>
            >>> # First update establishes baseline
            >>> await tracker.update(datetime(2024, 1, 1, 10, 0), "idle")
            >>> assert tracker.last_state == "idle"
            >>> assert tracker.last_ts == datetime(2024, 1, 1, 10, 0)
            >>>
            >>> # Second update records interval and updates state
            >>> await tracker.update(datetime(2024, 1, 1, 10, 30), "active")
            >>> assert tracker.last_state == "active"
            >>> # Forecaster now knows "idle" lasted 30 minutes at 10:00
            ```
        """
        if self.last_state is None:
            # First observation - just record state and timestamp
            self.last_state = new_state
            self.last_ts = timestamp
            return

        # Record the interval for the previous state
        await self.forecaster.update_interval(
            self.last_ts,
            timestamp,
            self.last_state,
        )

        # Update to the new state
        self.last_state = new_state
        self.last_ts = timestamp
