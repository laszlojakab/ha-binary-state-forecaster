"""
Time-aware forecaster with temporal indexing capabilities.

This module provides TimeAwareForecaster, which combines ForecasterEngine with
temporal indexing to enable time-based predictions. It automatically converts
timestamps to temporal keys using a TimeIndexer, allowing the forecaster to
learn and predict patterns within different temporal contexts.
"""

from datetime import datetime, timedelta
from typing import Any, Final, Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_runtime_parameters import (
    ForecasterEngineRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.prediction_result import (
    PredictionResult,
)

from .forecaster_engine import ForecasterEngine
from .state import (
    State,
)
from .structural_parameters import (
    StructuralParameters,
)


class TimeAwareForecaster:
    """
    Time-aware forecaster with temporal indexing.

    This forecaster wraps ForecasterEngine and automatically converts timestamps
    to temporal keys using a TimeIndexer. This enables learning and predicting
    patterns within different temporal contexts (e.g., different behavior at
    different times of day, days of week, or seasons).

    Attributes:
        _engine: Internal ForecasterEngine instance for core prediction logic.
        _indexer: TimeIndexer for converting timestamps to temporal keys.

    Example:
        >>> from datetime import datetime
        >>> params = TimeAwareForecasterParameters(
        ...     forecaster_engine_parameters=ForecasterEngineParameters(half_life=3600.0),
        ... )
        >>> forecaster = TimeAwareForecaster(params)
        >>> await forecaster.update(datetime.now(), "on")
        >>> prediction = await forecaster.predict(datetime.now())
    """

    def __init__(
        self: Self,
        structural_parameters: StructuralParameters,
        runtime_parameters: ForecasterEngineRuntimeParameters,
    ) -> None:
        """
        Initializes the time-aware forecaster.

        Args:
            structural_parameters: Structural parameters including the time indexer.
            runtime_parameters: Runtime parameters including the forecaster engine parameters.
        """
        self._engine: Final = ForecasterEngine(
            parameters=runtime_parameters,
        )
        self._structural_parameters: Final = structural_parameters

    async def update(self: Self, state: State, timestamp: datetime) -> None:
        """
        Updates the forecaster with a new state observation at a given time.

        Converts the timestamp to a temporal key using the indexer, then updates
        the underlying forecaster engine.

        Args:
            state: The observed state value.
            timestamp: Datetime when the state was observed.

        Example:
            >>> await forecaster.update("on", datetime(2024, 1, 15, 14, 30))
        """
        key = await self._structural_parameters.indexer.get_key(timestamp)
        self._engine.update(key, state, timestamp.timestamp())

    async def predict(self: Self, timestamp: datetime) -> PredictionResult | None:
        """
        Generates a state prediction for a given timestamp.

        Converts the timestamp to a temporal key and returns the prediction
        from the forecaster engine.

        Args:
            timestamp: Datetime for which to generate a prediction.

        Returns:
            PredictionResult with predicted distribution and contributions,
            or None if insufficient data is available.

        Example:
            >>> prediction = await forecaster.predict(datetime(2024, 1, 15, 15, 0))
            >>> if prediction:
            ...     dist = prediction.distribution
            ...     print(f"Probability of 'on': {dist.get('on', 0.0)}")
        """
        key = await self._structural_parameters.indexer.get_key(timestamp)
        return self._engine.predict(key)

    async def predict_with_persistence(
        self: Self,
        timestamp: datetime,
        current_state: State | None = None,
        current_state_duration: float | None = None,
    ) -> PredictionResult | None:
        """
        Generates a state prediction considering persistence of the current state.

        This method allows the forecaster to take into account how long the
        current state has been active, which can improve predictions for states
        that exhibit duration-dependent behavior.

        Args:
            timestamp: Datetime for which to generate a prediction.
            current_state: Optional current state to consider for persistence.
            current_state_duration: Optional duration (in seconds) that the
                current state has been active.

        Returns:
            PredictionResult with predicted distribution and contributions,
            or None if insufficient data is available.
        """
        key = await self._structural_parameters.indexer.get_key(timestamp)

        return self._engine.predict_with_persistence(
            key=key,
            current_state=current_state,
            current_state_duration=current_state_duration,
        )

    async def predict_interval(
        self: Self,
        start_ts: datetime,
        end_ts: datetime,
        resolution: float,
        current_state: State | None = None,
        current_state_duration: float | None = None,
        simulate_state_path: bool = True,
    ) -> list[tuple[datetime, PredictionResult]]:
        """
        Generates predictions over a time interval with forward simulation.

        This method performs forward simulation forecasting over a time range,
        optionally simulating the most likely state path. It:
        - Respects temporal key boundaries
        - Uses persistence-aware predictions
        - Updates state duration at each step
        - Optionally simulates future state transitions (argmax)

        The simulation advances in steps, respecting both the specified resolution
        and temporal boundaries (e.g., hour changes, day changes). At each step,
        it predicts the distribution considering the current simulated state and
        its duration.

        Args:
            start_ts: Start datetime for predictions.
            end_ts: End datetime for predictions (exclusive).
            resolution: Time resolution in seconds for prediction steps.
            current_state: Optional current state to start simulation from.
            current_state_duration: Optional duration (seconds) the current state
                has been active.
            simulate_state_path: If True, simulates state transitions by selecting
                the most likely (argmax) state at each step. If False, keeps the
                current state throughout but updates its duration.

        Returns:
            List of tuples (timestamp, prediction) for each prediction step.
            Timestamps are datetime objects.

        Example:
            >>> start = datetime(2024, 1, 15, 14, 0)
            >>> end = datetime(2024, 1, 15, 16, 0)
            >>> predictions = await forecaster.predict_interval(
            ...     start, end, resolution=900.0,  # 15 minutes
            ...     current_state="on", current_state_duration=300.0
            ... )
            >>> for ts, pred in predictions:
            ...     print(f"{ts}: {pred.distribution.distribution()}")
        """
        ts = start_ts
        results: list[tuple[datetime, PredictionResult]] = []

        sim_state = current_state
        sim_duration = current_state_duration or 0.0

        while ts < end_ts:
            key = await self._structural_parameters.indexer.get_key(ts)

            # Calculate step end time, respecting boundaries and resolution.

            step_end_ts = min(
                ts.timestamp() + resolution,
                (
                    await self._structural_parameters.indexer.next_boundary(ts)
                ).timestamp(),
                end_ts.timestamp(),
            )
            step_dt = step_end_ts - ts.timestamp()

            prediction = self._engine.predict_with_persistence(
                key=key,
                current_state=sim_state,
                current_state_duration=sim_duration,
            )

            if prediction is None:
                ts = datetime.fromtimestamp(step_end_ts, tz=start_ts.tzinfo)
                continue

            results.append((ts, prediction))

            # Forward simulation of state
            if simulate_state_path:
                dist = prediction.distribution

                if dist:
                    # Select most likely (argmax) state
                    next_state = max(dist.items(), key=lambda x: x[1])[0]

                    if sim_state == next_state:
                        # Continue in same state
                        sim_duration += step_dt
                    else:
                        # Transition to new state
                        sim_state = next_state
                        sim_duration = step_dt
                else:
                    # No prediction available
                    sim_state = None
                    sim_duration = 0.0
            # If not simulating state path, just update duration
            elif sim_state is not None:
                sim_duration += step_dt

            ts = datetime.fromtimestamp(step_end_ts, tz=start_ts.tzinfo)

        return results

    async def predict_next_transition(
        self: Self,
        timestamp: datetime,
        current_state: State | None = None,
        current_state_duration: float | None = None,
    ) -> datetime | None:
        """
        Predicts the next state transition time after the given timestamp.

        This method uses a binary search approach to find the next transition time
        efficiently. It repeatedly queries the forecaster for predictions at
        future timestamps, narrowing down the interval until it finds the next
        transition time or determines that no transition occurs within a reasonable
        horizon.

        Args:
            timestamp: The starting datetime to search for the next transition.
            current_state: Optional current state to consider for persistence in predictions.
            current_state_duration: Optional duration of the current state to consider
            for persistence in predictions.

        Returns:
            A datetime representing the predicted next transition time, or None if no
            transition is predicted within the search horizon.
        """
        # Define search parameters
        max_horizon_seconds = 7 * 24 * 3600  # Search up to 7 days into the future
        resolution_seconds = 3600  # Start with 1 hour resolution

        start_ts = timestamp
        end_ts = timestamp + timedelta(seconds=max_horizon_seconds)

        while start_ts < end_ts:
            prediction = await self.predict_with_persistence(
                start_ts,
                current_state=current_state,
                current_state_duration=(
                    (
                        current_state_duration
                        + (start_ts.timestamp() - timestamp.timestamp())
                    )
                    if current_state_duration is not None
                    else None
                ),
            )

            if prediction is None:
                # No prediction available, move forward by resolution
                start_ts += timedelta(seconds=resolution_seconds)
                continue

            dist = prediction.distribution

            if not dist:
                # No distribution available, move forward by resolution
                start_ts += timedelta(seconds=resolution_seconds)
                continue

            # Get most likely state
            most_likely_state = max(dist.items(), key=lambda x: x[1])[0]

            if current_state is not None and most_likely_state != current_state:
                # Transition predicted at this timestamp
                return start_ts

            # No transition predicted, move forward by resolution
            start_ts += timedelta(seconds=resolution_seconds)

        return None

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Serialize the forecaster to a dictionary.

        Returns:
            A dictionary containing structural parameters and runtime parameters.
        """
        return {
            "engine": self._engine.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        structural_parameters: StructuralParameters,
        runtime_parameters: ForecasterEngineRuntimeParameters,
    ) -> Self:
        """
        Deserialize the forecaster from a dictionary.

        Args:
            data: Dictionary containing serialized forecaster state.
            structural_parameters: Structural parameters to use for the forecaster.
            runtime_parameters: Runtime parameters to use for the forecaster.
        """
        forecaster = cls(structural_parameters, runtime_parameters)
        forecaster._engine = ForecasterEngine.from_dict(
            data["engine"], runtime_parameters
        )

        return forecaster
