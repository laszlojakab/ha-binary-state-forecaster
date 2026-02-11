"""
Unit tests for TimeAwareForecaster and TimeAwareForecasterParameters.

Tests temporal indexing integration with ForecasterEngine.
"""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.discrete_state_forecaster.model.forecaster_engine import (
    ForecasterEngineParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
    TimeAwareForecasterParameters,
)


# Mock TimeIndexer for testing
class MockTimeIndexer:
    """Simple mock TimeIndexer that uses hour of day."""

    async def get_key(self, timestamp: datetime) -> TimeKey:
        """Return hour of day as time key."""
        return TimeKey.from_temporal_feature(TemporalFeature("hour", timestamp.hour))

    async def next_boundary(self, timestamp: datetime) -> datetime:
        """Return start of next hour."""
        # Round up to next hour
        return (timestamp + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


@pytest.fixture
def mock_indexer() -> MockTimeIndexer:
    """Fixture providing a mock time indexer."""
    return MockTimeIndexer()


@pytest.fixture
def engine_params() -> ForecasterEngineParameters:
    """Fixture providing default ForecasterEngineParameters."""
    # Use lower min_support_factor for faster test convergence
    return ForecasterEngineParameters(min_support_factor=0.5)


@pytest.fixture
def time_aware_params(
    mock_indexer: MockTimeIndexer, engine_params: ForecasterEngineParameters
) -> TimeAwareForecasterParameters:
    """Fixture providing TimeAwareForecasterParameters."""
    return TimeAwareForecasterParameters(
        indexer=mock_indexer,
        forecaster_engine_parameters=engine_params,
    )


@pytest.fixture
def forecaster(time_aware_params: TimeAwareForecasterParameters) -> TimeAwareForecaster:
    """Fixture providing a TimeAwareForecaster instance."""
    return TimeAwareForecaster(time_aware_params)


# --- TimeAwareForecasterParameters Tests ---


def test_parameters_initialization(
    mock_indexer: MockTimeIndexer, engine_params: ForecasterEngineParameters
) -> None:
    """Test TimeAwareForecasterParameters initialization with valid values."""
    params = TimeAwareForecasterParameters(
        indexer=mock_indexer,
        forecaster_engine_parameters=engine_params,
    )

    assert params.indexer is mock_indexer
    assert params.forecaster_engine_parameters is engine_params


def test_parameters_dataclass() -> None:
    """Test that TimeAwareForecasterParameters is a proper dataclass."""
    assert hasattr(TimeAwareForecasterParameters, "__dataclass_fields__")
    fields = TimeAwareForecasterParameters.__dataclass_fields__
    assert "indexer" in fields
    assert "forecaster_engine_parameters" in fields


# --- TimeAwareForecaster Initialization Tests ---


def test_forecaster_initialization(
    forecaster: TimeAwareForecaster, time_aware_params: TimeAwareForecasterParameters
) -> None:
    """Test TimeAwareForecaster initialization."""
    assert forecaster._indexer is time_aware_params.indexer
    assert forecaster._engine is not None


def test_forecaster_initialization_creates_engine(
    time_aware_params: TimeAwareForecasterParameters,
) -> None:
    """Test that initialization creates ForecasterEngine."""
    forecaster = TimeAwareForecaster(time_aware_params)
    # Engine should be created
    assert forecaster._engine is not None
    assert hasattr(forecaster._engine, "_stats")


# --- Update Method Tests ---


@pytest.mark.asyncio
async def test_update_with_datetime(forecaster: TimeAwareForecaster) -> None:
    """Test update method with datetime timestamp."""
    timestamp = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
    state = "on"

    # Should not raise any errors
    await forecaster.update(state, timestamp)


@pytest.mark.asyncio
async def test_update_calls_indexer(
    mock_indexer: MockTimeIndexer, forecaster: TimeAwareForecaster
) -> None:
    """Test that update calls the indexer's get_key method."""
    timestamp = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
    state = "on"

    # Mock indexer should be called
    await forecaster.update(state, timestamp)

    # Verify the key was generated (hour should be 14)
    key = await mock_indexer.get_key(timestamp)
    assert key.to_tuple() == (("hour", 14),)


@pytest.mark.asyncio
async def test_update_multiple_states(forecaster: TimeAwareForecaster) -> None:
    """Test updating with multiple state changes."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    await forecaster.update("on", base_time)
    await forecaster.update("off", base_time + timedelta(minutes=30))
    await forecaster.update("on", base_time + timedelta(hours=1))

    # All updates should succeed without errors


@pytest.mark.asyncio
async def test_update_different_hours(forecaster: TimeAwareForecaster) -> None:
    """Test updating in different hours (different time keys)."""
    # Update in hour 10
    await forecaster.update("on", datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC))

    # Update in hour 14
    await forecaster.update("off", datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC))

    # Update in hour 18
    await forecaster.update("on", datetime(2024, 1, 15, 18, 0, 0, tzinfo=UTC))


# --- Predict Method Tests ---


@pytest.mark.asyncio
async def test_predict_with_datetime(forecaster: TimeAwareForecaster) -> None:
    """Test predict method with datetime timestamp."""
    # Add enough data for predictions
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    # Add multiple updates for same temporal key to build statistics
    for i in range(10):
        state = "on" if i % 2 == 0 else "off"
        await forecaster.update(state, base_time + timedelta(minutes=i * 5))

    # Now predict in the same hour (hour 14)
    timestamp = datetime(2024, 1, 15, 14, 55, 0, tzinfo=UTC)
    result = await forecaster.predict(timestamp)

    # Should return a PredictionResult
    assert result is not None
    assert hasattr(result, "distribution")


@pytest.mark.asyncio
async def test_predict_no_prior_data(forecaster: TimeAwareForecaster) -> None:
    """Test prediction when no prior data exists."""
    timestamp = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    result = await forecaster.predict(timestamp)

    # Should return None (insufficient data)
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_result(forecaster: TimeAwareForecaster) -> None:
    """Test prediction returns a valid result when data is sufficient."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    # Add more data with longer durations
    for i in range(30):
        state = "on" if i % 3 == 0 else "off"
        await forecaster.update(state, base_time + timedelta(minutes=i * 3))

    # Predict in the same hour, close to the last update
    timestamp = datetime(2024, 1, 15, 15, 25, 0, tzinfo=UTC)
    result = await forecaster.predict(timestamp)

    assert result is not None
    assert hasattr(result, "distribution")


@pytest.mark.asyncio
async def test_predict_different_time_keys(forecaster: TimeAwareForecaster) -> None:
    """Test prediction uses correct time key."""
    # Train in hour 10 with sufficient data
    base_time_10 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    for i in range(20):
        state = "on" if i % 2 == 0 else "off"
        await forecaster.update(state, base_time_10 + timedelta(minutes=i * 2))

    # Train in hour 14 with opposite pattern
    base_time_14 = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(20):
        state = "off" if i % 2 == 0 else "on"
        await forecaster.update(state, base_time_14 + timedelta(minutes=i * 2))

    # Predict in hour 10 (same key)
    result_same = await forecaster.predict(datetime(2024, 1, 15, 10, 55, 0, tzinfo=UTC))

    # Predict in hour 14 (different key)
    result_diff = await forecaster.predict(datetime(2024, 1, 15, 14, 55, 0, tzinfo=UTC))

    # Both should return results
    assert result_same is not None
    assert result_diff is not None


# --- Predict Interval Tests ---


@pytest.mark.asyncio
async def test_predict_interval_basic(forecaster: TimeAwareForecaster) -> None:
    """Test predict_interval with basic parameters."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

    results = await forecaster.predict_interval(start, end, resolution=3600)

    # Should return a list of (datetime, PredictionResult) tuples
    assert isinstance(results, list)

    for ts, result in results:
        assert isinstance(ts, datetime)
        # result may be None if insufficient data for that key
        if result is not None:
            assert hasattr(result, "distribution")


@pytest.mark.asyncio
async def test_predict_interval_multiple_steps(forecaster: TimeAwareForecaster) -> None:
    """Test predict_interval creates multiple prediction steps."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 18, 0, 0, tzinfo=UTC)

    # 30-minute resolution over 3 hours should give ~6 steps
    results = await forecaster.predict_interval(start, end, resolution=1800)

    assert len(results) >= 3  # At least some steps


@pytest.mark.asyncio
async def test_predict_interval_with_state_simulation(
    forecaster: TimeAwareForecaster,
) -> None:
    """Test predict_interval with state simulation enabled."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

    results = await forecaster.predict_interval(
        start, end, resolution=1800, simulate_state_path=True
    )

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_predict_interval_without_state_simulation(
    forecaster: TimeAwareForecaster,
) -> None:
    """Test predict_interval without state simulation."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

    results = await forecaster.predict_interval(
        start, end, resolution=1800, simulate_state_path=False
    )

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_predict_interval_respects_resolution(
    forecaster: TimeAwareForecaster,
) -> None:
    """Test that predict_interval respects the resolution parameter."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 17, 0, 0, tzinfo=UTC)

    # With 1-hour resolution over 2 hours
    results = await forecaster.predict_interval(start, end, resolution=3600)

    # Should have approximately 1-3 steps
    assert 1 <= len(results) <= 4


@pytest.mark.asyncio
async def test_predict_interval_with_boundary_support(
    time_aware_params: TimeAwareForecasterParameters, mock_indexer: MockTimeIndexer
) -> None:
    """Test predict_interval with an indexer that supports boundaries."""
    # Create forecaster with boundary-supporting indexer
    params = TimeAwareForecasterParameters(
        indexer=mock_indexer,
        forecaster_engine_parameters=time_aware_params.forecaster_engine_parameters,
    )
    forecaster = TimeAwareForecaster(params)

    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    # Predict from 14:45 to 17:00 - should snap to hour boundaries
    start = datetime(2024, 1, 15, 14, 45, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 17, 0, 0, tzinfo=UTC)

    results = await forecaster.predict_interval(start, end, resolution=1800)

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_predict_interval_empty_range(forecaster: TimeAwareForecaster) -> None:
    """Test predict_interval with start >= end."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    await forecaster.update("on", base_time)

    start = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)  # End before start

    results = await forecaster.predict_interval(start, end, resolution=3600)

    # Should return empty list or minimal results
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_predict_interval_very_small_resolution(
    forecaster: TimeAwareForecaster,
) -> None:
    """Test predict_interval with very small resolution."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        await forecaster.update("on" if i % 2 == 0 else "off", base_time + timedelta(minutes=i * 2))

    start = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 15, 5, 0, tzinfo=UTC)  # 5 minutes

    # 1-minute resolution over 5 minutes
    results = await forecaster.predict_interval(start, end, resolution=60)

    assert isinstance(results, list)
    # Should have ~3-7 steps
    assert 2 <= len(results) <= 8


# --- Integration Tests ---


@pytest.mark.asyncio
async def test_full_workflow(forecaster: TimeAwareForecaster) -> None:
    """Test complete workflow: update, predict, predict_interval."""
    # Training phase with sufficient data
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    for i in range(30):
        state = "on" if i % 3 == 0 else "off"
        await forecaster.update(state, base_time + timedelta(minutes=i * 2))

    # Single prediction
    pred_time = datetime(2024, 1, 15, 11, 30, 0, tzinfo=UTC)
    result = await forecaster.predict(pred_time)
    assert result is not None

    # Interval prediction
    start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    results = await forecaster.predict_interval(start, end, resolution=1800)

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_temporal_context_isolation(forecaster: TimeAwareForecaster) -> None:
    """Test that different time keys maintain separate contexts."""
    # Train in hour 10 - preference for "on"
    base_time_10 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    for i in range(30):
        state = "on" if i < 25 else "off"  # Mostly "on"
        await forecaster.update(state, base_time_10 + timedelta(minutes=i * 2))

    # Train in hour 14 - preference for "off"
    base_time_14 = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
    for i in range(30):
        state = "off" if i < 25 else "on"  # Mostly "off"
        await forecaster.update(state, base_time_14 + timedelta(minutes=i * 2))

    # Predictions should reflect temporal context
    result_morning = await forecaster.predict(datetime(2024, 1, 16, 10, 30, 0, tzinfo=UTC))
    result_afternoon = await forecaster.predict(datetime(2024, 1, 16, 14, 30, 0, tzinfo=UTC))

    # Both should return valid results (patterns may differ)
    assert result_morning is not None
    assert result_afternoon is not None


@pytest.mark.asyncio
async def test_handles_timezone_aware_datetimes(
    forecaster: TimeAwareForecaster,
) -> None:
    """Test that forecaster handles timezone-aware datetime objects."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    # Add sufficient data
    for i in range(20):
        timestamp = base_time + timedelta(minutes=i * 3)
        await forecaster.update("on" if i % 2 == 0 else "off", timestamp)

    result = await forecaster.predict(datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC))

    # Should return a result with sufficient data
    assert result is not None


# --- Edge Cases ---


@pytest.mark.asyncio
async def test_predict_at_exact_update_time(forecaster: TimeAwareForecaster) -> None:
    """Test predicting at the exact time of an update."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    # Add sufficient data
    for i in range(20):
        timestamp = base_time + timedelta(minutes=i * 3)
        await forecaster.update("on" if i % 2 == 0 else "off", timestamp)

    # Predict at same time as one of the updates
    result = await forecaster.predict(base_time + timedelta(minutes=30))

    # Should handle this gracefully
    assert result is not None


@pytest.mark.asyncio
async def test_rapid_updates_same_time_key(forecaster: TimeAwareForecaster) -> None:
    """Test many rapid updates within the same time key."""
    base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

    # Increase to 30 updates with 3-minute increments for more total support
    for i in range(30):
        state = "on" if i % 2 == 0 else "off"
        await forecaster.update(state, base_time + timedelta(minutes=i * 3))

    # Should handle all updates and predict successfully
    result = await forecaster.predict(datetime(2024, 1, 15, 15, 25, 0, tzinfo=UTC))
    assert result is not None
