"""Tests for StateTracker class."""

from datetime import datetime

import pytest

from custom_components.discrete_state_forecaster.model.state_tracker import (
    StateTracker,
)
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.model.time_indexers import (
    CompositeIndexer,
    DayOfWeekIndexer,
    TimeOfDayIndexer,
)


class TestStateTrackerInitialization:
    """Tests for StateTracker initialization."""

    def test_initialization(self) -> None:
        """Test StateTracker initializes with correct attributes."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        assert tracker.forecaster is forecaster
        assert tracker.last_state is None
        assert tracker.last_ts is None

    def test_initialization_with_composite_indexer(self) -> None:
        """Test StateTracker works with composite indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        assert tracker.forecaster is forecaster
        assert tracker.last_state is None
        assert tracker.last_ts is None


class TestStateTrackerUpdate:
    """Tests for StateTracker update method."""

    def test_first_update_only_records_state(self) -> None:
        """Test first update only records state without updating forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        ts = datetime(2024, 1, 1, 10, 0)
        tracker.update(ts, "on")

        assert tracker.last_state == "on"
        assert tracker.last_ts == ts

        # Forecaster should not have learned anything yet
        prediction = forecaster.predict(ts)
        assert prediction.state is None
        assert prediction.distribution == {}

    def test_second_update_records_interval(self) -> None:
        """Test second update records interval and updates forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # First update
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")

        # Second update - should record interval
        tracker.update(datetime(2024, 1, 1, 11, 0), "off")

        assert tracker.last_state == "off"
        assert tracker.last_ts == datetime(2024, 1, 1, 11, 0)

        # Forecaster should have learned the "on" state for 10:00-11:00
        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction.state == "on"
        assert prediction.distribution == {"on": 1.0}

    def test_multiple_updates_accumulate_patterns(self) -> None:
        """Test multiple updates accumulate learning in forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track multiple state transitions
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "off")
        tracker.update(datetime(2024, 1, 1, 13, 0), "on")
        tracker.update(datetime(2024, 1, 1, 14, 0), "off")

        # Check learned pattern at 10:00
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction.state == "on"

        # Check learned pattern at 11:00
        prediction = forecaster.predict(datetime(2024, 1, 2, 11, 30))
        assert prediction.state == "off"

    def test_update_with_same_state_accumulates_duration(self) -> None:
        """Test updating with same state multiple times accumulates duration."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track same state over time
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 10, 30), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 30), "off")

        # Should have learned "on" for 1.5 hours in 10:00 bucket
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction.state == "on"


class TestStateTrackerStateTracking:
    """Tests for state tracking behavior."""

    def test_last_state_updates_correctly(self) -> None:
        """Test last_state attribute updates with each new state."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "idle")
        assert tracker.last_state == "idle"

        tracker.update(datetime(2024, 1, 1, 10, 30), "active")
        assert tracker.last_state == "active"

        tracker.update(datetime(2024, 1, 1, 11, 0), "idle")
        assert tracker.last_state == "idle"

    def test_last_ts_updates_correctly(self) -> None:
        """Test last_ts attribute updates with each new timestamp."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        ts1 = datetime(2024, 1, 1, 10, 0)
        tracker.update(ts1, "on")
        assert tracker.last_ts == ts1

        ts2 = datetime(2024, 1, 1, 10, 30)
        tracker.update(ts2, "off")
        assert tracker.last_ts == ts2

        ts3 = datetime(2024, 1, 1, 11, 0)
        tracker.update(ts3, "on")
        assert tracker.last_ts == ts3

    def test_update_with_various_state_types(self) -> None:
        """Test tracker works with different state types."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # String states
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        assert tracker.last_state == "on"

        # Numeric states
        tracker.update(datetime(2024, 1, 1, 11, 0), 1)
        assert tracker.last_state == 1

        # Tuple states
        tracker.update(datetime(2024, 1, 1, 12, 0), ("active", "heating"))
        assert tracker.last_state == ("active", "heating")


class TestStateTrackerWithCompositeIndexer:
    """Tests for StateTracker with composite time indexers."""

    def test_tracker_with_day_and_time_indexer(self) -> None:
        """Test tracker with both day-of-week and time-of-day indexing."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Weekday morning pattern
        tracker.update(datetime(2024, 1, 1, 9, 0), "busy")  # Monday
        tracker.update(datetime(2024, 1, 1, 12, 0), "idle")
        tracker.update(datetime(2024, 1, 2, 9, 0), "busy")  # Tuesday
        tracker.update(datetime(2024, 1, 2, 12, 0), "idle")

        # Weekend morning pattern
        tracker.update(datetime(2024, 1, 6, 9, 0), "idle")  # Saturday
        tracker.update(datetime(2024, 1, 6, 12, 0), "idle")

        # Check weekday prediction
        weekday_prediction = forecaster.predict(datetime(2024, 1, 8, 10, 0))  # Monday
        assert weekday_prediction.state == "busy"

        # Check weekend prediction
        weekend_prediction = forecaster.predict(datetime(2024, 1, 13, 10, 0))  # Saturday
        assert weekend_prediction.state == "idle"


class TestStateTrackerIntervalCalculation:
    """Tests for interval calculation and forecaster updates."""

    def test_interval_duration_calculated_correctly(self) -> None:
        """Test that intervals are calculated with correct duration."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # 1 hour interval
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "off")

        # Check that the forecaster learned 1 hour of "on"
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction.state == "on"
        assert prediction.confidence.support_time == 3600.0  # 1 hour in seconds

    def test_short_intervals_filtered_by_forecaster(self) -> None:
        """Test that very short intervals are filtered by the forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Very short interval (< 5 seconds)
        tracker.update(datetime(2024, 1, 1, 10, 0, 0), "on")
        tracker.update(datetime(2024, 1, 1, 10, 0, 2), "off")  # 2 seconds later

        # Should not have learned this pattern
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 0, 1))
        assert prediction.state is None

    def test_spanning_midnight(self) -> None:
        """Test tracker handles intervals spanning midnight."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # State spanning midnight
        tracker.update(datetime(2024, 1, 1, 23, 0), "on")
        tracker.update(datetime(2024, 1, 2, 1, 0), "off")

        # Check both time buckets learned the pattern
        prediction_23 = forecaster.predict(datetime(2024, 1, 2, 23, 30))
        assert prediction_23.state == "on"

        prediction_00 = forecaster.predict(datetime(2024, 1, 2, 0, 30))
        assert prediction_00.state == "on"


class TestStateTrackerIntegration:
    """Integration tests for StateTracker."""

    def test_complete_tracking_workflow(self) -> None:
        """Test complete workflow: track states and make predictions."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Simulate a week of state observations
        for day in range(1, 8):
            # Morning: on
            tracker.update(datetime(2024, 1, day, 8, 0), "on")
            tracker.update(datetime(2024, 1, day, 12, 0), "off")
            # Afternoon: off
            tracker.update(datetime(2024, 1, day, 18, 0), "on")
            tracker.update(datetime(2024, 1, day, 22, 0), "off")

        # Predict for new day
        morning_prediction = forecaster.predict(datetime(2024, 1, 8, 10, 0))
        assert morning_prediction.state == "on"

        afternoon_prediction = forecaster.predict(datetime(2024, 1, 8, 14, 0))
        assert afternoon_prediction.state == "off"

        evening_prediction = forecaster.predict(datetime(2024, 1, 8, 20, 0))
        assert evening_prediction.state == "on"

    def test_incremental_learning(self) -> None:
        """Test that tracker enables incremental learning over time."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Day 1: Learn one pattern
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "off")

        prediction1 = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction1.state == "on"
        support_time_1 = prediction1.confidence.support_time

        # Day 2: Reinforce the pattern
        tracker.update(datetime(2024, 1, 2, 10, 0), "on")
        tracker.update(datetime(2024, 1, 2, 11, 0), "off")

        prediction2 = forecaster.predict(datetime(2024, 1, 3, 10, 30))
        assert prediction2.state == "on"
        support_time_2 = prediction2.confidence.support_time

        # Support time should increase with more data
        assert support_time_2 > support_time_1

    def test_pattern_change_learning(self) -> None:
        """Test tracker learns when patterns change over time."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Initial pattern: on in morning
        for day in range(1, 4):
            tracker.update(datetime(2024, 1, day, 10, 0), "on")
            tracker.update(datetime(2024, 1, day, 11, 0), "off")

        # Pattern changes: off in morning
        for day in range(4, 7):
            tracker.update(datetime(2024, 1, day, 10, 0), "off")
            tracker.update(datetime(2024, 1, day, 11, 0), "on")

        # Prediction should reflect mixed pattern
        prediction = forecaster.predict(datetime(2024, 1, 7, 10, 30))
        assert prediction.distribution["on"] == pytest.approx(0.5)
        assert prediction.distribution["off"] == pytest.approx(0.5)

    def test_multi_state_tracking(self) -> None:
        """Test tracking with more than two states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track three different states
        tracker.update(datetime(2024, 1, 1, 10, 0), "heating")
        tracker.update(datetime(2024, 1, 1, 10, 30), "cooling")
        tracker.update(datetime(2024, 1, 1, 10, 45), "idle")
        tracker.update(datetime(2024, 1, 1, 11, 0), "heating")

        # Check the learned distribution
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert "heating" in prediction.distribution
        assert "cooling" in prediction.distribution
        assert "idle" in prediction.distribution
