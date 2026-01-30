"""Tests for TimeAwareForecaster class."""

from datetime import datetime

import pytest

from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.day_of_week_indexer import (
    DayOfWeekIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestTimeAwareForecasterInitialization:
    """Test TimeAwareForecaster initialization."""

    def test_initialization_with_single_indexer(self) -> None:
        """Test initialization with a single time indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(30)])
        forecaster = TimeAwareForecaster(indexer)

        assert forecaster.indexer is indexer
        assert forecaster.model is not None
        assert forecaster.model._states == {}

    def test_initialization_with_composite_indexer(self) -> None:
        """Test initialization with composite indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        assert forecaster.indexer is indexer
        assert forecaster.model._states == {}


class TestTimeAwareForecasterUpdateInterval:
    """Test TimeAwareForecaster update_interval method."""

    def test_update_interval_within_single_bucket(self) -> None:
        """Test updating interval that fits within single time bucket."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 30-minute interval within 10:00-11:00 bucket
        start = datetime(2024, 1, 1, 10, 15)
        end = datetime(2024, 1, 1, 10, 45)

        forecaster.update_interval(start, end, "on")

        # Should have one time bucket with 30 minutes (1800 seconds)
        key = indexer.key(start)
        assert key in forecaster.model._states
        assert forecaster.model._states[key].durations.get("on") == pytest.approx(
            1800.0
        )

    def test_update_interval_spanning_two_buckets(self) -> None:
        """Test updating interval that spans two time buckets."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 30-minute interval: 15 min before 11:00, 15 min after
        start = datetime(2024, 1, 1, 10, 45)
        end = datetime(2024, 1, 1, 11, 15)

        forecaster.update_interval(start, end, "on")

        # Check first bucket (10:00-11:00)
        key1 = indexer.key(datetime(2024, 1, 1, 10, 45))
        assert forecaster.model._states[key1].durations.get("on") == pytest.approx(
            900.0
        )

        # Check second bucket (11:00-12:00)
        key2 = indexer.key(datetime(2024, 1, 1, 11, 15))
        assert forecaster.model._states[key2].durations.get("on") == pytest.approx(
            900.0
        )

    def test_update_interval_spanning_multiple_buckets(self) -> None:
        """Test updating interval that spans multiple time buckets."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 2.5 hour interval spanning 3 buckets
        start = datetime(2024, 1, 1, 10, 30)
        end = datetime(2024, 1, 1, 13, 0)

        forecaster.update_interval(start, end, "heating")

        # First bucket: 10:30-11:00 = 30 minutes
        key1 = indexer.key(datetime(2024, 1, 1, 10, 30))
        assert forecaster.model._states[key1].durations.get("heating") == pytest.approx(
            1800.0
        )

        # Second bucket: 11:00-12:00 = 60 minutes
        key2 = indexer.key(datetime(2024, 1, 1, 11, 30))
        assert forecaster.model._states[key2].durations.get("heating") == pytest.approx(
            3600.0
        )

        # Third bucket: 12:00-13:00 = 60 minutes
        key3 = indexer.key(datetime(2024, 1, 1, 12, 30))
        assert forecaster.model._states[key3].durations.get("heating") == pytest.approx(
            3600.0
        )

    def test_update_interval_invalid_end_before_start(self) -> None:
        """Test that invalid interval (end before start) is ignored."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 11, 0)
        end = datetime(2024, 1, 1, 10, 0)

        forecaster.update_interval(start, end, "on")

        # No updates should be made
        assert forecaster.model._states == {}

    def test_update_interval_invalid_end_equals_start(self) -> None:
        """Test that zero-duration interval is ignored."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        ts = datetime(2024, 1, 1, 10, 0)
        forecaster.update_interval(ts, ts, "on")

        # No updates should be made
        assert forecaster.model._states == {}

    def test_update_interval_crossing_midnight(self) -> None:
        """Test updating interval that crosses midnight."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 23:30 to 00:30 (crosses midnight)
        start = datetime(2024, 1, 1, 23, 30)
        end = datetime(2024, 1, 2, 0, 30)

        forecaster.update_interval(start, end, "on")

        # First bucket: 23:30-00:00 = 30 minutes
        key1 = indexer.key(datetime(2024, 1, 1, 23, 45))
        assert forecaster.model._states[key1].durations.get("on") == pytest.approx(
            1800.0
        )

        # Second bucket: 00:00-00:30 = 30 minutes
        key2 = indexer.key(datetime(2024, 1, 2, 0, 15))
        assert forecaster.model._states[key2].durations.get("on") == pytest.approx(
            1800.0
        )

    def test_update_interval_multiple_states_same_bucket(self) -> None:
        """Test updating same bucket with different states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Two intervals in same hour
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 30),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 11, 0),
            "off",
        )

        key = indexer.key(datetime(2024, 1, 1, 10, 15))
        assert forecaster.model._states[key].durations.get("on") == pytest.approx(
            1800.0
        )
        assert forecaster.model._states[key].durations.get("off") == pytest.approx(
            1800.0
        )

    def test_update_interval_accumulates_durations(self) -> None:
        """Test that repeated updates accumulate durations."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Update same prediction.state multiple times
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 15),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 10, 45),
            "on",
        )

        key = indexer.key(datetime(2024, 1, 1, 10, 0))
        # Total: 15 + 15 = 30 minutes = 1800 seconds
        assert forecaster.model._states[key].durations.get("on") == pytest.approx(
            1800.0
        )

    def test_update_interval_with_composite_indexer(self) -> None:
        """Test updating with multi-dimensional composite indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Monday 10:00-11:00
        start = datetime(2024, 1, 1, 10, 0)  # Monday
        end = datetime(2024, 1, 1, 11, 0)

        forecaster.update_interval(start, end, "busy")

        key = indexer.key(start)
        # Verify the key has the expected structure (weekday=0 for Monday, time_bucket=10 for 10:00)
        assert len(key) == 2
        assert key[0][0] == "weekday"
        assert key[0][1] == 0  # Monday
        assert key[1][0] == "time_bucket"
        assert key[1][1] == 10  # 10:00 hour
        assert forecaster.model._states[key].durations.get("busy") == pytest.approx(
            3600.0
        )

    def test_update_interval_short_durations_filtered(self) -> None:
        """Test that very short intervals are filtered by the model."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 3-second interval (should be filtered)
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 3)

        forecaster.update_interval(start, end, "on")

        # No data should be recorded (filtered as noise)
        assert forecaster.model._states == {}


class TestTimeAwareForecasterPredict:
    """Test TimeAwareForecaster predict method."""

    def test_predict_with_no_data(self) -> None:
        """Test prediction when no data has been learned."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 0))

        assert prediction.state is None
        assert prediction.distribution == {}
        assert prediction.confidence.max_probability == 0
        assert prediction.confidence.entropy_confidence == 0
        assert prediction.confidence.support_time == 0

    def test_predict_after_single_update(self) -> None:
        """Test prediction after single prediction.state update."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))

        assert prediction.state == "on"
        assert prediction.distribution == {"on": 1.0}
        assert prediction.confidence.max_probability == 1.0
        assert prediction.confidence.support_time == 3600.0

    def test_predict_returns_most_likely_state(self) -> None:
        """Test that predict returns the prediction.state with highest probability."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with 75% on, 25% off
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 45),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 45),
            datetime(2024, 1, 1, 11, 0),
            "off",
        )

        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))

        assert prediction.state == "on"
        assert prediction.distribution == pytest.approx({"on": 0.75, "off": 0.25})
        assert prediction.confidence.max_probability == pytest.approx(0.75)

    def test_predict_different_buckets_independent(self) -> None:
        """Test that different time buckets have independent predictions."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Morning: mostly on
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 45),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 45),
            datetime(2024, 1, 1, 11, 0),
            "off",
        )

        # Evening: mostly off
        forecaster.update_interval(
            datetime(2024, 1, 1, 18, 0),
            datetime(2024, 1, 1, 18, 15),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 18, 15),
            datetime(2024, 1, 1, 19, 0),
            "off",
        )

        morning_prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        evening_prediction = forecaster.predict(datetime(2024, 1, 1, 18, 30))

        assert morning_prediction.state == "on"
        assert (
            morning_prediction.distribution["on"]
            > morning_prediction.distribution["off"]
        )
        assert evening_prediction.state == "off"
        assert (
            evening_prediction.distribution["off"]
            > evening_prediction.distribution["on"]
        )

    def test_predict_with_composite_indexer(self) -> None:
        """Test prediction with multi-dimensional indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Weekday morning: on
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),  # Monday
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        # Weekend morning: off
        forecaster.update_interval(
            datetime(2024, 1, 6, 10, 0),  # Saturday
            datetime(2024, 1, 6, 11, 0),
            "off",
        )

        weekday_prediction = forecaster.predict(datetime(2024, 1, 8, 10, 30))  # Monday
        weekend_prediction = forecaster.predict(
            datetime(2024, 1, 13, 10, 30)
        )  # Saturday

        assert weekday_prediction.state == "on"
        assert weekend_prediction.state == "off"

    def test_predict_uses_correct_bucket(self) -> None:
        """Test that predict uses the correct time bucket for the query time."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Only train 10:00-11:00 bucket
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        # Predict in trained bucket
        prediction1 = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction1.state == "on"
        assert prediction1.distribution == {"on": 1.0}

        # Predict in different day, same hour (should use same bucket)
        prediction2 = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction2.state == "on"
        assert prediction2.distribution == {"on": 1.0}

        # Predict in untrained bucket
        prediction3 = forecaster.predict(datetime(2024, 1, 1, 14, 30))
        assert prediction3.state is None
        assert prediction3.distribution == {}


class TestTimeAwareForecasterIntegration:
    """Test integration scenarios and workflows."""

    def test_complete_workflow(self) -> None:
        """Test complete workflow: initialize, train, predict."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with historical data
        for day in range(1, 8):  # Week of data
            # Morning: on
            forecaster.update_interval(
                datetime(2024, 1, day, 8, 0),
                datetime(2024, 1, day, 12, 0),
                "on",
            )
            # Afternoon: off
            forecaster.update_interval(
                datetime(2024, 1, day, 12, 0),
                datetime(2024, 1, day, 18, 0),
                "off",
            )

        # Predict for new day
        morning_prediction = forecaster.predict(datetime(2024, 1, 8, 10, 0))
        afternoon_prediction = forecaster.predict(datetime(2024, 1, 8, 14, 0))

        assert morning_prediction.state == "on"
        assert morning_prediction.distribution == {"on": 1.0}
        assert morning_prediction.confidence.support_time > 0

        assert afternoon_prediction.state == "off"
        assert afternoon_prediction.distribution == {"off": 1.0}
        assert afternoon_prediction.confidence.support_time > 0

    def test_incremental_learning(self) -> None:
        """Test that forecaster learns incrementally as data arrives."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Initial prediction: no data
        prediction1 = forecaster.predict(datetime(2024, 1, 1, 10, 0))
        assert prediction1.state is None

        # Add first observation
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        prediction2 = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction2.state == "on"
        assert prediction2.distribution == {"on": 1.0}

        # Add conflicting observation
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )
        prediction3 = forecaster.predict(datetime(2024, 1, 3, 10, 30))
        assert prediction3.distribution == pytest.approx({"on": 0.5, "off": 0.5})

    def test_multi_state_learning(self) -> None:
        """Test learning with more than two states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with three states
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 30),
            "heating",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 10, 45),
            "cooling",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 45),
            datetime(2024, 1, 1, 11, 0),
            "idle",
        )

        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))

        assert prediction.state == "heating"  # Longest duration
        expected = {"heating": 0.5, "cooling": 0.25, "idle": 0.25}
        assert prediction.distribution == pytest.approx(expected)

    def test_long_interval_spanning_many_buckets(self) -> None:
        """Test handling of very long intervals."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 12-hour interval
        forecaster.update_interval(
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 12, 0),
            "on",
        )

        # Check multiple buckets received updates
        for hour in [2, 5, 8, 11]:
            prediction = forecaster.predict(datetime(2024, 1, 1, hour, 0))
            assert prediction.state == "on"
            assert prediction.confidence.support_time == 3600.0

    def test_pattern_learning_with_composite_indexer(self) -> None:
        """Test learning complex patterns with multiple time dimensions."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Weekdays (Mon-Fri): busy in morning
        for day in range(5):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 9, 0),
                datetime(2024, 1, 1 + day, 10, 0),
                "busy",
            )

        # Weekend (Sat-Sun): idle in morning
        for day in range(5, 7):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 9, 0),
                datetime(2024, 1, 1 + day, 10, 0),
                "idle",
            )

        # Predict for next week
        monday_prediction = forecaster.predict(datetime(2024, 1, 8, 9, 30))  # Monday
        saturday_prediction = forecaster.predict(
            datetime(2024, 1, 13, 9, 30)
        )  # Saturday

        assert monday_prediction.state == "busy"
        assert saturday_prediction.state == "idle"

    def test_confidence_increases_with_data(self) -> None:
        """Test that confidence metrics improve with more training data."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Small amount of data
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 10),
            "on",
        )
        prediction1 = forecaster.predict(datetime(2024, 1, 1, 10, 5))

        # More data
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "on",
        )
        prediction2 = forecaster.predict(datetime(2024, 1, 3, 10, 5))

        assert prediction2.confidence.support_time > prediction1.confidence.support_time

    def test_boundary_alignment(self) -> None:
        """Test that intervals are correctly split at bucket boundaries."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Interval exactly aligned with boundaries
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 12, 0),
            "on",
        )

        # Each bucket should have exactly 1 hour
        for hour in [10, 11]:
            key = indexer.key(datetime(2024, 1, 1, hour, 0))
            assert forecaster.model._states[key].durations.get("on") == pytest.approx(
                3600.0
            )
