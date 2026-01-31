"""
Comprehensive tests for StateTracker.

Tests cover:
- Initialization with various forecaster configurations
- First update behavior (no interval recorded)
- State transitions and interval tracking
- Timestamp edge cases (equal, backward, extreme)
- State type variety (string, int, bool, tuple)
- Same state accumulation
- Forecaster integration with different indexers
- Realistic workflow patterns
"""

from datetime import datetime, timedelta

from custom_components.discrete_state_forecaster.model.state_tracker import (
    StateTracker,
)
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.day_of_week_indexer import (
    DayOfWeekIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.month_indexer import (
    MonthIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestInitialization:
    """Tests for StateTracker initialization."""

    def test_init_with_simple_forecaster(self) -> None:
        """Test initialization with basic forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        assert tracker.forecaster is forecaster
        assert tracker.last_state is None
        assert tracker.last_ts is None

    def test_init_with_composite_indexer(self) -> None:
        """Test initialization with multi-dimensional forecaster."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        assert tracker.forecaster is forecaster
        assert tracker.last_state is None
        assert tracker.last_ts is None

    def test_init_with_custom_half_life(self) -> None:
        """Test initialization with forecaster using custom half_life."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, half_life=1800.0)
        tracker = StateTracker(forecaster)

        assert tracker.forecaster is forecaster


class TestFirstUpdate:
    """Tests for first update behavior."""

    def test_first_update_records_state(self) -> None:
        """Test that first update records state without interval."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        ts = datetime(2024, 1, 1, 10, 0)
        tracker.update(ts, "on")

        assert tracker.last_state == "on"
        assert tracker.last_ts == ts

    def test_first_update_no_forecaster_training(self) -> None:
        """Test that first update doesn't train forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "on")

        # Forecaster should have no data yet
        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        # Empty model returns None for best_state
        assert prediction.state is None

    def test_first_update_various_state_types(self) -> None:
        """Test first update with different state types."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # String state
        tracker1 = StateTracker(forecaster)
        tracker1.update(datetime(2024, 1, 1, 10, 0), "active")
        assert tracker1.last_state == "active"

        # Integer state
        tracker2 = StateTracker(forecaster)
        tracker2.update(datetime(2024, 1, 1, 10, 0), 1)
        assert tracker2.last_state == 1

        # Boolean state
        tracker3 = StateTracker(forecaster)
        tracker3.update(datetime(2024, 1, 1, 10, 0), True)
        assert tracker3.last_state is True


class TestStateTransitions:
    """Tests for state transition tracking."""

    def test_second_update_creates_interval(self) -> None:
        """Test that second update creates interval and trains forecaster."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 11, 0)

        tracker.update(start, "on")
        tracker.update(end, "off")

        # Verify state updated
        assert tracker.last_state == "off"
        assert tracker.last_ts == end

        # Verify forecaster learned the "on" interval
        prediction = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction.state == "on"

    def test_multiple_transitions(self) -> None:
        """Test multiple state transitions accumulate correctly."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track: on -> off -> on -> idle
        tracker.update(datetime(2024, 1, 1, 9, 0), "on")
        tracker.update(datetime(2024, 1, 1, 10, 0), "off")
        tracker.update(datetime(2024, 1, 1, 11, 0), "on")
        tracker.update(datetime(2024, 1, 1, 13, 0), "idle")

        assert tracker.last_state == "idle"
        assert tracker.last_ts == datetime(2024, 1, 1, 13, 0)

        # Verify forecaster learned patterns
        # At 9:30 should predict "on"
        pred1 = forecaster.predict(datetime(2024, 1, 1, 9, 30))
        assert pred1.state == "on"

        # At 10:30 should predict "off"
        pred2 = forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert pred2.state == "off"

    def test_alternating_states(self) -> None:
        """Test rapid alternating between two states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        base = datetime(2024, 1, 1, 10, 0)
        for i in range(10):
            state = "on" if i % 2 == 0 else "off"
            tracker.update(base + timedelta(minutes=i * 5), state)

        expected_state = "on" if 9 % 2 == 0 else "off"
        assert tracker.last_state == expected_state


class TestIntervalCalculation:
    """Tests for interval calculation accuracy."""

    def test_short_interval(self) -> None:
        """Test tracking very short intervals."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 3)  # 3 seconds

        tracker.update(start, "on")
        tracker.update(end, "off")

        # Very short interval gets filtered by MIN_DURATION (5 sec)
        # but tracker should still update state
        assert tracker.last_state == "off"
        assert tracker.last_ts == end

    def test_long_interval(self) -> None:
        """Test tracking long intervals (hours)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 18, 0)  # 8 hours

        tracker.update(start, "on")
        tracker.update(end, "off")

        assert tracker.last_state == "off"
        # Forecaster should have learned the long "on" interval
        prediction = forecaster.predict(datetime(2024, 1, 1, 14, 0))
        assert prediction.state == "on"

    def test_interval_spanning_midnight(self) -> None:
        """Test interval that crosses midnight boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 1, 23, 0)
        end = datetime(2024, 1, 2, 1, 0)  # Crosses midnight

        tracker.update(start, "sleeping")
        tracker.update(end, "awake")

        assert tracker.last_state == "awake"
        assert tracker.last_ts == end

    def test_interval_spanning_month(self) -> None:
        """Test interval that crosses month boundary."""
        indexer = CompositeIndexer([MonthIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 31, 22, 0)
        end = datetime(2024, 2, 1, 2, 0)  # Crosses month

        tracker.update(start, "active")
        tracker.update(end, "idle")

        assert tracker.last_state == "idle"


class TestTimestampEdgeCases:
    """Tests for timestamp edge cases."""

    def test_equal_timestamps(self) -> None:
        """Test updates with equal timestamps (zero duration)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        ts = datetime(2024, 1, 1, 10, 0)

        tracker.update(ts, "on")
        tracker.update(ts, "off")  # Same timestamp

        # State should update even with zero duration
        assert tracker.last_state == "off"
        assert tracker.last_ts == ts

    def test_backward_timestamp(self) -> None:
        """Test update with timestamp before previous (backward time)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 11, 0), "on")
        tracker.update(datetime(2024, 1, 1, 10, 0), "off")  # Earlier time

        # Tracker should still update (negative interval filtered by model)
        assert tracker.last_state == "off"
        assert tracker.last_ts == datetime(2024, 1, 1, 10, 0)

    def test_microsecond_precision(self) -> None:
        """Test timestamps with microsecond precision."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 1, 1, 10, 0, 0, 123456)
        end = datetime(2024, 1, 1, 10, 0, 1, 654321)

        tracker.update(start, "on")
        tracker.update(end, "off")

        assert tracker.last_ts == end

    def test_leap_year_date(self) -> None:
        """Test tracking on leap year date (Feb 29)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        start = datetime(2024, 2, 29, 10, 0)  # 2024 is leap year
        end = datetime(2024, 2, 29, 11, 0)

        tracker.update(start, "on")
        tracker.update(end, "off")

        assert tracker.last_state == "off"


class TestStateTypes:
    """Tests for various state types."""

    def test_string_states(self) -> None:
        """Test tracking with string states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "idle")
        tracker.update(datetime(2024, 1, 1, 11, 0), "active")
        tracker.update(datetime(2024, 1, 1, 12, 0), "sleeping")

        assert tracker.last_state == "sleeping"

    def test_integer_states(self) -> None:
        """Test tracking with integer states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), 0)
        tracker.update(datetime(2024, 1, 1, 11, 0), 1)
        tracker.update(datetime(2024, 1, 1, 12, 0), 2)

        assert tracker.last_state == 2

    def test_boolean_states(self) -> None:
        """Test tracking with boolean states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), True)
        tracker.update(datetime(2024, 1, 1, 11, 0), False)
        tracker.update(datetime(2024, 1, 1, 12, 0), True)

        assert tracker.last_state is True

    def test_tuple_states(self) -> None:
        """Test tracking with tuple states (composite states)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), ("mode", "heating"))
        tracker.update(datetime(2024, 1, 1, 11, 0), ("mode", "cooling"))
        tracker.update(datetime(2024, 1, 1, 12, 0), ("mode", "off"))

        assert tracker.last_state == ("mode", "off")

    def test_mixed_state_types(self) -> None:
        """Test tracking with mixed state types in sequence."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), 1)
        tracker.update(datetime(2024, 1, 1, 12, 0), True)

        assert tracker.last_state is True


class TestSameStateAccumulation:
    """Tests for repeated same state behavior."""

    def test_same_state_repeated(self) -> None:
        """Test that repeated same state accumulates duration."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track same state "on" for 3 hours
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "on")
        tracker.update(datetime(2024, 1, 1, 12, 0), "on")
        tracker.update(datetime(2024, 1, 1, 13, 0), "off")

        assert tracker.last_state == "off"

        # Forecaster should have accumulated "on" time
        prediction = forecaster.predict(datetime(2024, 1, 1, 11, 30))
        assert prediction.state == "on"

    def test_same_state_many_updates(self) -> None:
        """Test many rapid updates of same state."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        base = datetime(2024, 1, 1, 10, 0)
        for i in range(20):
            tracker.update(base + timedelta(minutes=i), "constant")

        assert tracker.last_state == "constant"


class TestForecasterIntegration:
    """Tests for integration with different forecaster configurations."""

    def test_with_time_of_day_indexer(self) -> None:
        """Test StateTracker with time-of-day indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track morning "on" pattern
        for day in range(7):
            base_date = datetime(2024, 1, 1 + day, 9, 0)
            tracker.update(base_date, "on")
            tracker.update(base_date + timedelta(hours=1), "off")

        # Predict at 9:30 should be "on"
        prediction = forecaster.predict(datetime(2024, 1, 8, 9, 30))
        assert prediction.state == "on"

    def test_with_day_of_week_indexer(self) -> None:
        """Test StateTracker with day-of-week indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track weekday vs weekend pattern
        # Mon-Fri: work mode
        for day in range(5):  # Monday to Friday
            base = datetime(2024, 1, 1 + day, 9, 0)
            tracker.update(base, "work")
            tracker.update(base + timedelta(hours=8), "home")

        # Sat-Sun: relax mode
        for day in range(2):  # Saturday and Sunday
            base = datetime(2024, 1, 6 + day, 9, 0)
            tracker.update(base, "relax")
            tracker.update(base + timedelta(hours=8), "home")

        # Predict Monday at 9:30 should be "work"
        monday_pred = forecaster.predict(datetime(2024, 1, 8, 9, 30))
        assert monday_pred.state == "work"

    def test_with_hierarchical_indexer(self) -> None:
        """Test StateTracker with multi-level hierarchical indexer."""
        indexer = CompositeIndexer([MonthIndexer(), DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track complex pattern
        tracker.update(datetime(2024, 1, 1, 10, 0), "winter_weekday_morning")
        tracker.update(datetime(2024, 1, 1, 18, 0), "winter_weekday_evening")
        tracker.update(datetime(2024, 6, 15, 10, 0), "summer_weekday_morning")

        assert tracker.last_state == "summer_weekday_morning"

    def test_with_custom_half_life(self) -> None:
        """Test StateTracker with forecaster using exponential decay."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        # Short half-life means recent data matters more
        forecaster = TimeAwareForecaster(indexer, half_life=1800.0)
        tracker = StateTracker(forecaster)

        # Old pattern: "on" at 10:00
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0), "off")

        # Recent pattern: "idle" at 10:00 (after decay period)
        far_future = datetime(2024, 1, 10, 10, 0)
        tracker.update(far_future, "idle")
        tracker.update(far_future + timedelta(hours=1), "active")

        # Recent observation should dominate due to decay
        assert tracker.last_state == "active"


class TestRealisticWorkflows:
    """Tests with realistic usage patterns."""

    def test_home_automation_daily_pattern(self) -> None:
        """Test realistic home automation pattern."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Simulate week of home/away pattern
        for day in range(7):
            base = datetime(2024, 1, 1 + day, 0, 0)
            # Night: sleeping (0:00-7:00)
            tracker.update(base, "sleeping")
            # Morning: home (7:00-9:00)
            tracker.update(base + timedelta(hours=7), "home")
            # Day: away (9:00-17:00)
            tracker.update(base + timedelta(hours=9), "away")
            # Evening: home (17:00-23:00)
            tracker.update(base + timedelta(hours=17), "home")
            # Night: sleeping (23:00-24:00)
            tracker.update(base + timedelta(hours=23), "sleeping")

        # Predictions should match pattern
        pred_morning = forecaster.predict(datetime(2024, 1, 8, 8, 0))
        assert pred_morning.state == "home"

        pred_midday = forecaster.predict(datetime(2024, 1, 8, 12, 0))
        assert pred_midday.state == "away"

        pred_evening = forecaster.predict(datetime(2024, 1, 8, 20, 0))
        assert pred_evening.state == "home"

    def test_thermostat_control_pattern(self) -> None:
        """Test realistic thermostat state pattern."""
        indexer = CompositeIndexer([TimeOfDayIndexer(30)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Simulate heating/cooling cycles
        for day in range(5):
            base = datetime(2024, 1, 1 + day, 0, 0)
            # Night: idle (0:00-6:00)
            tracker.update(base, "idle")
            # Morning warmup: heating (6:00-8:00)
            tracker.update(base + timedelta(hours=6), "heating")
            # Day: idle (8:00-16:00)
            tracker.update(base + timedelta(hours=8), "idle")
            # Evening warmup: heating (16:00-18:00)
            tracker.update(base + timedelta(hours=16), "heating")
            # Night: idle (18:00-24:00)
            tracker.update(base + timedelta(hours=18), "idle")

        # Should predict heating during warmup periods
        pred = forecaster.predict(datetime(2024, 1, 6, 7, 0))
        assert pred.state == "heating"

    def test_long_term_tracking(self) -> None:
        """Test tracking over extended period (weeks)."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Track 4 weeks of data
        for week in range(4):
            for day in range(7):
                date = datetime(2024, 1, 1 + week * 7 + day, 10, 0)
                state = "work" if day < 5 else "relax"
                tracker.update(date, state)
                tracker.update(date + timedelta(hours=8), "rest")

        # Pattern should be well-established
        weekday_pred = forecaster.predict(datetime(2024, 2, 1, 10, 0))
        # Feb 1, 2024 is Thursday
        assert weekday_pred.state == "work"

    def test_irregular_updates(self) -> None:
        """Test tracking with irregular update intervals."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        # Irregular intervals: 5min, 2hr, 30min, 3hr
        tracker.update(datetime(2024, 1, 1, 10, 0), "on")
        tracker.update(datetime(2024, 1, 1, 10, 5), "off")
        tracker.update(datetime(2024, 1, 1, 12, 5), "on")
        tracker.update(datetime(2024, 1, 1, 12, 35), "off")
        tracker.update(datetime(2024, 1, 1, 15, 35), "on")

        assert tracker.last_state == "on"
        assert tracker.last_ts == datetime(2024, 1, 1, 15, 35)


class TestEdgeCases:
    """Tests for additional edge cases."""

    def test_very_first_update_only(self) -> None:
        """Test tracker with only one update ever."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "single")

        assert tracker.last_state == "single"
        assert tracker.last_ts == datetime(2024, 1, 1, 10, 0)

    def test_extreme_future_timestamp(self) -> None:
        """Test with very far future timestamp."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0), "now")
        tracker.update(datetime(2100, 12, 31, 23, 59), "future")

        assert tracker.last_state == "future"

    def test_transition_at_exact_hour(self) -> None:
        """Test state transition at exact hour boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        tracker.update(datetime(2024, 1, 1, 10, 0, 0), "on")
        tracker.update(datetime(2024, 1, 1, 11, 0, 0), "off")
        tracker.update(datetime(2024, 1, 1, 12, 0, 0), "on")

        assert tracker.last_state == "on"

    def test_rapid_state_changes(self) -> None:
        """Test very rapid state changes (seconds apart)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        tracker = StateTracker(forecaster)

        base = datetime(2024, 1, 1, 10, 0)
        states = ["a", "b", "c", "d", "e"]

        for i, state in enumerate(states):
            tracker.update(base + timedelta(seconds=i), state)

        assert tracker.last_state == "e"
