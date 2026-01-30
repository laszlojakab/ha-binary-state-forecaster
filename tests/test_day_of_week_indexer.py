"""Tests for DayOfWeekIndexer class."""

from datetime import datetime, timedelta

from custom_components.discrete_state_forecaster.model.time_indexers.day_of_week_indexer import (
    DayOfWeekIndexer,
)


class TestDayOfWeekIndexerInitialization:
    """Test DayOfWeekIndexer initialization."""

    def test_initialization(self) -> None:
        """Test indexer initializes with correct name."""
        indexer = DayOfWeekIndexer()
        assert indexer.name == "weekday"


class TestDayOfWeekIndexerKey:
    """Test DayOfWeekIndexer key calculation."""

    def test_key_monday(self) -> None:
        """Test key for Monday."""
        indexer = DayOfWeekIndexer()
        # January 26, 2026 is a Monday
        ts = datetime(2026, 1, 26, 12, 0, 0)
        assert indexer.key(ts) == 0

    def test_key_tuesday(self) -> None:
        """Test key for Tuesday."""
        indexer = DayOfWeekIndexer()
        # January 27, 2026 is a Tuesday
        ts = datetime(2026, 1, 27, 12, 0, 0)
        assert indexer.key(ts) == 1

    def test_key_wednesday(self) -> None:
        """Test key for Wednesday."""
        indexer = DayOfWeekIndexer()
        # January 28, 2026 is a Wednesday
        ts = datetime(2026, 1, 28, 12, 0, 0)
        assert indexer.key(ts) == 2

    def test_key_thursday(self) -> None:
        """Test key for Thursday."""
        indexer = DayOfWeekIndexer()
        # January 29, 2026 is a Thursday
        ts = datetime(2026, 1, 29, 12, 0, 0)
        assert indexer.key(ts) == 3

    def test_key_friday(self) -> None:
        """Test key for Friday."""
        indexer = DayOfWeekIndexer()
        # January 30, 2026 is a Friday
        ts = datetime(2026, 1, 30, 12, 0, 0)
        assert indexer.key(ts) == 4

    def test_key_saturday(self) -> None:
        """Test key for Saturday."""
        indexer = DayOfWeekIndexer()
        # January 31, 2026 is a Saturday
        ts = datetime(2026, 1, 31, 12, 0, 0)
        assert indexer.key(ts) == 5

    def test_key_sunday(self) -> None:
        """Test key for Sunday."""
        indexer = DayOfWeekIndexer()
        # February 1, 2026 is a Sunday
        ts = datetime(2026, 2, 1, 12, 0, 0)
        assert indexer.key(ts) == 6

    def test_key_same_weekday_different_weeks(self) -> None:
        """Test that same weekday returns same key across different weeks."""
        indexer = DayOfWeekIndexer()

        # All Mondays should have key 0
        monday1 = datetime(2026, 1, 26, 10, 0, 0)  # Monday
        monday2 = datetime(2026, 2, 2, 15, 30, 0)  # Monday, different week
        monday3 = datetime(2026, 3, 9, 8, 45, 0)  # Monday, different month

        assert indexer.key(monday1) == indexer.key(monday2) == indexer.key(monday3) == 0

    def test_key_ignores_time_of_day(self) -> None:
        """Test that key only depends on date, not time of day."""
        indexer = DayOfWeekIndexer()

        # Same day, different times
        ts1 = datetime(2026, 1, 26, 0, 0, 0)  # Monday midnight
        ts2 = datetime(2026, 1, 26, 12, 0, 0)  # Monday noon
        ts3 = datetime(2026, 1, 26, 23, 59, 59)  # Monday almost midnight

        assert indexer.key(ts1) == indexer.key(ts2) == indexer.key(ts3) == 0

    def test_key_full_week_sequence(self) -> None:
        """Test a complete week sequence."""
        indexer = DayOfWeekIndexer()

        base_date = datetime(2026, 1, 26, 12, 0, 0)  # Monday noon
        expected_keys = [0, 1, 2, 3, 4, 5, 6]  # Mon-Sun

        for day_offset, expected_key in enumerate(expected_keys):
            ts = base_date + timedelta(days=day_offset)
            assert indexer.key(ts) == expected_key


class TestDayOfWeekIndexerNextBoundary:
    """Test DayOfWeekIndexer next_boundary calculation."""

    def test_next_boundary_at_midnight(self) -> None:
        """Test next_boundary when timestamp is at midnight."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 1, 26, 0, 0, 0)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 27, 0, 0, 0)

    def test_next_boundary_at_noon(self) -> None:
        """Test next_boundary when timestamp is at noon."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 1, 26, 12, 0, 0)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 27, 0, 0, 0)

    def test_next_boundary_near_midnight(self) -> None:
        """Test next_boundary when timestamp is near end of day."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 1, 26, 23, 59, 59, 999999)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 27, 0, 0, 0)

    def test_next_boundary_weekday_to_weekday(self) -> None:
        """Test next_boundary from weekday to next weekday."""
        indexer = DayOfWeekIndexer()
        # Monday to Tuesday
        ts = datetime(2026, 1, 26, 15, 30, 0)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 27, 0, 0, 0)
        assert indexer.key(next_bound) == 1  # Tuesday

    def test_next_boundary_friday_to_saturday(self) -> None:
        """Test next_boundary from Friday to Saturday."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 1, 30, 18, 0, 0)  # Friday
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 31, 0, 0, 0)  # Saturday
        assert indexer.key(next_bound) == 5

    def test_next_boundary_saturday_to_sunday(self) -> None:
        """Test next_boundary from Saturday to Sunday."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 1, 31, 20, 0, 0)  # Saturday
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 2, 1, 0, 0, 0)  # Sunday
        assert indexer.key(next_bound) == 6

    def test_next_boundary_sunday_to_monday(self) -> None:
        """Test next_boundary from Sunday to Monday (week wrap)."""
        indexer = DayOfWeekIndexer()
        ts = datetime(2026, 2, 1, 21, 0, 0)  # Sunday
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 2, 2, 0, 0, 0)  # Monday
        assert indexer.key(next_bound) == 0

    def test_next_boundary_month_transition(self) -> None:
        """Test next_boundary across month boundary."""
        indexer = DayOfWeekIndexer()
        # January 31, 2026
        ts = datetime(2026, 1, 31, 14, 0, 0)
        next_bound = indexer.next_boundary(ts)

        # Should be February 1, 2026 at midnight
        assert next_bound == datetime(2026, 2, 1, 0, 0, 0)
        assert next_bound.month == 2

    def test_next_boundary_year_transition(self) -> None:
        """Test next_boundary across year boundary."""
        indexer = DayOfWeekIndexer()
        # December 31, 2026
        ts = datetime(2026, 12, 31, 23, 0, 0)
        next_bound = indexer.next_boundary(ts)

        # Should be January 1, 2027 at midnight
        assert next_bound == datetime(2027, 1, 1, 0, 0, 0)
        assert next_bound.year == 2027

    def test_next_boundary_always_midnight(self) -> None:
        """Test that next_boundary always returns midnight."""
        indexer = DayOfWeekIndexer()

        test_times = [
            datetime(2026, 1, 26, 1, 15, 30, 123456),
            datetime(2026, 1, 26, 9, 45, 12, 999999),
            datetime(2026, 1, 26, 17, 30, 0, 500000),
            datetime(2026, 1, 26, 23, 59, 59, 999999),
        ]

        for ts in test_times:
            next_bound = indexer.next_boundary(ts)
            assert next_bound.hour == 0
            assert next_bound.minute == 0
            assert next_bound.second == 0
            assert next_bound.microsecond == 0

    def test_next_boundary_sequence(self) -> None:
        """Test a sequence of next_boundary calls."""
        indexer = DayOfWeekIndexer()

        ts = datetime(2026, 1, 26, 12, 0, 0)  # Monday
        expected_boundaries = [
            datetime(2026, 1, 27, 0, 0, 0),  # Tuesday
            datetime(2026, 1, 28, 0, 0, 0),  # Wednesday
            datetime(2026, 1, 29, 0, 0, 0),  # Thursday
            datetime(2026, 1, 30, 0, 0, 0),  # Friday
        ]

        for expected in expected_boundaries:
            ts = indexer.next_boundary(ts)
            assert ts == expected


class TestDayOfWeekIndexerEdgeCases:
    """Test edge cases for DayOfWeekIndexer."""

    def test_leap_year_february_29(self) -> None:
        """Test handling of leap year date."""
        indexer = DayOfWeekIndexer()
        # February 29, 2024 is a leap year Thursday
        ts = datetime(2024, 2, 29, 12, 0, 0)

        assert indexer.key(ts) == 3  # Thursday
        next_bound = indexer.next_boundary(ts)
        assert next_bound == datetime(2024, 3, 1, 0, 0, 0)

    def test_key_consistency_across_boundaries(self) -> None:
        """Test that keys change correctly across boundaries."""
        indexer = DayOfWeekIndexer()

        # Monday evening
        ts1 = datetime(2026, 1, 26, 23, 59, 59)
        key1 = indexer.key(ts1)

        # Get boundary (Tuesday midnight)
        ts2 = indexer.next_boundary(ts1)
        key2 = indexer.key(ts2)

        # Keys should differ by 1 (or wrap from 6 to 0)
        assert key1 == 0  # Monday
        assert key2 == 1  # Tuesday

    def test_same_time_different_years(self) -> None:
        """Test same date/time in different years may have different weekdays."""
        indexer = DayOfWeekIndexer()

        # January 26 in different years
        ts_2026 = datetime(2026, 1, 26, 12, 0, 0)  # Monday
        ts_2027 = datetime(2027, 1, 26, 12, 0, 0)  # Tuesday
        ts_2028 = datetime(2028, 1, 26, 12, 0, 0)  # Wednesday

        assert indexer.key(ts_2026) == 0  # Monday
        assert indexer.key(ts_2027) == 1  # Tuesday
        assert indexer.key(ts_2028) == 2  # Wednesday
