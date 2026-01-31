"""Tests for CompositeIndexer class."""

from datetime import datetime

import pytest

from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.day_of_week_indexer import (
    DayOfWeekIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestCompositeIndexerInitialization:
    """Test CompositeIndexer initialization."""

    def test_initialization_with_single_indexer(self) -> None:
        """Test composite indexer with single indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(bucket_minutes=30)])
        assert len(indexer.indexers) == 1
        assert isinstance(indexer.indexers[0], TimeOfDayIndexer)

    def test_initialization_with_multiple_indexers(self) -> None:
        """Test composite indexer with multiple indexers."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=30)]
        )
        assert len(indexer.indexers) == 2
        assert isinstance(indexer.indexers[0], DayOfWeekIndexer)
        assert isinstance(indexer.indexers[1], TimeOfDayIndexer)

    def test_initialization_preserves_order(self) -> None:
        """Test that indexer order is preserved."""
        time_idx = TimeOfDayIndexer(bucket_minutes=60)
        day_idx = DayOfWeekIndexer()

        indexer1 = CompositeIndexer([time_idx, day_idx])
        indexer2 = CompositeIndexer([day_idx, time_idx])

        assert indexer1.indexers[0] is time_idx
        assert indexer1.indexers[1] is day_idx
        assert indexer2.indexers[0] is day_idx
        assert indexer2.indexers[1] is time_idx

    def test_initialization_with_empty_list(self) -> None:
        """Test composite indexer with no indexers."""
        indexer = CompositeIndexer([])
        assert len(indexer.indexers) == 0


class TestCompositeIndexerKey:
    """Test CompositeIndexer key generation."""

    def test_key_single_indexer(self) -> None:
        """Test composite key with single indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(bucket_minutes=30)])
        ts = datetime(2026, 1, 26, 14, 30)
        key = indexer.key(ts)

        assert isinstance(key, TimeKey)
        assert len(key) == 1
        assert key.items[0] == ("time_bucket", 29)  # 14:30 -> bucket 29

    def test_key_two_indexers(self) -> None:
        """Test composite key with two indexers."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        # Monday at 14:30
        ts = datetime(2026, 1, 26, 14, 30)
        key = indexer.key(ts)

        assert isinstance(key, TimeKey)
        assert len(key) == 2
        assert key.items[0] == ("weekday", 0)  # Monday
        assert key.items[1] == ("time_bucket", 14)  # Hour 14

    def test_key_structure(self) -> None:
        """Test that key structure is (name, value) pairs."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=30)]
        )
        ts = datetime(2026, 1, 30, 18, 45)  # Friday at 18:45
        key = indexer.key(ts)

        # Each element should be (name, value) tuple
        for element in key.items:
            assert isinstance(element, tuple)
            assert len(element) == 2
            assert isinstance(element[0], str)  # name is string

        assert key == TimeKey((("weekday", 4), ("time_bucket", 37)))

    def test_key_preserves_indexer_order(self) -> None:
        """Test that key preserves the order of indexers."""
        time_idx = TimeOfDayIndexer(bucket_minutes=60)
        day_idx = DayOfWeekIndexer()

        indexer1 = CompositeIndexer([time_idx, day_idx])
        indexer2 = CompositeIndexer([day_idx, time_idx])

        ts = datetime(2026, 1, 26, 14, 30)

        key1 = indexer1.key(ts)
        key2 = indexer2.key(ts)

        # Keys should have same elements but different order
        assert key1.items[0][0] == "time_bucket"
        assert key1.items[1][0] == "weekday"
        assert key2.items[0][0] == "weekday"
        assert key2.items[1][0] == "time_bucket"

    def test_key_different_times_same_day(self) -> None:
        """Test keys for different times on the same day."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        ts1 = datetime(2026, 1, 26, 10, 0)  # Monday 10:00
        ts2 = datetime(2026, 1, 26, 14, 0)  # Monday 14:00

        key1 = indexer.key(ts1)
        key2 = indexer.key(ts2)

        # Same day, different time
        assert key1.items[0] == key2.items[0]  # Same weekday
        assert key1.items[1] != key2.items[1]  # Different time bucket

    def test_key_different_days_same_time(self) -> None:
        """Test keys for different days at the same time."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        ts1 = datetime(2026, 1, 26, 14, 0)  # Monday 14:00
        ts2 = datetime(2026, 1, 27, 14, 0)  # Tuesday 14:00

        key1 = indexer.key(ts1)
        key2 = indexer.key(ts2)

        # Different day, same time
        assert key1.items[0] != key2.items[0]  # Different weekday
        assert key1.items[1] == key2.items[1]  # Same time bucket

    def test_key_completely_different_timestamps(self) -> None:
        """Test keys for completely different timestamps."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=30)]
        )

        ts1 = datetime(2026, 1, 26, 10, 15)  # Monday 10:15
        ts2 = datetime(2026, 1, 30, 18, 45)  # Friday 18:45

        key1 = indexer.key(ts1)
        key2 = indexer.key(ts2)

        # Both dimensions should differ
        assert key1.items[0] != key2.items[0]  # Different weekday
        assert key1.items[1] != key2.items[1]  # Different time bucket
        assert key1 != key2

    def test_key_same_composite_bucket(self) -> None:
        """Test that same composite bucket produces same key."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        # Both are Monday at hour 14
        ts1 = datetime(2026, 1, 26, 14, 15)
        ts2 = datetime(2026, 2, 2, 14, 45)

        key1 = indexer.key(ts1)
        key2 = indexer.key(ts2)

        assert key1 == key2
        assert key1 == TimeKey((("weekday", 0), ("time_bucket", 14)))

    def test_key_empty_indexers(self) -> None:
        """Test key generation with no indexers."""
        indexer = CompositeIndexer([])
        ts = datetime(2026, 1, 26, 14, 30)
        key = indexer.key(ts)

        assert key == TimeKey.GLOBAL
        assert len(key) == 0


class TestCompositeIndexerNextBoundary:
    """Test CompositeIndexer next_boundary calculation."""

    def test_next_boundary_single_indexer(self) -> None:
        """Test next boundary with single indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(bucket_minutes=30)])
        ts = datetime(2026, 1, 26, 14, 15)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 26, 14, 30)

    def test_next_boundary_takes_minimum(self) -> None:
        """Test that next_boundary returns the earliest boundary."""
        # Day boundary at midnight, time boundary at next hour
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        # 14:30 - next hour (15:00) comes before next day (midnight)
        ts = datetime(2026, 1, 26, 14, 30)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 26, 15, 0)

    def test_next_boundary_day_comes_first(self) -> None:
        """Test next boundary when day boundary comes first."""
        # Day boundary at midnight, time boundary every 12 hours
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=12 * 60)]
        )

        # 23:30 - next day (00:00) comes before next 12-hour mark (12:00)
        ts = datetime(2026, 1, 26, 23, 30)
        next_bound = indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 27, 0, 0)

    def test_next_boundary_multiple_indexers_same_time(self) -> None:
        """Test when multiple indexers have the same next boundary."""
        # Both change at midnight
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=24 * 60)]
        )

        ts = datetime(2026, 1, 26, 12, 0)
        next_bound = indexer.next_boundary(ts)

        # Both boundaries are at midnight
        assert next_bound == datetime(2026, 1, 27, 0, 0)

    def test_next_boundary_three_indexers(self) -> None:
        """Test next boundary with three indexers."""
        indexer = CompositeIndexer(
            [
                DayOfWeekIndexer(),  # Changes at midnight
                TimeOfDayIndexer(bucket_minutes=60),  # Changes every hour
                TimeOfDayIndexer(bucket_minutes=15),  # Changes every 15 min
            ]
        )

        ts = datetime(2026, 1, 26, 14, 12)
        next_bound = indexer.next_boundary(ts)

        # 15-minute boundary comes first (14:15)
        assert next_bound == datetime(2026, 1, 26, 14, 15)

    def test_next_boundary_sequence(self) -> None:
        """Test a sequence of next_boundary calls."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        ts = datetime(2026, 1, 26, 22, 30)

        # Next boundary is 23:00 (time bucket)
        ts = indexer.next_boundary(ts)
        assert ts == datetime(2026, 1, 26, 23, 0)

        # Next boundary is 00:00 (both day and time bucket)
        ts = indexer.next_boundary(ts)
        assert ts == datetime(2026, 1, 27, 0, 0)

        # Next boundary is 01:00 (time bucket)
        ts = indexer.next_boundary(ts)
        assert ts == datetime(2026, 1, 27, 1, 0)

    def test_next_boundary_empty_indexers(self) -> None:
        """Test next_boundary with no indexers raises error."""
        indexer = CompositeIndexer([])
        ts = datetime(2026, 1, 26, 14, 30)

        # min() on empty sequence should raise ValueError
        with pytest.raises(ValueError, match="min.*empty"):
            indexer.next_boundary(ts)


class TestCompositeIndexerEdgeCases:
    """Test edge cases for CompositeIndexer."""

    def test_repeated_indexer_types(self) -> None:
        """Test composite with multiple indexers of same type."""
        # Two time-of-day indexers with different granularities
        indexer = CompositeIndexer(
            [TimeOfDayIndexer(bucket_minutes=60), TimeOfDayIndexer(bucket_minutes=15)]
        )

        ts = datetime(2026, 1, 26, 14, 37)
        key = indexer.key(ts)

        # Should have two time_bucket entries
        assert len(key) == 2
        assert key.items[0] == ("time_bucket", 14)  # Hourly
        assert key.items[1] == ("time_bucket", 58)  # 15-minute (14*4 + 2)

    def test_key_boundary_consistency(self) -> None:
        """Test that keys change at boundaries."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        # Just before boundary
        ts1 = datetime(2026, 1, 26, 14, 59, 59)
        key1 = indexer.key(ts1)

        # At boundary
        ts2 = indexer.next_boundary(ts1)
        key2 = indexer.key(ts2)

        # At least one dimension should change
        assert key1 != key2
        # Time bucket should change
        assert key1.items[1] != key2.items[1]

    def test_complex_three_way_composite(self) -> None:
        """Test a complex composite with three different indexers."""
        indexer = CompositeIndexer(
            [
                DayOfWeekIndexer(),
                TimeOfDayIndexer(bucket_minutes=60),
                TimeOfDayIndexer(bucket_minutes=30),
            ]
        )

        ts = datetime(2026, 1, 30, 18, 45)  # Friday 18:45
        key = indexer.key(ts)

        assert len(key) == 3
        assert key.items[0] == ("weekday", 4)  # Friday
        assert key.items[1] == ("time_bucket", 18)  # Hour 18
        assert key.items[2] == ("time_bucket", 37)  # 30-min bucket (18*2 + 1)

    def test_boundary_at_day_and_time_transition(self) -> None:
        """Test boundary when both day and time change simultaneously."""
        indexer = CompositeIndexer(
            [DayOfWeekIndexer(), TimeOfDayIndexer(bucket_minutes=60)]
        )

        # 23:30 on Monday
        ts = datetime(2026, 1, 26, 23, 30)
        key_before = indexer.key(ts)

        # Next boundary is midnight (both change)
        next_ts = indexer.next_boundary(ts)
        assert next_ts == datetime(2026, 1, 27, 0, 0)

        key_after = indexer.key(next_ts)

        # Both dimensions should change
        assert key_before.items[0] != key_after.items[0]  # Monday -> Tuesday
        assert key_before.items[1] != key_after.items[1]  # Hour 23 -> Hour 0
