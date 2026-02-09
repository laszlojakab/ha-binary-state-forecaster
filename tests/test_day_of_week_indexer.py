"""Unit tests for DayOfWeekIndexer."""

from datetime import datetime, timedelta
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.temporal.day_of_week_indexer import (
    DayOfWeekIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestDayOfWeekIndexerBasics:
    """Tests for basic DayOfWeekIndexer functionality."""

    def test_name_attribute(self: Self) -> None:
        """Test that name is set correctly."""
        indexer = DayOfWeekIndexer()
        assert indexer.name == "day_of_week"

    @pytest.mark.asyncio
    async def test_returns_timekey(self: Self) -> None:
        """Test that get_key returns a TimeKey."""
        indexer = DayOfWeekIndexer()
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert isinstance(key, TimeKey)

    @pytest.mark.asyncio
    async def test_timekey_has_one_feature(self: Self) -> None:
        """Test that returned TimeKey has exactly one feature."""
        indexer = DayOfWeekIndexer()
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert len(key) == 1
        assert key.to_tuple() == (("day_of_week", 0),)


class TestDayOfWeekIndexerMappings:
    """Tests for day-of-week to index mappings."""

    @pytest.mark.asyncio
    async def test_monday(self: Self) -> None:
        """Test Monday (January 8, 2024 is Monday)."""
        indexer = DayOfWeekIndexer()
        # January 8, 2024 is Monday
        timestamp = datetime(2024, 1, 8, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 0),)

    @pytest.mark.asyncio
    async def test_tuesday(self: Self) -> None:
        """Test Tuesday."""
        indexer = DayOfWeekIndexer()
        # January 9, 2024 is Tuesday
        timestamp = datetime(2024, 1, 9, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 1),)

    @pytest.mark.asyncio
    async def test_wednesday(self: Self) -> None:
        """Test Wednesday."""
        indexer = DayOfWeekIndexer()
        # January 10, 2024 is Wednesday
        timestamp = datetime(2024, 1, 10, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 2),)

    @pytest.mark.asyncio
    async def test_thursday(self: Self) -> None:
        """Test Thursday."""
        indexer = DayOfWeekIndexer()
        # January 11, 2024 is Thursday
        timestamp = datetime(2024, 1, 11, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 3),)

    @pytest.mark.asyncio
    async def test_friday(self: Self) -> None:
        """Test Friday."""
        indexer = DayOfWeekIndexer()
        # January 12, 2024 is Friday
        timestamp = datetime(2024, 1, 12, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 4),)

    @pytest.mark.asyncio
    async def test_saturday(self: Self) -> None:
        """Test Saturday."""
        indexer = DayOfWeekIndexer()
        # January 13, 2024 is Saturday
        timestamp = datetime(2024, 1, 13, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 5),)

    @pytest.mark.asyncio
    async def test_sunday(self: Self) -> None:
        """Test Sunday."""
        indexer = DayOfWeekIndexer()
        # January 14, 2024 is Sunday
        timestamp = datetime(2024, 1, 14, 10, 30)
        key = await indexer.get_key(timestamp)
        assert key.to_tuple() == (("day_of_week", 6),)


class TestDayOfWeekIndexerIgnoresTime:
    """Tests that time of day is ignored."""

    @pytest.mark.asyncio
    async def test_same_day_different_times(self: Self) -> None:
        """Test that same day with different times produce same key."""
        indexer = DayOfWeekIndexer()
        # January 15, 2024 is Monday
        times = [
            datetime(2024, 1, 15, 0, 0, 0),
            datetime(2024, 1, 15, 6, 30, 15),
            datetime(2024, 1, 15, 12, 30, 45),
            datetime(2024, 1, 15, 18, 15, 30),
            datetime(2024, 1, 15, 23, 59, 59),
        ]
        keys = []
        for ts in times:
            key = await indexer.get_key(ts)
            keys.append(key)
        # All keys should be equal
        assert all(k == keys[0] for k in keys)
        assert keys[0].to_tuple() == (("day_of_week", 0),)

    @pytest.mark.asyncio
    async def test_time_components_ignored(self: Self) -> None:
        """Test that hours, minutes, seconds are ignored."""
        indexer = DayOfWeekIndexer()
        # January 15, 2024 is Monday
        key1 = await indexer.get_key(datetime(2024, 1, 15, 0, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 23, 59, 59))
        assert key1 == key2


class TestDayOfWeekIndexerConsistency:
    """Tests for consistency across different weeks and years."""

    @pytest.mark.asyncio
    async def test_same_weekday_different_weeks(self: Self) -> None:
        """Test that same weekday in different weeks produce same key."""
        indexer = DayOfWeekIndexer()
        mondays = [
            datetime(2024, 1, 8, 10, 30),   # Monday
            datetime(2024, 1, 15, 10, 30),  # Monday
            datetime(2024, 1, 22, 10, 30),  # Monday
            datetime(2024, 1, 29, 10, 30),  # Monday
        ]
        keys = [await indexer.get_key(ts) for ts in mondays]
        assert all(k == (("day_of_week", 0),) for k in [k.to_tuple() for k in keys])

    @pytest.mark.asyncio
    async def test_same_weekday_different_years(self: Self) -> None:
        """Test that same weekday in different years produce same key."""
        indexer = DayOfWeekIndexer()
        fridays = [
            datetime(2023, 1, 6, 10, 30),   # Friday
            datetime(2024, 1, 12, 10, 30),  # Friday
            datetime(2025, 1, 10, 10, 30),  # Friday
        ]
        keys = [await indexer.get_key(ts) for ts in fridays]
        assert all(k == (("day_of_week", 4),) for k in [k.to_tuple() for k in keys])

    @pytest.mark.asyncio
    async def test_consecutive_days_different_keys(self: Self) -> None:
        """Test that consecutive days produce different keys."""
        indexer = DayOfWeekIndexer()
        keys = []
        for i in range(7):
            ts = datetime(2024, 1, 15 + i, 10, 30)
            key = await indexer.get_key(ts)
            keys.append(key)
        # All keys should be different
        unique_keys = set(keys)
        assert len(unique_keys) == 7

    @pytest.mark.asyncio
    async def test_week_cycle_repeats(self: Self) -> None:
        """Test that week cycle repeats after 7 days."""
        indexer = DayOfWeekIndexer()
        start = datetime(2024, 1, 15, 10, 30)  # Monday
        key1 = await indexer.get_key(start)
        key2 = await indexer.get_key(start + timedelta(days=7))
        assert key1 == key2


class TestDayOfWeekIndexerHashability:
    """Tests for hashability and use in collections."""

    @pytest.mark.asyncio
    async def test_keys_are_hashable(self: Self) -> None:
        """Test that returned keys are hashable."""
        indexer = DayOfWeekIndexer()
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        hash_value = hash(key)
        assert isinstance(hash_value, int)

    @pytest.mark.asyncio
    async def test_equal_keys_same_hash(self: Self) -> None:
        """Test that equal keys have equal hashes."""
        indexer = DayOfWeekIndexer()
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 20, 15))
        assert key1 == key2
        assert hash(key1) == hash(key2)

    @pytest.mark.asyncio
    async def test_different_keys_different_hash(self: Self) -> None:
        """Test that different keys have different hashes."""
        indexer = DayOfWeekIndexer()
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 30))  # Monday
        key2 = await indexer.get_key(datetime(2024, 1, 16, 10, 30))  # Tuesday
        assert key1 != key2
        assert hash(key1) != hash(key2)

    @pytest.mark.asyncio
    async def test_usable_as_dict_key(self: Self) -> None:
        """Test that keys can be used as dictionary keys."""
        indexer = DayOfWeekIndexer()
        key_mon = await indexer.get_key(datetime(2024, 1, 8, 10, 30))    # Monday
        key_wed = await indexer.get_key(datetime(2024, 1, 10, 10, 30))   # Wednesday
        key_fri = await indexer.get_key(datetime(2024, 1, 12, 10, 30))   # Friday

        patterns = {
            key_mon: "Monday pattern",
            key_wed: "Wednesday pattern",
            key_fri: "Friday pattern",
        }
        assert patterns[key_mon] == "Monday pattern"
        assert patterns[key_wed] == "Wednesday pattern"
        assert patterns[key_fri] == "Friday pattern"

    @pytest.mark.asyncio
    async def test_usable_in_set(self: Self) -> None:
        """Test that keys can be used in sets."""
        indexer = DayOfWeekIndexer()
        # Same day, different times
        key1 = await indexer.get_key(datetime(2024, 1, 15, 8, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 14, 0))
        key3 = await indexer.get_key(datetime(2024, 1, 16, 10, 0))

        s = {key1, key2, key3}
        assert len(s) == 2  # key1 and key2 are the same


class TestDayOfWeekIndexerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_year_boundary(self: Self) -> None:
        """Test across year boundaries."""
        indexer = DayOfWeekIndexer()
        # December 31, 2023 is Sunday
        end_of_2023 = datetime(2023, 12, 31, 10, 30)
        # January 1, 2024 is Monday
        start_of_2024 = datetime(2024, 1, 1, 10, 30)
        key1 = await indexer.get_key(end_of_2023)
        key2 = await indexer.get_key(start_of_2024)
        assert key1.to_tuple() == (("day_of_week", 6),)  # Sunday
        assert key2.to_tuple() == (("day_of_week", 0),)  # Monday

    @pytest.mark.asyncio
    async def test_leap_year(self: Self) -> None:
        """Test with leap year dates."""
        indexer = DayOfWeekIndexer()
        # February 29, 2024 is Thursday
        leap_day = datetime(2024, 2, 29, 10, 30)
        key = await indexer.get_key(leap_day)
        assert key.to_tuple() == (("day_of_week", 3),)  # Thursday

    @pytest.mark.asyncio
    async def test_far_future_date(self: Self) -> None:
        """Test with far future date."""
        indexer = DayOfWeekIndexer()
        # December 25, 2099 is a Friday
        future = datetime(2099, 12, 25, 10, 30)
        key = await indexer.get_key(future)
        assert key.to_tuple() == (("day_of_week", 4),)


class TestDayOfWeekIndexerValueRange:
    """Tests that values are within expected range."""

    @pytest.mark.asyncio
    async def test_all_values_in_range(self: Self) -> None:
        """Test that all day values are in range 0-6."""
        indexer = DayOfWeekIndexer()
        # Test the 7 days of a week
        for day_offset in range(7):
            ts = datetime(2024, 1, 8 + day_offset, 10, 30)
            key = await indexer.get_key(ts)
            value = key.to_tuple()[0][1]
            assert 0 <= value <= 6

    @pytest.mark.asyncio
    async def test_all_values_represented(self: Self) -> None:
        """Test that all 7 values (0-6) are represented in a week."""
        indexer = DayOfWeekIndexer()
        values = set()
        # January 8-14, 2024 is a complete week starting Monday
        for day_offset in range(7):
            ts = datetime(2024, 1, 8 + day_offset, 10, 30)
            key = await indexer.get_key(ts)
            value = key.to_tuple()[0][1]
            values.add(value)
        assert values == {0, 1, 2, 3, 4, 5, 6}
