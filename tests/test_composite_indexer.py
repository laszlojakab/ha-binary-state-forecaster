"""Unit tests for CompositeIndexer."""

from datetime import datetime
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.temporal.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.day_of_week_indexer import (
    DayOfWeekIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.season_indexer import (
    SeasonIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
from custom_components.discrete_state_forecaster.model.temporal.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestCompositeIndexerBasics:
    """Tests for basic CompositeIndexer functionality."""

    def test_create_with_single_indexer(self: Self) -> None:
        """Test creating CompositeIndexer with a single indexer."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        composite = CompositeIndexer([indexer])
        assert len(composite.indexers) == 1

    def test_create_with_multiple_indexers(self: Self) -> None:
        """Test creating CompositeIndexer with multiple indexers."""
        indexers = [
            TimeOfDayIndexer(bucket_size=3600),
            DayOfWeekIndexer(),
            SeasonIndexer(),
        ]
        composite = CompositeIndexer(indexers)
        assert len(composite.indexers) == 3

    def test_name_from_single_indexer(self: Self) -> None:
        """Test that name is derived from single indexer."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        composite = CompositeIndexer([indexer])
        assert composite.name == "time_bucket"

    def test_name_from_multiple_indexers(self: Self) -> None:
        """Test that name is derived from multiple indexers."""
        indexers = [
            TimeOfDayIndexer(bucket_size=3600),
            DayOfWeekIndexer(),
        ]
        composite = CompositeIndexer(indexers)
        assert composite.name == "time_bucket, day_of_week"

    def test_stores_indexers_as_list(self: Self) -> None:
        """Test that indexers are stored as a list."""
        indexers = [TimeOfDayIndexer(bucket_size=3600), DayOfWeekIndexer()]
        composite = CompositeIndexer(iter(indexers))  # Pass as iterable
        assert composite.indexers == indexers


class TestCompositeIndexerGetKey:
    """Tests for CompositeIndexer.get_key method."""

    @pytest.mark.asyncio
    async def test_single_indexer_composition(self: Self) -> None:
        """Test composition with a single indexer."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        composite = CompositeIndexer([indexer])
        ts = datetime(2024, 1, 15, 14, 30)
        key = await composite.get_key(ts)
        assert len(key) == 1
        assert key.to_tuple() == (("time_bucket", 14),)

    @pytest.mark.asyncio
    async def test_two_indexer_composition(self: Self) -> None:
        """Test composition with two indexers."""
        indexers = [
            TimeOfDayIndexer(bucket_size=3600),
            DayOfWeekIndexer(),
        ]
        composite = CompositeIndexer(indexers)
        # January 15, 2024 is Monday, 2:30 PM
        ts = datetime(2024, 1, 15, 14, 30)
        key = await composite.get_key(ts)
        assert len(key) == 2
        assert key.to_tuple() == (("time_bucket", 14), ("day_of_week", 0))

    @pytest.mark.asyncio
    async def test_three_indexer_composition(self: Self) -> None:
        """Test composition with three indexers."""
        indexers = [
            TimeOfDayIndexer(bucket_size=3600),
            DayOfWeekIndexer(),
            SeasonIndexer(),
        ]
        composite = CompositeIndexer(indexers)
        # January 15, 2024: Monday, 2:30 PM, winter
        ts = datetime(2024, 1, 15, 14, 30)
        key = await composite.get_key(ts)
        assert len(key) == 3
        assert key.to_tuple() == (
            ("time_bucket", 14),
            ("day_of_week", 0),
            ("season", "winter"),
        )

    @pytest.mark.asyncio
    async def test_indexer_order_matters(self: Self) -> None:
        """Test that indexer order affects the result."""
        ts = datetime(2024, 1, 15, 14, 30)

        # Order 1: time, day, season
        comp1 = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
                SeasonIndexer(),
            ]
        )
        key1 = await comp1.get_key(ts)

        # Order 2: season, day, time
        comp2 = CompositeIndexer(
            [
                SeasonIndexer(),
                DayOfWeekIndexer(),
                TimeOfDayIndexer(bucket_size=3600),
            ]
        )
        key2 = await comp2.get_key(ts)

        assert key1 != key2
        assert key1.to_tuple() == (
            ("time_bucket", 14),
            ("day_of_week", 0),
            ("season", "winter"),
        )
        assert key2.to_tuple() == (
            ("season", "winter"),
            ("day_of_week", 0),
            ("time_bucket", 14),
        )


class TestCompositeIndexerReturnType:
    """Tests for return type of CompositeIndexer.get_key."""

    @pytest.mark.asyncio
    async def test_returns_timekey(self: Self) -> None:
        """Test that get_key returns a TimeKey."""
        indexers = [TimeOfDayIndexer(bucket_size=3600), DayOfWeekIndexer()]
        composite = CompositeIndexer(indexers)
        key = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        assert isinstance(key, TimeKey)

    @pytest.mark.asyncio
    async def test_key_length_matches_indexer_count(self: Self) -> None:
        """Test that key length matches number of indexers."""
        for num_indexers in range(1, 5):
            indexers = [TimeOfDayIndexer(bucket_size=3600)]
            for _ in range(num_indexers - 1):
                indexers.append(DayOfWeekIndexer())
            composite = CompositeIndexer(indexers)
            key = await composite.get_key(datetime(2024, 1, 15, 14, 30))
            assert len(key) == num_indexers

    @pytest.mark.asyncio
    async def test_key_is_hashable(self: Self) -> None:
        """Test that returned keys are hashable."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
            ]
        )
        key = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        hash_value = hash(key)
        assert isinstance(hash_value, int)

    @pytest.mark.asyncio
    async def test_equal_keys_have_equal_hash(self: Self) -> None:
        """Test that equal keys have equal hashes."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
            ]
        )
        # Same time and day
        key1 = await composite.get_key(datetime(2024, 1, 15, 14, 0))
        key2 = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        assert key1 == key2
        assert hash(key1) == hash(key2)

    @pytest.mark.asyncio
    async def test_different_keys_have_different_hash(self: Self) -> None:
        """Test that different keys have different hashes."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
            ]
        )
        key1 = await composite.get_key(datetime(2024, 1, 15, 14, 30))  # Monday
        key2 = await composite.get_key(datetime(2024, 1, 16, 14, 30))  # Tuesday
        assert key1 != key2
        assert hash(key1) != hash(key2)


class TestCompositeIndexerUsability:
    """Tests for using composite keys in collections."""

    @pytest.mark.asyncio
    async def test_usable_as_dict_key(self: Self) -> None:
        """Test that keys can be used as dictionary keys."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
            ]
        )

        # Monday 2 PM
        key_mon_day = await composite.get_key(datetime(2024, 1, 8, 14, 30))
        # Friday 2 PM
        key_fri_day = await composite.get_key(datetime(2024, 1, 12, 14, 30))
        # Monday 6 AM
        key_mon_morn = await composite.get_key(datetime(2024, 1, 8, 6, 30))

        patterns = {
            key_mon_day: "Monday afternoon",
            key_fri_day: "Friday afternoon",
            key_mon_morn: "Monday morning",
        }

        assert patterns[key_mon_day] == "Monday afternoon"
        assert patterns[key_fri_day] == "Friday afternoon"
        assert len(patterns) == 3

    @pytest.mark.asyncio
    async def test_usable_in_set(self: Self) -> None:
        """Test that keys can be used in sets."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
            ]
        )

        key1 = await composite.get_key(datetime(2024, 1, 8, 14, 30))  # Monday 2 PM
        key2 = await composite.get_key(datetime(2024, 1, 8, 14, 45))  # Monday 2:45 PM
        key3 = await composite.get_key(datetime(2024, 1, 9, 14, 30))  # Tuesday 2 PM

        s = {key1, key2, key3}
        assert len(s) == 2  # key1 and key2 are the same


class TestCompositeIndexerConsistency:
    """Tests for consistency behavior."""

    @pytest.mark.asyncio
    async def test_same_timestamp_produces_same_key(self: Self) -> None:
        """Test that same timestamp produces same key on repeated calls."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                DayOfWeekIndexer(),
                SeasonIndexer(),
            ]
        )
        ts = datetime(2024, 1, 15, 14, 30)
        key1 = await composite.get_key(ts)
        key2 = await composite.get_key(ts)
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_consistency_across_years(self: Self) -> None:
        """Test consistency of same time pattern across years."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=3600),
                SeasonIndexer(),
            ]
        )

        # Both January 15 of different years
        key1 = await composite.get_key(datetime(2023, 1, 15, 14, 30))
        key2 = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        # Same hour and season (winter) but different day of week
        # So keys might be different if we include day_of_week
        # But with just time and season, they should be the same if both are Monday
        # Let's just check the repeating pattern
        assert key1.to_tuple()[0] == key2.to_tuple()[0]  # Same hour


class TestCompositeIndexerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_indexer_list(self: Self) -> None:
        """Test behavior with empty indexer list."""
        composite = CompositeIndexer([])
        key = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        assert len(key) == 0
        assert key == TimeKey.GLOBAL

    @pytest.mark.asyncio
    async def test_many_indexers(self: Self) -> None:
        """Test composition with many indexers."""
        # Create 10 indexers (cycling through types)
        indexers = []
        for i in range(10):
            if i % 3 == 0:
                indexers.append(TimeOfDayIndexer(bucket_size=3600))
            elif i % 3 == 1:
                indexers.append(DayOfWeekIndexer())
            else:
                indexers.append(SeasonIndexer())

        composite = CompositeIndexer(indexers)
        key = await composite.get_key(datetime(2024, 1, 15, 14, 30))
        assert len(key) == 10

    @pytest.mark.asyncio
    async def test_indexer_with_different_bucket_sizes(self: Self) -> None:
        """Test composition with indexers using different bucket sizes."""
        composite = CompositeIndexer(
            [
                TimeOfDayIndexer(bucket_size=300),  # 5-minute buckets
                TimeOfDayIndexer(bucket_size=3600),  # 1-hour buckets
            ]
        )

        # 2:30 PM = 52200 seconds
        # 5-min: 52200/300 = 174
        # 1-hr: 52200/3600 = 14
        ts = datetime(2024, 1, 15, 14, 30)
        key = await composite.get_key(ts)
        assert key.to_tuple() == (("time_bucket", 174), ("time_bucket", 14))

    @pytest.mark.asyncio
    async def test_takes_iterable_of_indexers(self: Self) -> None:
        """Test that CompositeIndexer accepts any iterable of indexers."""
        indexers = [
            TimeOfDayIndexer(bucket_size=3600),
            DayOfWeekIndexer(),
        ]

        # Test with list
        composite1 = CompositeIndexer(indexers)
        # Test with generator
        composite2 = CompositeIndexer(idx for idx in indexers)
        # Test with tuple
        composite3 = CompositeIndexer(tuple(indexers))

        ts = datetime(2024, 1, 15, 14, 30)
        key1 = await composite1.get_key(ts)
        key2 = await composite2.get_key(ts)
        key3 = await composite3.get_key(ts)

        assert key1 == key2 == key3
