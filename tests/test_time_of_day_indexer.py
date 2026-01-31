"""Tests for TimeOfDayIndexer class."""

import pytest
from datetime import datetime

from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestTimeOfDayIndexerInitialization:
    """Test TimeOfDayIndexer initialization."""

    @pytest.mark.asyncio


    async def test_initialization(self) -> None:
        """Test indexer initializes with bucket size."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        assert indexer.bucket == 30
        assert indexer.name == "time_bucket"

    @pytest.mark.asyncio


    async def test_initialization_different_buckets(self) -> None:
        """Test indexer initializes with different bucket sizes."""
        indexer_15 = TimeOfDayIndexer(bucket_minutes=15)
        assert indexer_15.bucket == 15

        indexer_60 = TimeOfDayIndexer(bucket_minutes=60)
        assert indexer_60.bucket == 60

        indexer_5 = TimeOfDayIndexer(bucket_minutes=5)
        assert indexer_5.bucket == 5


class TestTimeOfDayIndexerKey:
    """Test TimeOfDayIndexer key calculation."""

    @pytest.mark.asyncio


    async def test_key_midnight(self) -> None:
        """Test key at midnight."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 0, 0, 0)
        assert await indexer.key(ts) == 0

    @pytest.mark.asyncio


    async def test_key_first_bucket(self) -> None:
        """Test key in first bucket."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 0, 15, 0)
        assert await indexer.key(ts) == 0

    @pytest.mark.asyncio


    async def test_key_second_bucket(self) -> None:
        """Test key in second bucket."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 0, 30, 0)
        assert await indexer.key(ts) == 1

    @pytest.mark.asyncio


    async def test_key_various_times(self) -> None:
        """Test key calculation for various times."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        # 6:00 AM -> bucket 12 (6 * 60 / 30 = 12)
        assert await indexer.key(datetime(2026, 1, 29, 6, 0, 0)) == 12

        # 6:29 AM -> still bucket 12
        assert await indexer.key(datetime(2026, 1, 29, 6, 29, 0)) == 12

        # 6:30 AM -> bucket 13
        assert await indexer.key(datetime(2026, 1, 29, 6, 30, 0)) == 13

        # 12:00 PM (noon) -> bucket 24 (12 * 60 / 30 = 24)
        assert await indexer.key(datetime(2026, 1, 29, 12, 0, 0)) == 24

        # 23:59 PM -> bucket 47 (23 * 60 + 59 = 1439, 1439 // 30 = 47)
        assert await indexer.key(datetime(2026, 1, 29, 23, 59, 0)) == 47

    @pytest.mark.asyncio


    async def test_key_15_minute_buckets(self) -> None:
        """Test key calculation with 15-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_minutes=15)

        # 6:00 AM -> bucket 24 (6 * 60 / 15 = 24)
        assert await indexer.key(datetime(2026, 1, 29, 6, 0, 0)) == 24

        # 6:14 AM -> still bucket 24
        assert await indexer.key(datetime(2026, 1, 29, 6, 14, 0)) == 24

        # 6:15 AM -> bucket 25
        assert await indexer.key(datetime(2026, 1, 29, 6, 15, 0)) == 25

        # 6:30 AM -> bucket 26
        assert await indexer.key(datetime(2026, 1, 29, 6, 30, 0)) == 26

        # 6:45 AM -> bucket 27
        assert await indexer.key(datetime(2026, 1, 29, 6, 45, 0)) == 27

    @pytest.mark.asyncio


    async def test_key_60_minute_buckets(self) -> None:
        """Test key calculation with 60-minute (hourly) buckets."""
        indexer = TimeOfDayIndexer(bucket_minutes=60)

        # 6:00 AM -> bucket 6
        assert await indexer.key(datetime(2026, 1, 29, 6, 0, 0)) == 6

        # 6:59 AM -> still bucket 6
        assert await indexer.key(datetime(2026, 1, 29, 6, 59, 0)) == 6

        # 7:00 AM -> bucket 7
        assert await indexer.key(datetime(2026, 1, 29, 7, 0, 0)) == 7

        # 23:00 PM -> bucket 23
        assert await indexer.key(datetime(2026, 1, 29, 23, 0, 0)) == 23

    @pytest.mark.asyncio


    async def test_key_ignores_date(self) -> None:
        """Test that key only depends on time, not date."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        # Same time, different dates should give same key
        ts1 = datetime(2026, 1, 29, 14, 30, 0)
        ts2 = datetime(2025, 12, 25, 14, 30, 0)
        ts3 = datetime(2027, 6, 15, 14, 30, 0)

        assert await indexer.key(ts1) == await indexer.key(ts2) == await indexer.key(ts3)

    @pytest.mark.asyncio


    async def test_key_ignores_seconds(self) -> None:
        """Test that key only depends on hours and minutes, not seconds."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        # Same hour and minute, different seconds should give same key
        ts1 = datetime(2026, 1, 29, 14, 30, 0)
        ts2 = datetime(2026, 1, 29, 14, 30, 30)
        ts3 = datetime(2026, 1, 29, 14, 30, 59)

        assert await indexer.key(ts1) == await indexer.key(ts2) == await indexer.key(ts3)


class TestTimeOfDayIndexerNextBoundary:
    """Test TimeOfDayIndexer next_boundary calculation."""

    @pytest.mark.asyncio


    async def test_next_boundary_at_bucket_start(self) -> None:
        """Test next_boundary when timestamp is at bucket start."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 6, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 29, 6, 30, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_middle_of_bucket(self) -> None:
        """Test next_boundary when timestamp is in middle of bucket."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 6, 15, 30)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 29, 6, 30, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_near_end_of_bucket(self) -> None:
        """Test next_boundary when timestamp is near end of bucket."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)
        ts = datetime(2026, 1, 29, 6, 29, 59)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 1, 29, 6, 30, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_15_minute_buckets(self) -> None:
        """Test next_boundary with 15-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_minutes=15)

        # From 6:00 -> 6:15
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 0, 0)) == datetime(
            2026, 1, 29, 6, 15, 0
        )

        # From 6:07 -> 6:15
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 7, 0)) == datetime(
            2026, 1, 29, 6, 15, 0
        )

        # From 6:15 -> 6:30
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 15, 0)) == datetime(
            2026, 1, 29, 6, 30, 0
        )

        # From 6:30 -> 6:45
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 30, 0)) == datetime(
            2026, 1, 29, 6, 45, 0
        )

    @pytest.mark.asyncio


    async def test_next_boundary_60_minute_buckets(self) -> None:
        """Test next_boundary with 60-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_minutes=60)

        # From 6:00 -> 7:00
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 0, 0)) == datetime(
            2026, 1, 29, 7, 0, 0
        )

        # From 6:30 -> 7:00
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 30, 0)) == datetime(
            2026, 1, 29, 7, 0, 0
        )

        # From 6:59 -> 7:00
        assert await indexer.next_boundary(datetime(2026, 1, 29, 6, 59, 59)) == datetime(
            2026, 1, 29, 7, 0, 0
        )

    @pytest.mark.asyncio


    async def test_next_boundary_midnight_wrap(self) -> None:
        """Test next_boundary wrapping to next day after midnight."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        # From 23:30 -> next day 00:00
        ts = datetime(2026, 1, 29, 23, 30, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 1, 30, 0, 0, 0)

        # From 23:45 -> next day 00:00
        ts = datetime(2026, 1, 29, 23, 45, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 1, 30, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_removes_seconds(self) -> None:
        """Test that next_boundary always has zero seconds and microseconds."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        ts = datetime(2026, 1, 29, 6, 15, 45, 123456)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound.second == 0
        assert next_bound.microsecond == 0

    @pytest.mark.asyncio


    async def test_next_boundary_preserves_date(self) -> None:
        """Test that next_boundary preserves the date (unless wrapping to next day)."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        ts = datetime(2026, 1, 29, 14, 15, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound.year == 2026
        assert next_bound.month == 1
        assert next_bound.day == 29

    @pytest.mark.asyncio


    async def test_next_boundary_sequence(self) -> None:
        """Test a sequence of next_boundary calls."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        ts = datetime(2026, 1, 29, 0, 0, 0)
        expected_boundaries = [
            datetime(2026, 1, 29, 0, 30, 0),
            datetime(2026, 1, 29, 1, 0, 0),
            datetime(2026, 1, 29, 1, 30, 0),
            datetime(2026, 1, 29, 2, 0, 0),
        ]

        for expected in expected_boundaries:
            ts = await indexer.next_boundary(ts)
            assert ts == expected


class TestTimeOfDayIndexerEdgeCases:
    """Test edge cases for TimeOfDayIndexer."""

    @pytest.mark.asyncio


    async def test_single_minute_buckets(self) -> None:
        """Test with 1-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_minutes=1)

        # 6:15 -> bucket 375 (6 * 60 + 15 = 375)
        assert await indexer.key(datetime(2026, 1, 29, 6, 15, 0)) == 375

        # Next boundary should be 6:16
        ts = datetime(2026, 1, 29, 6, 15, 30)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 1, 29, 6, 16, 0)

    @pytest.mark.asyncio


    async def test_large_bucket_size(self) -> None:
        """Test with large bucket size (2 hours = 120 minutes)."""
        indexer = TimeOfDayIndexer(bucket_minutes=120)

        # 0:00 -> bucket 0
        assert await indexer.key(datetime(2026, 1, 29, 0, 0, 0)) == 0

        # 1:59 -> still bucket 0
        assert await indexer.key(datetime(2026, 1, 29, 1, 59, 0)) == 0

        # 2:00 -> bucket 1
        assert await indexer.key(datetime(2026, 1, 29, 2, 0, 0)) == 1

        # 6:00 -> bucket 3 (6 * 60 / 120 = 3)
        assert await indexer.key(datetime(2026, 1, 29, 6, 0, 0)) == 3

    @pytest.mark.asyncio


    async def test_key_consistency_across_boundaries(self) -> None:
        """Test that keys change correctly across boundaries."""
        indexer = TimeOfDayIndexer(bucket_minutes=30)

        # Just before boundary
        ts1 = datetime(2026, 1, 29, 6, 29, 59)
        key1 = await indexer.key(ts1)

        # At boundary
        ts2 = datetime(2026, 1, 29, 6, 30, 0)
        key2 = await indexer.key(ts2)

        # Keys should differ
        assert key2 == key1 + 1

        # Next boundary should align with when key changes
        next_bound = await indexer.next_boundary(ts1)
        assert await indexer.key(next_bound) == key2
