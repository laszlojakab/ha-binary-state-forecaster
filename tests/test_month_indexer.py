"""Tests for MonthIndexer class."""

import pytest
from datetime import datetime

from custom_components.discrete_state_forecaster.model.time_indexers.month_indexer import (
    MonthIndexer,
)


class TestMonthIndexerInitialization:
    """Test MonthIndexer initialization."""

    @pytest.mark.asyncio


    async def test_initialization(self) -> None:
        """Test indexer initializes with correct name."""
        indexer = MonthIndexer()
        assert indexer.name == "month"


class TestMonthIndexerKey:
    """Test MonthIndexer key calculation."""

    @pytest.mark.asyncio


    async def test_key_january(self) -> None:
        """Test key for January."""
        indexer = MonthIndexer()
        ts = datetime(2026, 1, 15, 12, 0, 0)
        assert await indexer.key(ts) == 1

    @pytest.mark.asyncio


    async def test_key_february(self) -> None:
        """Test key for February."""
        indexer = MonthIndexer()
        ts = datetime(2026, 2, 14, 12, 0, 0)
        assert await indexer.key(ts) == 2

    @pytest.mark.asyncio


    async def test_key_march(self) -> None:
        """Test key for March."""
        indexer = MonthIndexer()
        ts = datetime(2026, 3, 20, 12, 0, 0)
        assert await indexer.key(ts) == 3

    @pytest.mark.asyncio


    async def test_key_april(self) -> None:
        """Test key for April."""
        indexer = MonthIndexer()
        ts = datetime(2026, 4, 10, 12, 0, 0)
        assert await indexer.key(ts) == 4

    @pytest.mark.asyncio


    async def test_key_may(self) -> None:
        """Test key for May."""
        indexer = MonthIndexer()
        ts = datetime(2026, 5, 1, 12, 0, 0)
        assert await indexer.key(ts) == 5

    @pytest.mark.asyncio


    async def test_key_june(self) -> None:
        """Test key for June."""
        indexer = MonthIndexer()
        ts = datetime(2026, 6, 30, 12, 0, 0)
        assert await indexer.key(ts) == 6

    @pytest.mark.asyncio


    async def test_key_july(self) -> None:
        """Test key for July."""
        indexer = MonthIndexer()
        ts = datetime(2026, 7, 4, 12, 0, 0)
        assert await indexer.key(ts) == 7

    @pytest.mark.asyncio


    async def test_key_august(self) -> None:
        """Test key for August."""
        indexer = MonthIndexer()
        ts = datetime(2026, 8, 15, 12, 0, 0)
        assert await indexer.key(ts) == 8

    @pytest.mark.asyncio


    async def test_key_september(self) -> None:
        """Test key for September."""
        indexer = MonthIndexer()
        ts = datetime(2026, 9, 22, 12, 0, 0)
        assert await indexer.key(ts) == 9

    @pytest.mark.asyncio


    async def test_key_october(self) -> None:
        """Test key for October."""
        indexer = MonthIndexer()
        ts = datetime(2026, 10, 31, 12, 0, 0)
        assert await indexer.key(ts) == 10

    @pytest.mark.asyncio


    async def test_key_november(self) -> None:
        """Test key for November."""
        indexer = MonthIndexer()
        ts = datetime(2026, 11, 11, 12, 0, 0)
        assert await indexer.key(ts) == 11

    @pytest.mark.asyncio


    async def test_key_december(self) -> None:
        """Test key for December."""
        indexer = MonthIndexer()
        ts = datetime(2026, 12, 25, 12, 0, 0)
        assert await indexer.key(ts) == 12

    @pytest.mark.asyncio


    async def test_key_same_month_different_years(self) -> None:
        """Test that same month returns same key across different years."""
        indexer = MonthIndexer()

        # All January dates should have key 1
        jan1 = datetime(2026, 1, 10, 10, 0, 0)
        jan2 = datetime(2025, 1, 20, 15, 30, 0)
        jan3 = datetime(2027, 1, 5, 8, 45, 0)

        assert await indexer.key(jan1) == await indexer.key(jan2) == await indexer.key(jan3) == 1

    @pytest.mark.asyncio


    async def test_key_ignores_day_and_time(self) -> None:
        """Test that key only depends on month, not day or time."""
        indexer = MonthIndexer()

        # Same month, different days and times
        ts1 = datetime(2026, 6, 1, 0, 0, 0)  # June 1st midnight
        ts2 = datetime(2026, 6, 15, 12, 0, 0)  # June 15th noon
        ts3 = datetime(2026, 6, 30, 23, 59, 59)  # June 30th almost midnight

        assert await indexer.key(ts1) == await indexer.key(ts2) == await indexer.key(ts3) == 6

    @pytest.mark.asyncio


    async def test_key_full_year_sequence(self) -> None:
        """Test a complete year sequence."""
        indexer = MonthIndexer()

        expected_keys = list(range(1, 13))  # 1 through 12

        for month, expected_key in enumerate(expected_keys, start=1):
            ts = datetime(2026, month, 15, 12, 0, 0)
            assert await indexer.key(ts) == expected_key

    @pytest.mark.asyncio


    async def test_key_leap_year_february(self) -> None:
        """Test February in leap year."""
        indexer = MonthIndexer()

        # February 29, 2024 (leap year)
        ts = datetime(2024, 2, 29, 12, 0, 0)
        assert await indexer.key(ts) == 2


class TestMonthIndexerNextBoundary:
    """Test MonthIndexer next_boundary calculation."""

    @pytest.mark.asyncio


    async def test_next_boundary_january_to_february(self) -> None:
        """Test next_boundary from January to February."""
        indexer = MonthIndexer()
        ts = datetime(2026, 1, 15, 14, 30, 45)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 2, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_february_to_march(self) -> None:
        """Test next_boundary from February to March."""
        indexer = MonthIndexer()
        ts = datetime(2026, 2, 28, 23, 59, 59)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_leap_year_february(self) -> None:
        """Test next_boundary from February 29 in leap year."""
        indexer = MonthIndexer()
        ts = datetime(2024, 2, 29, 12, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2024, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_december_to_january(self) -> None:
        """Test next_boundary from December to January (year wrap)."""
        indexer = MonthIndexer()
        ts = datetime(2026, 12, 31, 23, 59, 59)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2027, 1, 1, 0, 0, 0)
        assert next_bound.year == 2027

    @pytest.mark.asyncio


    async def test_next_boundary_first_day_of_month(self) -> None:
        """Test next_boundary when timestamp is first day of month."""
        indexer = MonthIndexer()
        ts = datetime(2026, 6, 1, 0, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 7, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_last_day_of_month(self) -> None:
        """Test next_boundary when timestamp is last day of month."""
        indexer = MonthIndexer()
        # June has 30 days
        ts = datetime(2026, 6, 30, 23, 59, 59)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 7, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_31_day_month_to_30_day_month(self) -> None:
        """Test next_boundary from 31-day month to 30-day month."""
        indexer = MonthIndexer()
        # January has 31 days, February has 28/29
        ts = datetime(2026, 1, 31, 12, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 2, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_30_day_month_to_31_day_month(self) -> None:
        """Test next_boundary from 30-day month to 31-day month."""
        indexer = MonthIndexer()
        # June has 30 days, July has 31
        ts = datetime(2026, 6, 30, 12, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2026, 7, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_next_boundary_always_midnight(self) -> None:
        """Test that next_boundary always returns midnight."""
        indexer = MonthIndexer()

        test_times = [
            datetime(2026, 1, 15, 1, 15, 30, 123456),
            datetime(2026, 6, 30, 9, 45, 12, 999999),
            datetime(2026, 12, 25, 17, 30, 0, 500000),
        ]

        for ts in test_times:
            next_bound = await indexer.next_boundary(ts)
            assert next_bound.hour == 0
            assert next_bound.minute == 0
            assert next_bound.second == 0
            assert next_bound.microsecond == 0

    @pytest.mark.asyncio


    async def test_next_boundary_always_first_of_month(self) -> None:
        """Test that next_boundary always returns the 1st day."""
        indexer = MonthIndexer()

        test_times = [
            datetime(2026, 1, 5, 12, 0, 0),
            datetime(2026, 2, 14, 14, 30, 0),
            datetime(2026, 3, 31, 23, 59, 59),
        ]

        for ts in test_times:
            next_bound = await indexer.next_boundary(ts)
            assert next_bound.day == 1

    @pytest.mark.asyncio


    async def test_next_boundary_sequence(self) -> None:
        """Test a sequence of next_boundary calls through the year."""
        indexer = MonthIndexer()

        ts = datetime(2026, 1, 15, 12, 0, 0)  # January
        expected_boundaries = [
            datetime(2026, 2, 1, 0, 0, 0),  # February
            datetime(2026, 3, 1, 0, 0, 0),  # March
            datetime(2026, 4, 1, 0, 0, 0),  # April
            datetime(2026, 5, 1, 0, 0, 0),  # May
        ]

        for expected in expected_boundaries:
            ts = await indexer.next_boundary(ts)
            assert ts == expected

    @pytest.mark.asyncio


    async def test_next_boundary_year_end_sequence(self) -> None:
        """Test next_boundary sequence across year boundary."""
        indexer = MonthIndexer()

        ts = datetime(2026, 11, 15, 12, 0, 0)  # November

        # November -> December
        ts = await indexer.next_boundary(ts)
        assert ts == datetime(2026, 12, 1, 0, 0, 0)

        # December -> January (next year)
        ts = await indexer.next_boundary(ts)
        assert ts == datetime(2027, 1, 1, 0, 0, 0)
        assert ts.year == 2027


class TestMonthIndexerEdgeCases:
    """Test edge cases for MonthIndexer."""

    @pytest.mark.asyncio


    async def test_key_consistency_across_boundaries(self) -> None:
        """Test that keys change correctly across boundaries."""
        indexer = MonthIndexer()

        # Last moment of January
        ts1 = datetime(2026, 1, 31, 23, 59, 59)
        key1 = await indexer.key(ts1)

        # Get boundary (February 1st)
        ts2 = await indexer.next_boundary(ts1)
        key2 = await indexer.key(ts2)

        # Keys should differ
        assert key1 == 1  # January
        assert key2 == 2  # February

    @pytest.mark.asyncio


    async def test_same_date_different_years(self) -> None:
        """Test same date in different years may have same month key."""
        indexer = MonthIndexer()

        # June 15 in different years
        ts_2024 = datetime(2024, 6, 15, 12, 0, 0)
        ts_2025 = datetime(2025, 6, 15, 12, 0, 0)
        ts_2026 = datetime(2026, 6, 15, 12, 0, 0)

        # All should be June (6)
        assert await indexer.key(ts_2024) == 6
        assert await indexer.key(ts_2025) == 6
        assert await indexer.key(ts_2026) == 6

    @pytest.mark.asyncio


    async def test_december_boundary_increments_year(self) -> None:
        """Test that December boundary correctly increments year."""
        indexer = MonthIndexer()

        ts = datetime(2026, 12, 1, 0, 0, 0)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound.year == 2027
        assert next_bound.month == 1
        assert next_bound.day == 1

    @pytest.mark.asyncio


    async def test_all_months_have_unique_keys(self) -> None:
        """Test that all 12 months produce unique keys."""
        indexer = MonthIndexer()

        keys = set()
        for month in range(1, 13):
            ts = datetime(2026, month, 15, 12, 0, 0)
            key = await indexer.key(ts)
            keys.add(key)

        # Should have 12 unique keys
        assert len(keys) == 12
        assert keys == set(range(1, 13))

    @pytest.mark.asyncio


    async def test_february_non_leap_year(self) -> None:
        """Test February boundary in non-leap year."""
        indexer = MonthIndexer()

        # February 28, 2025 (non-leap year)
        ts = datetime(2025, 2, 28, 23, 59, 59)
        next_bound = await indexer.next_boundary(ts)

        assert next_bound == datetime(2025, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio


    async def test_multiple_year_transitions(self) -> None:
        """Test multiple December to January transitions."""
        indexer = MonthIndexer()

        # 2024 -> 2025
        ts = datetime(2024, 12, 15, 12, 0, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2025, 1, 1, 0, 0, 0)

        # 2025 -> 2026
        ts = datetime(2025, 12, 20, 18, 30, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 1, 1, 0, 0, 0)
