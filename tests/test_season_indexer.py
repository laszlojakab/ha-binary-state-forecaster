"""Unit tests for SeasonIndexer."""

from datetime import UTC, datetime
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.temporal.season_indexer import (
    SeasonIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestSeasonIndexerBasics:
    """Tests for basic SeasonIndexer functionality."""

    def test_name_attribute(self: Self) -> None:
        """Test that name is set correctly."""
        indexer = SeasonIndexer()
        assert indexer.name == "season"

    @pytest.mark.asyncio
    async def test_returns_timekey(self: Self) -> None:
        """Test that get_key returns a TimeKey."""
        indexer = SeasonIndexer()
        key = await indexer.get_key(datetime(2024, 3, 15, 10, 30))
        assert isinstance(key, TimeKey)

    @pytest.mark.asyncio
    async def test_timekey_has_one_feature(self: Self) -> None:
        """Test that returned TimeKey has exactly one feature."""
        indexer = SeasonIndexer()
        key = await indexer.get_key(datetime(2024, 3, 15, 10, 30))
        assert len(key) == 1


class TestSeasonIndexerMappings:
    """Tests for month to season mappings."""

    @pytest.mark.asyncio
    async def test_spring_mapping(self: Self) -> None:
        """Test Spring (March 1 - May 31)."""
        indexer = SeasonIndexer()
        spring_months = [3, 4, 5]
        for month in spring_months:
            ts = datetime(2024, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", "spring"),), f"Month {month} should be spring"

    @pytest.mark.asyncio
    async def test_summer_mapping(self: Self) -> None:
        """Test Summer (June 1 - August 31)."""
        indexer = SeasonIndexer()
        summer_months = [6, 7, 8]
        for month in summer_months:
            ts = datetime(2024, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", "summer"),), f"Month {month} should be summer"

    @pytest.mark.asyncio
    async def test_autumn_mapping(self: Self) -> None:
        """Test Autumn (September 1 - November 30)."""
        indexer = SeasonIndexer()
        autumn_months = [9, 10, 11]
        for month in autumn_months:
            ts = datetime(2024, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", "autumn"),), f"Month {month} should be autumn"

    @pytest.mark.asyncio
    async def test_winter_mapping(self: Self) -> None:
        """Test Winter (December 1 - February 28/29)."""
        indexer = SeasonIndexer()
        winter_months = [12, 1, 2]
        for month in winter_months:
            if month == 12:
                year = 2024
            else:
                year = 2025  # January and February of next year
            ts = datetime(year, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", "winter"),), f"Month {month} should be winter"

    @pytest.mark.asyncio
    async def test_boundary_first_day_of_season(self: Self) -> None:
        """Test first day of each season."""
        indexer = SeasonIndexer()
        boundaries = [
            (1, "winter"),  # January 1
            (3, "spring"),  # March 1
            (6, "summer"),  # June 1
            (9, "autumn"),  # September 1
            (12, "winter"),  # December 1
        ]
        for month, expected_season in boundaries:
            ts = datetime(2024, month, 1, 0, 0, 0)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", expected_season),)

    @pytest.mark.asyncio
    async def test_boundary_last_day_of_season(self: Self) -> None:
        """Test last day of each season."""
        indexer = SeasonIndexer()
        boundaries = [
            (2, 28, "winter"),  # February 28 (2024 is leap year, so 29)
            (5, 31, "spring"),  # May 31
            (8, 31, "summer"),  # August 31
            (11, 30, "autumn"),  # November 30
        ]
        for month, day, expected_season in boundaries:
            ts = datetime(2024, month, day, 23, 59, 59)
            key = await indexer.get_key(ts)
            assert key.to_tuple() == (("season", expected_season),)


class TestSeasonIndexerIgnoresTime:
    """Tests that time of day is ignored."""

    @pytest.mark.asyncio
    async def test_same_day_different_times(self: Self) -> None:
        """Test that same day with different times produce same key."""
        indexer = SeasonIndexer()
        times = [
            datetime(2024, 3, 15, 0, 0, 0),
            datetime(2024, 3, 15, 6, 30, 15),
            datetime(2024, 3, 15, 12, 30, 45),
            datetime(2024, 3, 15, 18, 15, 30),
            datetime(2024, 3, 15, 23, 59, 59),
        ]
        keys = [await indexer.get_key(ts) for ts in times]
        assert all(k == keys[0] for k in keys)
        assert keys[0].to_tuple() == (("season", "spring"),)


class TestSeasonIndexerConsistency:
    """Tests for consistency across different years."""

    @pytest.mark.asyncio
    async def test_same_month_different_years(self: Self) -> None:
        """Test that same month in different years produces same season."""
        indexer = SeasonIndexer()
        march_dates = [
            datetime(2023, 3, 15, 10, 30),
            datetime(2024, 3, 15, 10, 30),
            datetime(2025, 3, 15, 10, 30),
        ]
        keys = [await indexer.get_key(ts) for ts in march_dates]
        assert all(k.to_tuple() == (("season", "spring"),) for k in keys)


class TestSeasonIndexerHashability:
    """Tests for hashability and use in collections."""

    @pytest.mark.asyncio
    async def test_keys_are_hashable(self: Self) -> None:
        """Test that returned keys are hashable."""
        indexer = SeasonIndexer()
        key = await indexer.get_key(datetime(2024, 3, 15, 10, 30))
        hash_value = hash(key)
        assert isinstance(hash_value, int)

    @pytest.mark.asyncio
    async def test_equal_keys_same_hash(self: Self) -> None:
        """Test that equal keys (same season) have equal hashes."""
        indexer = SeasonIndexer()
        key1 = await indexer.get_key(datetime(2024, 3, 1, 10, 30))
        key2 = await indexer.get_key(datetime(2024, 5, 31, 20, 15))
        assert key1 == key2  # Both spring
        assert hash(key1) == hash(key2)

    @pytest.mark.asyncio
    async def test_different_keys_different_hash(self: Self) -> None:
        """Test that different seasons have different hashes."""
        indexer = SeasonIndexer()
        key_spring = await indexer.get_key(datetime(2024, 3, 15, 10, 30))
        key_summer = await indexer.get_key(datetime(2024, 7, 15, 10, 30))
        assert key_spring != key_summer
        assert hash(key_spring) != hash(key_summer)

    @pytest.mark.asyncio
    async def test_usable_as_dict_key(self: Self) -> None:
        """Test that keys can be used as dictionary keys."""
        indexer = SeasonIndexer()
        key_spring = await indexer.get_key(datetime(2024, 3, 15, 10, 30))
        key_summer = await indexer.get_key(datetime(2024, 7, 15, 10, 30))
        key_autumn = await indexer.get_key(datetime(2024, 9, 15, 10, 30))
        key_winter = await indexer.get_key(datetime(2024, 1, 15, 10, 30))

        patterns = {
            key_spring: "Spring pattern",
            key_summer: "Summer pattern",
            key_autumn: "Autumn pattern",
            key_winter: "Winter pattern",
        }
        assert patterns[key_spring] == "Spring pattern"
        assert patterns[key_summer] == "Summer pattern"
        assert len(patterns) == 4

    @pytest.mark.asyncio
    async def test_usable_in_set(self: Self) -> None:
        """Test that keys can be used in sets."""
        indexer = SeasonIndexer()
        key1 = await indexer.get_key(datetime(2024, 3, 1, 10, 30))  # Spring
        key2 = await indexer.get_key(datetime(2024, 3, 31, 20, 30))  # Spring
        key3 = await indexer.get_key(datetime(2024, 7, 1, 10, 30))  # Summer

        s = {key1, key2, key3}
        assert len(s) == 2  # key1 and key2 are the same (spring)


class TestSeasonIndexerNextBoundary:
    """Tests for next_boundary method."""

    @pytest.mark.asyncio
    async def test_spring_to_summer_boundary(self: Self) -> None:
        """Test next boundary from spring transitions to summer."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 3, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 6, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_summer_to_autumn_boundary(self: Self) -> None:
        """Test next boundary from summer transitions to autumn."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 7, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 9, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_autumn_to_winter_boundary(self: Self) -> None:
        """Test next boundary from autumn transitions to winter."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 10, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 12, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_winter_to_spring_boundary(self: Self) -> None:
        """Test next boundary from December winter transitions to spring (next year)."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 12, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2025, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_january_to_spring_boundary(self: Self) -> None:
        """Test next boundary from January winter transitions to spring (same year)."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 1, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_boundary_strictly_after_timestamp(self: Self) -> None:
        """Test that next boundary is strictly after the timestamp."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 3, 15, 14, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary > ts

    @pytest.mark.asyncio
    async def test_boundary_at_midnight(self: Self) -> None:
        """Test that boundary is at midnight (00:00:00)."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 3, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary.hour == 0
        assert boundary.minute == 0
        assert boundary.second == 0

    @pytest.mark.asyncio
    async def test_boundary_at_first_of_month(self: Self) -> None:
        """Test that boundary is at first day of month."""
        indexer = SeasonIndexer()
        ts = datetime(2024, 3, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary.day == 1

    @pytest.mark.asyncio
    async def test_preserves_timezone(self: Self) -> None:
        """Test that timezone information is preserved."""
        indexer = SeasonIndexer()
        tz = UTC
        ts = datetime(2024, 3, 15, 14, 30, tzinfo=tz)
        boundary = await indexer.next_boundary(ts)
        assert boundary.tzinfo == tz

    @pytest.mark.asyncio
    async def test_boundary_from_first_day_of_season(self: Self) -> None:
        """Test next boundary from the first day of a season."""
        indexer = SeasonIndexer()
        # June 1, 2024 is the first day of summer
        ts = datetime(2024, 6, 1, 0, 0, 0)
        boundary = await indexer.next_boundary(ts)
        # Next boundary should be September 1 (first day of autumn)
        assert boundary == datetime(2024, 9, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_boundary_from_last_day_of_season(self: Self) -> None:
        """Test next boundary from the last day of a season."""
        indexer = SeasonIndexer()
        # May 31, 2024 is the last day of spring
        ts = datetime(2024, 5, 31, 23, 59, 59)
        boundary = await indexer.next_boundary(ts)
        # Next boundary should be June 1 (first day of summer)
        assert boundary == datetime(2024, 6, 1, 0, 0, 0)


class TestSeasonIndexerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_leap_year_february(self: Self) -> None:
        """Test with leap year (February 29)."""
        indexer = SeasonIndexer()
        # February 29, 2024 is winter
        ts = datetime(2024, 2, 29, 10, 30)
        key = await indexer.get_key(ts)
        assert key.to_tuple() == (("season", "winter"),)

    @pytest.mark.asyncio
    async def test_year_boundary_december_to_january(self: Self) -> None:
        """Test across year boundary (December to January)."""
        indexer = SeasonIndexer()
        dec = datetime(2024, 12, 31, 23, 59, 59)
        jan = datetime(2025, 1, 1, 0, 0, 0)
        key_dec = await indexer.get_key(dec)
        key_jan = await indexer.get_key(jan)
        assert key_dec.to_tuple() == (("season", "winter"),)
        assert key_jan.to_tuple() == (("season", "winter"),)
        assert key_dec == key_jan

    @pytest.mark.asyncio
    async def test_far_future_date(self: Self) -> None:
        """Test with far future date."""
        indexer = SeasonIndexer()
        ts = datetime(2099, 7, 15, 10, 30)
        key = await indexer.get_key(ts)
        assert key.to_tuple() == (("season", "summer"),)


class TestSeasonIndexerValueRange:
    """Tests that values are valid season names."""

    @pytest.mark.asyncio
    async def test_all_values_valid_seasons(self: Self) -> None:
        """Test that all values are valid season names."""
        indexer = SeasonIndexer()
        valid_seasons = {"spring", "summer", "autumn", "winter"}

        # Test each month
        for month in range(1, 13):
            ts = datetime(2024, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            season = key.to_tuple()[0][1]
            assert season in valid_seasons

    @pytest.mark.asyncio
    async def test_all_seasons_represented(self: Self) -> None:
        """Test that all four seasons are represented in a year."""
        indexer = SeasonIndexer()
        seasons = set()

        months_per_season = {
            "spring": 3,
            "summer": 6,
            "autumn": 9,
            "winter": 12,
        }

        for _season, month in months_per_season.items():
            ts = datetime(2024, month, 15, 10, 30)
            key = await indexer.get_key(ts)
            found_season = key.to_tuple()[0][1]
            seasons.add(found_season)

        assert seasons == {"spring", "summer", "autumn", "winter"}
