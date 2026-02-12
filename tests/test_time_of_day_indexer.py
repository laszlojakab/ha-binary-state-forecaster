"""Unit tests for TimeOfDayIndexer."""

from datetime import datetime
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey
from custom_components.discrete_state_forecaster.model.temporal.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestTimeOfDayIndexerInitialization:
    """Tests for TimeOfDayIndexer initialization."""

    def test_create_with_valid_bucket_size(self: Self) -> None:
        """Test creating indexer with valid bucket size."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        assert indexer.bucket_size == 3600
        assert indexer.name == "time_bucket"

    def test_create_with_small_bucket_size(self: Self) -> None:
        """Test creating indexer with small bucket size."""
        indexer = TimeOfDayIndexer(bucket_size=1)
        assert indexer.bucket_size == 1

    def test_create_with_large_bucket_size(self: Self) -> None:
        """Test creating indexer with large bucket size."""
        indexer = TimeOfDayIndexer(bucket_size=86400)
        assert indexer.bucket_size == 86400

    def test_invalid_zero_bucket_size(self: Self) -> None:
        """Test that bucket_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            TimeOfDayIndexer(bucket_size=0)

    def test_invalid_negative_bucket_size(self: Self) -> None:
        """Test that negative bucket_size raises ValueError."""
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            TimeOfDayIndexer(bucket_size=-100)


class TestTimeOfDayIndexerGetKey:
    """Tests for TimeOfDayIndexer.get_key method."""

    @pytest.mark.asyncio
    async def test_midnight(self: Self) -> None:
        """Test that midnight maps to bucket 0."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        timestamp = datetime(2024, 1, 15, 0, 0, 0)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 0),)

    @pytest.mark.asyncio
    async def test_hour_14(self: Self) -> None:
        """Test that hour 14 (2 PM) maps to bucket 14 with 1-hour buckets."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        timestamp = datetime(2024, 1, 15, 14, 30, 45)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 14),)

    @pytest.mark.asyncio
    async def test_hour_23(self: Self) -> None:
        """Test that hour 23 (11 PM) maps to bucket 23 with 1-hour buckets."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        timestamp = datetime(2024, 1, 15, 23, 59, 59)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 23),)

    @pytest.mark.asyncio
    async def test_5_minute_buckets(self: Self) -> None:
        """Test with 5-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_size=300)
        # 14:30 = 52200 seconds = bucket 174 (52200 // 300)
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 174),)

    @pytest.mark.asyncio
    async def test_1_minute_buckets(self: Self) -> None:
        """Test with 1-minute buckets."""
        indexer = TimeOfDayIndexer(bucket_size=60)
        # 14:30 = 52200 seconds = bucket 870 (52200 // 60)
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 870),)

    @pytest.mark.asyncio
    async def test_ignores_seconds_in_hour_buckets(self: Self) -> None:
        """Test that seconds don't affect bucket with 1-hour buckets."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        ts1 = datetime(2024, 1, 15, 14, 30, 0)
        ts2 = datetime(2024, 1, 15, 14, 30, 59)
        key1 = await indexer.get_key(ts1)
        key2 = await indexer.get_key(ts2)
        assert key1 == key2
        assert key1.parts == (("time_bucket", 14),)

    @pytest.mark.asyncio
    async def test_keys_are_different_across_hours(self: Self) -> None:
        """Test that different hours produce different keys."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 11, 0))
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_same_time_different_dates(self: Self) -> None:
        """Test that same time on different dates produces same bucket."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key1 = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        key2 = await indexer.get_key(datetime(2024, 2, 15, 14, 30))
        key3 = await indexer.get_key(datetime(2024, 12, 31, 14, 30))
        assert key1 == key2 == key3

    @pytest.mark.asyncio
    async def test_boundary_at_1_second_into_hour(self: Self) -> None:
        """Test bucket at 1 second into an hour."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        timestamp = datetime(2024, 1, 15, 14, 0, 1)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 14),)

    @pytest.mark.asyncio
    async def test_boundary_at_last_second_of_hour(self: Self) -> None:
        """Test bucket at last second before new hour."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        timestamp = datetime(2024, 1, 15, 14, 59, 59)
        key = await indexer.get_key(timestamp)
        assert key.parts == (("time_bucket", 14),)


class TestTimeOfDayIndexerReturnType:
    """Tests for return type and structure of TimeOfDayIndexer.get_key."""

    @pytest.mark.asyncio
    async def test_returns_timekey(self: Self) -> None:
        """Test that get_key returns a TimeKey."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        assert isinstance(key, TimeKey)

    @pytest.mark.asyncio
    async def test_timekey_has_one_feature(self: Self) -> None:
        """Test that returned TimeKey has exactly one feature."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        assert len(key) == 1

    @pytest.mark.asyncio
    async def test_feature_name_is_time_bucket(self: Self) -> None:
        """Test that feature name is always 'time_bucket'."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        name, value = key.parts[0]
        assert name == "time_bucket"
        assert isinstance(value, int)

    @pytest.mark.asyncio
    async def test_key_is_hashable(self: Self) -> None:
        """Test that returned TimeKey is hashable."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        hash_value = hash(key)
        assert isinstance(hash_value, int)

    @pytest.mark.asyncio
    async def test_keys_usable_in_dict(self: Self) -> None:
        """Test that keys can be used as dictionary keys."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 11, 0))
        d = {key1: "morning", key2: "late morning"}
        assert d[key1] == "morning"
        assert d[key2] == "late morning"

    @pytest.mark.asyncio
    async def test_keys_usable_in_set(self: Self) -> None:
        """Test that keys can be used in sets."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        key3 = await indexer.get_key(datetime(2024, 1, 15, 11, 0))
        s = {key1, key2, key3}
        assert len(s) == 2  # key1 and key2 are the same


class TestTimeOfDayIndexerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_entire_day_with_hour_buckets(self: Self) -> None:
        """Test all hours of the day produce different buckets."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        keys = []
        for hour in range(24):
            timestamp = datetime(2024, 1, 15, hour, 0)
            key = await indexer.get_key(timestamp)
            keys.append(key)
        # All keys should be different
        assert len(set(keys)) == 24

    @pytest.mark.asyncio
    async def test_bucket_value_is_non_negative(self: Self) -> None:
        """Test that bucket values are always non-negative."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = datetime(2024, 1, 15, hour, minute)
                key = await indexer.get_key(timestamp)
                value = key.parts[0][1]
                assert value >= 0

    @pytest.mark.asyncio
    async def test_bucket_value_less_than_max_possible(self: Self) -> None:
        """Test that bucket values never exceed the daily maximum."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        max_bucket = (86400 - 1) // 3600  # 23
        timestamp = datetime(2024, 1, 15, 23, 59, 59)
        key = await indexer.get_key(timestamp)
        value = key.parts[0][1]
        assert value <= max_bucket

    @pytest.mark.asyncio
    async def test_leap_second_handling(self: Self) -> None:
        """Test handling of edge time case (though Python doesn't support leap seconds)."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        # Test around midnight
        before_midnight = datetime(2024, 1, 15, 23, 59, 59)
        at_midnight = datetime(2024, 1, 16, 0, 0, 0)
        key1 = await indexer.get_key(before_midnight)
        key2 = await indexer.get_key(at_midnight)
        assert key1.parts == (("time_bucket", 23),)
        assert key2.parts == (("time_bucket", 0),)

    @pytest.mark.asyncio
    async def test_very_small_bucket_size(self: Self) -> None:
        """Test with very small bucket size (1 second)."""
        indexer = TimeOfDayIndexer(bucket_size=1)
        ts1 = datetime(2024, 1, 15, 10, 30, 25)
        ts2 = datetime(2024, 1, 15, 10, 30, 26)
        key1 = await indexer.get_key(ts1)
        key2 = await indexer.get_key(ts2)
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_full_day_bucket(self: Self) -> None:
        """Test with bucket size equal to full day."""
        indexer = TimeOfDayIndexer(bucket_size=86400)
        # All times during the day should map to bucket 0
        for hour in [0, 6, 12, 18, 23]:
            timestamp = datetime(2024, 1, 15, hour, 30)
            key = await indexer.get_key(timestamp)
            assert key.parts == (("time_bucket", 0),)


class TestTimeOfDayIndexerEquality:
    """Tests for equality behavior of TimeOfDayIndexer results."""

    @pytest.mark.asyncio
    async def test_equal_keys_have_equal_hash(self: Self) -> None:
        """Test that equal keys have equal hashes."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        key1 = await indexer.get_key(datetime(2024, 1, 15, 14, 0))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        assert key1 == key2
        assert hash(key1) == hash(key2)

    @pytest.mark.asyncio
    async def test_different_hours_different_keys(self: Self) -> None:
        """Test that different hours produce different keys."""
        indexer = TimeOfDayIndexer(bucket_size=3600)
        keys = []
        for hour in [8, 12, 16, 20]:
            timestamp = datetime(2024, 1, 15, hour, 0)
            key = await indexer.get_key(timestamp)
            keys.append(key)
        # All keys should be different
        unique_keys = set(keys)
        assert len(unique_keys) == len(keys)
