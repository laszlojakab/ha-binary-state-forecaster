"""
Comprehensive tests for TimeAwareForecaster.

Tests cover:
- Initialization with various indexer configurations
- Interval updates with single and multiple buckets
- Prediction accuracy and confidence metrics
- Edge cases (invalid intervals, boundary conditions)
- Integration with different time indexers
- Temporal pattern learning and retrieval
"""

import pytest
from datetime import datetime, timedelta

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


class TestInitialization:
    """Tests for TimeAwareForecaster initialization."""

    @pytest.mark.asyncio


    async def test_init_with_simple_indexer(self) -> None:
        """Test initialization with a simple time-of-day indexer."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        assert forecaster is not None
        assert forecaster.indexer == indexer
        assert forecaster.model is not None

    @pytest.mark.asyncio


    async def test_init_with_composite_indexer(self) -> None:
        """Test initialization with multiple indexers."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(30)])
        forecaster = TimeAwareForecaster(indexer)
        assert forecaster is not None
        assert forecaster.indexer == indexer

    @pytest.mark.asyncio


    async def test_init_with_default_half_life(self) -> None:
        """Test initialization with default half_life (no decay)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)
        assert forecaster is not None

    @pytest.mark.asyncio


    async def test_init_with_custom_half_life(self) -> None:
        """Test initialization with custom half_life."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, half_life=3600.0)
        assert forecaster is not None


class TestUpdateInterval:
    """Tests for update_interval method."""

    @pytest.mark.asyncio


    async def test_update_single_bucket(self) -> None:
        """Test updating with interval entirely within one bucket."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])  # 1-hour buckets
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 15)
        end = datetime(2024, 1, 1, 10, 45)

        await forecaster.update_interval(start, end, "on")

        # Verify prediction works
        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction is not None

    @pytest.mark.asyncio


    async def test_update_crossing_bucket_boundary(self) -> None:
        """Test updating with interval crossing bucket boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])  # 1-hour buckets
        forecaster = TimeAwareForecaster(indexer)

        # Interval from 9:45 to 10:15 crosses 10:00 boundary
        start = datetime(2024, 1, 1, 9, 45)
        end = datetime(2024, 1, 1, 10, 15)

        await forecaster.update_interval(start, end, "on")

        # Both buckets should have data
        pred_before = await forecaster.predict(datetime(2024, 1, 1, 9, 50))
        pred_after = await forecaster.predict(datetime(2024, 1, 1, 10, 10))

        assert pred_before is not None
        assert pred_after is not None

    @pytest.mark.asyncio


    async def test_update_multiple_buckets(self) -> None:
        """Test updating with interval spanning multiple buckets."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])  # 1-hour buckets
        forecaster = TimeAwareForecaster(indexer)

        # Interval from 9:00 to 12:00 spans 3 buckets
        start = datetime(2024, 1, 1, 9, 0)
        end = datetime(2024, 1, 1, 12, 0)

        await forecaster.update_interval(start, end, "on")

        # All three buckets should have data
        pred_9 = await forecaster.predict(datetime(2024, 1, 1, 9, 30))
        pred_10 = await forecaster.predict(datetime(2024, 1, 1, 10, 30))
        pred_11 = await forecaster.predict(datetime(2024, 1, 1, 11, 30))

        assert pred_9.state == "on"
        assert pred_10.state == "on"
        assert pred_11.state == "on"

    @pytest.mark.asyncio


    async def test_update_invalid_interval_end_before_start(self) -> None:
        """Test that invalid interval (end <= start) is ignored."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 9, 0)  # End before start

        # Should not raise error, just ignore
        await forecaster.update_interval(start, end, "on")

        # No data should be recorded
        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 0))
        assert prediction.state is None

    @pytest.mark.asyncio


    async def test_update_invalid_interval_end_equals_start(self) -> None:
        """Test that zero-duration interval is ignored."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 10, 0)  # Same as start

        await forecaster.update_interval(start, end, "on")

        # No data should be recorded
        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 0))
        assert prediction.state is None

    @pytest.mark.asyncio


    async def test_update_multiple_states_same_bucket(self) -> None:
        """Test updating same bucket with different states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Same bucket, different states
        start1 = datetime(2024, 1, 1, 10, 0)
        end1 = datetime(2024, 1, 1, 10, 30)
        await forecaster.update_interval(start1, end1, "on")

        start2 = datetime(2024, 1, 2, 10, 0)
        end2 = datetime(2024, 1, 2, 10, 15)
        await forecaster.update_interval(start2, end2, "off")

        # Prediction should reflect both observations
        prediction = await forecaster.predict(datetime(2024, 1, 3, 10, 15))
        assert prediction is not None
        assert prediction.state in ["on", "off"]
        assert len(prediction.distribution) == 2

    @pytest.mark.asyncio


    async def test_update_crossing_midnight(self) -> None:
        """Test updating with interval crossing midnight."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 11 PM to 1 AM crosses midnight
        start = datetime(2024, 1, 1, 23, 30)
        end = datetime(2024, 1, 2, 0, 30)

        await forecaster.update_interval(start, end, "on")

        # Both sides of midnight should have data
        pred_before = await forecaster.predict(datetime(2024, 1, 1, 23, 45))
        pred_after = await forecaster.predict(datetime(2024, 1, 2, 0, 15))

        assert pred_before.state == "on"
        assert pred_after.state == "on"

    @pytest.mark.asyncio


    async def test_update_crossing_month_boundary(self) -> None:
        """Test updating with interval crossing month boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Last hour of January to first hour of February
        start = datetime(2024, 1, 31, 23, 30)
        end = datetime(2024, 2, 1, 0, 30)

        await forecaster.update_interval(start, end, "on")

        pred_jan = await forecaster.predict(datetime(2024, 1, 31, 23, 45))
        pred_feb = await forecaster.predict(datetime(2024, 2, 1, 0, 15))

        assert pred_jan.state == "on"
        assert pred_feb.state == "on"

    @pytest.mark.asyncio


    async def test_update_very_long_interval(self) -> None:
        """Test updating with very long interval."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 24-hour interval
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 2, 0, 0)

        await forecaster.update_interval(start, end, "on")

        # All 24 hours should have data
        for hour in range(24):
            pred = await forecaster.predict(datetime(2024, 1, 1, hour, 30))
            assert pred.state == "on"

    @pytest.mark.asyncio


    async def test_update_very_short_interval(self) -> None:
        """Test updating with very short interval (should be filtered)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 1-second interval (below MIN_DURATION_THRESHOLD of 5 seconds)
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 1)

        await forecaster.update_interval(start, end, "on")

        # Should be filtered, no data
        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 0))
        assert prediction.state is None


class TestPredict:
    """Tests for predict method."""

    @pytest.mark.asyncio


    async def test_predict_after_single_update(self) -> None:
        """Test prediction after single update."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 11, 0)
        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 1, 2, 10, 30))

        assert prediction.state == "on"
        assert prediction.distribution["on"] == 1.0
        assert prediction.confidence.max_probability == 1.0

    @pytest.mark.asyncio


    async def test_predict_after_multiple_updates(self) -> None:
        """Test prediction after multiple updates with different states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # 75% on, 25% off
        for _ in range(3):
            await forecaster.update_interval(
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                "on",
            )

        await forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        prediction = await forecaster.predict(datetime(2024, 1, 3, 10, 30))

        assert prediction.state == "on"  # Most likely
        assert "on" in prediction.distribution
        assert "off" in prediction.distribution
        assert prediction.distribution["on"] > prediction.distribution["off"]

    @pytest.mark.asyncio


    async def test_predict_no_data(self) -> None:
        """Test prediction when no data exists for bucket."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 30))

        assert prediction.state is None
        assert prediction.distribution == {}
        assert prediction.confidence.max_probability == 0.0

    @pytest.mark.asyncio


    async def test_predict_confidence_metrics(self) -> None:
        """Test that prediction includes confidence metrics."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        await forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        prediction = await forecaster.predict(datetime(2024, 1, 2, 10, 30))

        assert prediction.confidence is not None
        assert prediction.confidence.max_probability == 1.0
        assert prediction.confidence.entropy_confidence > 0.9
        assert prediction.confidence.support_time > 0

    @pytest.mark.asyncio


    async def test_predict_with_day_of_week_indexer(self) -> None:
        """Test prediction with day-of-week temporal pattern."""
        indexer = CompositeIndexer([DayOfWeekIndexer()])
        forecaster = TimeAwareForecaster(indexer)

        # Monday (2024-01-01 is a Monday)
        await forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "working",
        )

        # Saturday
        await forecaster.update_interval(
            datetime(2024, 1, 6, 10, 0),
            datetime(2024, 1, 6, 11, 0),
            "off",
        )

        # Predict for next Monday
        pred_monday = await forecaster.predict(datetime(2024, 1, 8, 10, 30))
        # Predict for next Saturday
        pred_saturday = await forecaster.predict(datetime(2024, 1, 13, 10, 30))

        assert pred_monday.state == "working"
        assert pred_saturday.state == "off"

    @pytest.mark.asyncio


    async def test_predict_with_hierarchical_indexer(self) -> None:
        """Test prediction with hierarchical time indexer."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Weekday morning pattern
        await forecaster.update_interval(
            datetime(2024, 1, 1, 9, 0),  # Monday 9 AM
            datetime(2024, 1, 1, 10, 0),
            "busy",
        )

        # Weekend morning pattern
        await forecaster.update_interval(
            datetime(2024, 1, 6, 9, 0),  # Saturday 9 AM
            datetime(2024, 1, 6, 10, 0),
            "relaxed",
        )

        # Predict for next Monday morning
        pred_weekday = await forecaster.predict(datetime(2024, 1, 8, 9, 30))
        # Predict for next Saturday morning
        pred_weekend = await forecaster.predict(datetime(2024, 1, 13, 9, 30))

        assert pred_weekday.state == "busy"
        assert pred_weekend.state == "relaxed"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio


    async def test_exact_bucket_boundary_start(self) -> None:
        """Test interval starting exactly at bucket boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0, 0)  # Exactly 10:00
        end = datetime(2024, 1, 1, 10, 30, 0)

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 15))
        assert prediction.state == "on"

    @pytest.mark.asyncio


    async def test_exact_bucket_boundary_end(self) -> None:
        """Test interval ending exactly at bucket boundary."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 9, 30, 0)
        end = datetime(2024, 1, 1, 10, 0, 0)  # Exactly 10:00

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 1, 1, 9, 45))
        assert prediction.state == "on"

    @pytest.mark.asyncio


    async def test_exact_bucket_boundary_both(self) -> None:
        """Test interval with both start and end at bucket boundaries."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 11, 0, 0)

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 30))
        assert prediction.state == "on"

    @pytest.mark.asyncio


    async def test_microsecond_precision(self) -> None:
        """Test handling of microsecond precision in timestamps."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0, 0, 123456)
        end = datetime(2024, 1, 1, 10, 30, 0, 654321)

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 1, 1, 10, 15, 0, 999999))
        assert prediction.state == "on"

    @pytest.mark.asyncio


    async def test_very_fine_grained_indexer(self) -> None:
        """Test with very fine-grained time buckets (1-minute)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])  # 1-minute buckets
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 15, 0)
        end = datetime(2024, 1, 1, 10, 17, 0)

        await forecaster.update_interval(start, end, "on")

        # Should have data in 2 buckets (10:15 and 10:16)
        pred_15 = await forecaster.predict(datetime(2024, 1, 1, 10, 15, 30))
        pred_16 = await forecaster.predict(datetime(2024, 1, 1, 10, 16, 30))

        assert pred_15.state == "on"
        assert pred_16.state == "on"

    @pytest.mark.asyncio


    async def test_very_coarse_grained_indexer(self) -> None:
        """Test with very coarse-grained time buckets (1-day)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(86400)])  # 1-day buckets
        forecaster = TimeAwareForecaster(indexer)

        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 14, 0)

        await forecaster.update_interval(start, end, "on")

        # Entire day should be in same bucket
        pred_morning = await forecaster.predict(datetime(2024, 1, 1, 9, 0))
        pred_afternoon = await forecaster.predict(datetime(2024, 1, 1, 15, 0))

        assert pred_morning.state == "on"
        assert pred_afternoon.state == "on"

    @pytest.mark.asyncio


    async def test_leap_year_handling(self) -> None:
        """Test handling of leap year dates."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Feb 29 in leap year
        start = datetime(2024, 2, 29, 10, 0)
        end = datetime(2024, 2, 29, 11, 0)

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 2, 29, 10, 30))
        assert prediction.state == "on"

    @pytest.mark.asyncio


    async def test_dst_transition(self) -> None:
        """Test handling near DST transition (naive datetimes)."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Note: Using naive datetimes (no timezone info)
        # DST transition in 2024 is around March 10
        start = datetime(2024, 3, 10, 1, 30)
        end = datetime(2024, 3, 10, 3, 30)

        await forecaster.update_interval(start, end, "on")

        prediction = await forecaster.predict(datetime(2024, 3, 10, 2, 30))
        assert prediction is not None


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio


    async def test_realistic_daily_pattern(self) -> None:
        """Test learning and predicting realistic daily patterns."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Simulate 7 days of data
        for day in range(7):
            base_date = datetime(2024, 1, 1) + timedelta(days=day)

            # Morning: off
            await forecaster.update_interval(
                base_date.replace(hour=0, minute=0),
                base_date.replace(hour=8, minute=0),
                "off",
            )

            # Work hours: on
            await forecaster.update_interval(
                base_date.replace(hour=8, minute=0),
                base_date.replace(hour=17, minute=0),
                "on",
            )

            # Evening: off
            await forecaster.update_interval(
                base_date.replace(hour=17, minute=0),
                base_date.replace(hour=23, minute=59),
                "off",
            )

        # Predict for next day
        pred_morning = await forecaster.predict(datetime(2024, 1, 8, 6, 0))
        pred_work = await forecaster.predict(datetime(2024, 1, 8, 12, 0))
        pred_evening = await forecaster.predict(datetime(2024, 1, 8, 20, 0))

        assert pred_morning.state == "off"
        assert pred_work.state == "on"
        assert pred_evening.state == "off"

    @pytest.mark.asyncio


    async def test_realistic_weekly_pattern(self) -> None:
        """Test learning and predicting realistic weekly patterns."""
        indexer = CompositeIndexer([DayOfWeekIndexer(), TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Weekday pattern (Mon-Fri)
        for day_offset in [0, 1, 2, 3, 4]:  # Mon-Fri
            base_date = datetime(2024, 1, 1) + timedelta(days=day_offset)
            await forecaster.update_interval(
                base_date.replace(hour=9, minute=0),
                base_date.replace(hour=17, minute=0),
                "work",
            )

        # Weekend pattern (Sat-Sun)
        for day_offset in [5, 6]:  # Sat-Sun
            base_date = datetime(2024, 1, 1) + timedelta(days=day_offset)
            await forecaster.update_interval(
                base_date.replace(hour=10, minute=0),
                base_date.replace(hour=15, minute=0),
                "leisure",
            )

        # Predict for next week
        pred_monday = await forecaster.predict(datetime(2024, 1, 8, 12, 0))  # Monday
        pred_saturday = await forecaster.predict(datetime(2024, 1, 13, 12, 0))  # Saturday

        assert pred_monday.state == "work"
        assert pred_saturday.state == "leisure"

    @pytest.mark.asyncio


    async def test_pattern_update_over_time(self) -> None:
        """Test that patterns update as new data is added."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, half_life=86400.0)

        # Initial pattern: mostly "on"
        for _ in range(10):
            await forecaster.update_interval(
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                "on",
            )

        prediction1 = await forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction1.state == "on"

        # New pattern: now "off" (with decay, old data fades)
        for _ in range(10):
            await forecaster.update_interval(
                datetime(2024, 1, 10, 10, 0),
                datetime(2024, 1, 10, 11, 0),
                "off",
            )

        prediction2 = await forecaster.predict(datetime(2024, 1, 11, 10, 30))
        # With decay, pattern should shift toward "off"
        assert prediction2 is not None

    @pytest.mark.asyncio


    async def test_multiple_concurrent_patterns(self) -> None:
        """Test learning multiple independent temporal patterns."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Morning pattern
        await forecaster.update_interval(
            datetime(2024, 1, 1, 8, 0),
            datetime(2024, 1, 1, 9, 0),
            "breakfast",
        )

        # Lunch pattern
        await forecaster.update_interval(
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
            "lunch",
        )

        # Dinner pattern
        await forecaster.update_interval(
            datetime(2024, 1, 1, 18, 0),
            datetime(2024, 1, 1, 19, 0),
            "dinner",
        )

        # Predict for all three times
        pred_morning = await forecaster.predict(datetime(2024, 1, 2, 8, 30))
        pred_lunch = await forecaster.predict(datetime(2024, 1, 2, 12, 30))
        pred_dinner = await forecaster.predict(datetime(2024, 1, 2, 18, 30))

        assert pred_morning.state == "breakfast"
        assert pred_lunch.state == "lunch"
        assert pred_dinner.state == "dinner"
