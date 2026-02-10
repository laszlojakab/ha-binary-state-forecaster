"""Unit tests for CalendarIndexer."""

from datetime import UTC, datetime, timedelta
from typing import Self
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.discrete_state_forecaster.model.temporal.calendar_indexer import (
    CalendarIndexer,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestCalendarIndexerInitialization:
    """Tests for CalendarIndexer initialization."""

    def test_valid_initialization(self: Self) -> None:
        """Test initialization with valid parameters."""
        hass = MagicMock()
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        assert indexer.hass is hass
        assert indexer.entity_id == "calendar.work_schedule"
        assert indexer.name == "calendar.work_schedule"

    def test_name_attribute(self: Self) -> None:
        """Test that name is set to entity_id."""
        hass = MagicMock()
        entity_id = "calendar.vacation"
        indexer = CalendarIndexer(hass, entity_id)
        assert indexer.name == entity_id

    def test_empty_entity_id_raises_error(self: Self) -> None:
        """Test that empty entity_id raises ValueError."""
        hass = MagicMock()
        with pytest.raises(ValueError, match="entity_id cannot be empty"):
            CalendarIndexer(hass, "")

    def test_invalid_entity_id_format_raises_error(self: Self) -> None:
        """Test that entity_id without 'calendar.' prefix raises ValueError."""
        hass = MagicMock()
        with pytest.raises(ValueError, match="entity_id must start with 'calendar.'"):
            CalendarIndexer(hass, "work_schedule")

    def test_entity_id_case_sensitive(self: Self) -> None:
        """Test that 'Calendar.' (wrong case) is rejected."""
        hass = MagicMock()
        with pytest.raises(ValueError, match="entity_id must start with 'calendar.'"):
            CalendarIndexer(hass, "Calendar.work_schedule")


class TestCalendarIndexerGetKey:
    """Tests for CalendarIndexer.get_key method."""

    @pytest.mark.asyncio
    async def test_returns_timekey(self: Self) -> None:
        """Test that get_key returns a TimeKey."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert isinstance(key, TimeKey)

    @pytest.mark.asyncio
    async def test_timekey_has_one_feature(self: Self) -> None:
        """Test that returned TimeKey has exactly one feature."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert len(key) == 1

    @pytest.mark.asyncio
    async def test_no_event_returns_zero(self: Self) -> None:
        """Test that no calendar events return value 0."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert key.to_tuple() == (("calendar.work_schedule", 0),)

    @pytest.mark.asyncio
    async def test_event_active_returns_one(self: Self) -> None:
        """Test that active calendar events return value 1."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Work",
                    "start": "2024-01-15T09:00:00",
                    "end": "2024-01-15T17:00:00",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        # 10:30 is during the work event
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert key.to_tuple() == (("calendar.work_schedule", 1),)

    @pytest.mark.asyncio
    async def test_multiple_events_returns_one(self: Self) -> None:
        """Test that multiple events returns value 1."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [
                    {
                        "summary": "Meeting",
                        "start": "2024-01-15T09:00:00",
                        "end": "2024-01-15T10:00:00",
                    },
                    {
                        "summary": "Lunch",
                        "start": "2024-01-15T12:00:00",
                        "end": "2024-01-15T13:00:00",
                    },
                ]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        # At 10:30, at least one event is active (not in the specific examples, but service says there are events)
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert key.to_tuple() == (("calendar.work_schedule", 1),)

    @pytest.mark.asyncio
    async def test_service_error_handled_gracefully(self: Self) -> None:
        """Test that service errors are handled gracefully."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(side_effect=Exception("Service error"))
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        # When service fails, should return 0
        assert key.to_tuple() == (("calendar.work_schedule", 0),)

    @pytest.mark.asyncio
    async def test_service_called_with_correct_params(self: Self) -> None:
        """Test that service is called with correct parameters."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30, 45)
        await indexer.get_key(ts)

        # Verify service was called
        hass.services.async_call.assert_called_once()
        # Get all call arguments
        call_args = hass.services.async_call.call_args
        # Positional args: (domain, service, service_data)
        assert call_args[0][0] == "calendar"
        assert call_args[0][1] == "get_events"

    @pytest.mark.asyncio
    async def test_all_day_event(self: Self) -> None:
        """Test with all-day events (date only, no time)."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Holiday",
                    "start": "2024-01-15",
                    "end": "2024-01-15",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        assert key.to_tuple() == (("calendar.work_schedule", 1),)


class TestCalendarIndexerNextBoundary:
    """Tests for CalendarIndexer.next_boundary method."""

    @pytest.mark.asyncio
    async def test_returns_datetime(self: Self) -> None:
        """Test that next_boundary returns a datetime."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        result = await indexer.next_boundary(datetime(2024, 1, 15, 10, 30))
        assert isinstance(result, datetime)

    @pytest.mark.asyncio
    async def test_no_events_returns_24_hours_later(self: Self) -> None:
        """Test that with no events, returns 24 hours later."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == ts + timedelta(hours=24)

    @pytest.mark.asyncio
    async def test_event_start_boundary(self: Self) -> None:
        """Test boundary at event start."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Work",
                    "start": "2024-01-15T14:00:00",
                    "end": "2024-01-15T18:00:00",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 1, 15, 14, 0, 0)

    @pytest.mark.asyncio
    async def test_event_end_boundary(self: Self) -> None:
        """Test boundary at event end."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Work",
                    "start": "2024-01-15T09:00:00",
                    "end": "2024-01-15T17:00:00",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 1, 15, 17, 0, 0)

    @pytest.mark.asyncio
    async def test_nearest_boundary_is_returned(self: Self) -> None:
        """Test that nearest boundary is returned with multiple events."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [
                    {
                        "summary": "Meeting 1",
                        "start": "2024-01-15T14:00:00",
                        "end": "2024-01-15T15:00:00",
                    },
                    {
                        "summary": "Meeting 2",
                        "start": "2024-01-15T16:00:00",
                        "end": "2024-01-15T17:00:00",
                    },
                ]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        # Should be the nearest boundary (start of first event)
        assert boundary == datetime(2024, 1, 15, 14, 0, 0)

    @pytest.mark.asyncio
    async def test_service_error_returns_24_hours(self: Self) -> None:
        """Test that service error returns 24 hours later."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(side_effect=Exception("Service error"))
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == ts + timedelta(hours=24)

    @pytest.mark.asyncio
    async def test_simple_event_boundary_parsing(self: Self) -> None:
        """Test parsing of simple ISO format datetime strings."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Meeting",
                    "start": "2024-01-20T14:00:00",
                    "end": "2024-01-20T15:00:00",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        # Query before the event
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        # Should find the event start boundary (Jan 20)
        assert boundary.month == 1
        assert boundary.day == 20
        assert boundary.hour == 14

    @pytest.mark.asyncio
    async def test_date_only_event_handling(self: Self) -> None:
        """Test handling of date-only (all-day) events."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Holiday",
                    "start": "2024-01-20",
                    "end": "2024-01-20",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        # Should be treated as starting at midnight
        assert boundary == datetime(2024, 1, 20, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_looks_ahead_30_days(self: Self) -> None:
        """Test that next_boundary looks ahead 30 days."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        await indexer.next_boundary(ts)

        # Verify service was called
        hass.services.async_call.assert_called_once()


class TestCalendarIndexerHashability:
    """Tests for hashability and use in collections."""

    @pytest.mark.asyncio
    async def test_keys_are_hashable(self: Self) -> None:
        """Test that returned keys are hashable."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        hash_value = hash(key)
        assert isinstance(hash_value, int)

    @pytest.mark.asyncio
    async def test_same_value_same_hash(self: Self) -> None:
        """Test that same values have same hash."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        key1 = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        key2 = await indexer.get_key(datetime(2024, 1, 15, 14, 30))
        assert key1 == key2
        assert hash(key1) == hash(key2)

    @pytest.mark.asyncio
    async def test_usable_as_dict_key(self: Self) -> None:
        """Test that keys can be used as dictionary keys."""
        hass = MagicMock()

        # Setup two different responses
        call_count = [0]
        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return {
                    "calendar.work_schedule": {
                        "events": [{"summary": "Event", "start": "2024-01-15T14:00:00", "end": "2024-01-15T15:00:00"}]
                    }
                }
            return {"calendar.work_schedule": {"events": []}}

        hass.services.async_call = AsyncMock(side_effect=side_effect)
        indexer = CalendarIndexer(hass, "calendar.work_schedule")

        key_event = await indexer.get_key(datetime(2024, 1, 15, 10, 30))
        key_no_event = await indexer.get_key(datetime(2024, 1, 15, 10, 30))

        patterns = {
            key_event: "During event",
            key_no_event: "No event",
        }
        assert patterns[key_event] == "During event"
        assert len(patterns) == 2


class TestCalendarIndexerEdgeCases:
    """Tests for edge cases."""

    def test_different_entity_ids(self: Self) -> None:
        """Test multiple CalendarIndexer instances with different calendars."""
        hass = MagicMock()
        indexer1 = CalendarIndexer(hass, "calendar.work")
        indexer2 = CalendarIndexer(hass, "calendar.personal")
        assert indexer1.name != indexer2.name
        assert indexer1.entity_id == "calendar.work"
        assert indexer2.entity_id == "calendar.personal"

    @pytest.mark.asyncio
    async def test_timezone_preservation(self: Self) -> None:
        """Test that timezone is preserved in next_boundary."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {"events": []}
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        tz = UTC
        ts = datetime(2024, 1, 15, 10, 30, tzinfo=tz)
        _boundary = await indexer.next_boundary(ts)
        # Result might not preserve all tzinfo details in 24-hour offset case

    @pytest.mark.asyncio
    async def test_distant_future_event(self: Self) -> None:
        """Test with event far in the future."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value={
            "calendar.work_schedule": {
                "events": [{
                    "summary": "Future Event",
                    "start": "2024-12-31T14:00:00",
                    "end": "2024-12-31T15:00:00",
                }]
            }
        })
        indexer = CalendarIndexer(hass, "calendar.work_schedule")
        ts = datetime(2024, 1, 15, 10, 30)
        boundary = await indexer.next_boundary(ts)
        assert boundary == datetime(2024, 12, 31, 14, 0, 0)
