"""
Calendar-based indexer for event-aware pattern analysis.

This module implements a time indexer that partitions time based on Home Assistant
calendar events, enabling modeling of patterns that correlate with scheduled events
(e.g., workday hours, vacation periods, holidays).
"""

from datetime import datetime, timedelta
from typing import Final, Self

from homeassistant.core import HomeAssistant

from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey

from .time_indexer import (
    TimeIndexer,
)


class CalendarIndexer(TimeIndexer):
    """
    Indexes timestamps by calendar event presence.

    Maps each timestamp to a binary state based on whether an active calendar
    event is present at that time. This allows the forecaster to learn different
    patterns for when calendar events are active versus inactive.

    The indexer queries Home Assistant's calendar integration to determine event
    boundaries and active states. This enables predictions that adapt to scheduled
    events like work hours, vacations, holidays, or any other calendar-tracked
    activities.

    Attributes:
        name: Unique identifier in format "calendar:{entity_id}".
        hass: Home Assistant instance for accessing calendar services.
        entity_id: The calendar entity ID to query for events.

    Example:
        ```
        >>> indexer = CalendarIndexer(hass, "calendar.work_schedule")
        >>> # At timestamp during a calendar event
        >>> await indexer.key(datetime(2024, 1, 15, 10, 0))  # Returns 1
        >>> # At timestamp outside any calendar event
        >>> await indexer.key(datetime(2024, 1, 15, 18, 0))  # Returns 0
        ```

    Note:
        This indexer requires the calendar integration to be configured and
        the specified entity_id to exist in Home Assistant.

    """

    def __init__(self: Self, hass: HomeAssistant, entity_id: str) -> None:
        """
        Initialize the calendar-based indexer.

        Args:
            hass: Home Assistant instance for accessing calendar services.
            entity_id: The calendar entity ID to query for events
                (e.g., "calendar.work_schedule").

        Raises:
            ValueError: If entity_id is empty or invalid format.

        """
        if not entity_id:
            msg = "entity_id cannot be empty"
            raise ValueError(msg)

        if not entity_id.startswith("calendar."):
            msg = f"entity_id must start with 'calendar.', got: {entity_id}"
            raise ValueError(msg)

        self.hass: Final[HomeAssistant] = hass
        self.entity_id: Final[str] = entity_id
        self.name: Final[str] = f"{entity_id}"

    async def get_key(self: Self, timestamp: datetime) -> TimeKey:
        """
        Returns whether a calendar event is active at the given timestamp.

        Queries the Home Assistant calendar service to determine if any event
        is active at the specified time.

        Args:
            timestamp: The timestamp to check for calendar events.

        Returns:
            1 if a calendar event is active at the timestamp, 0 otherwise.

        Example:
            >>> # During a calendar event
            >>> await indexer.key(datetime(2024, 1, 15, 10, 0))
            1
            >>> # No calendar event
            >>> await indexer.key(datetime(2024, 1, 15, 18, 0))
            0

        """
        # Query for events that overlap with this timestamp
        # Using a 1-second window to check if any event is active at timestamp
        start_time = timestamp
        end_time = timestamp + timedelta(seconds=1)

        try:
            response = await self.hass.services.async_call(
                "calendar",
                "get_events",
                {
                    "entity_id": self.entity_id,
                    "start_date_time": start_time.isoformat(),
                    "end_date_time": end_time.isoformat(),
                },
                blocking=True,
                return_response=True,
            )

            # Extract events from response
            events = response.get(self.entity_id, {}).get("events", [])

            # Return 1 if any events are active, 0 otherwise
            return TimeKey.from_tuple(((self.name, 1 if events else 0),))

        except Exception:  # noqa: BLE001
            # If calendar service fails, assume no event
            return TimeKey.from_tuple(((self.name, 0),))

    async def next_boundary(self: Self, timestamp: datetime) -> datetime:
        """
        Returns the next calendar event boundary after the given timestamp.

        Finds the soonest time when the calendar state changes - either when
        an event starts (if currently no event) or when an event ends (if
        currently in an event).

        Args:
            timestamp: The timestamp to find the next boundary after.

        Returns:
            The timestamp of the next calendar state change. If no future
            events are known, returns timestamp + 24 hours as a default.

        Example:
            >>> # Currently no event, next event starts at 9 AM
            >>> await indexer.next_boundary(datetime(2024, 1, 15, 8, 0))
            datetime(2024, 1, 15, 9, 0, 0)
            >>> # Currently in event ending at 5 PM
            >>> await indexer.next_boundary(datetime(2024, 1, 15, 14, 0))
            datetime(2024, 1, 15, 17, 0, 0)

        """
        # Query future events (look ahead 30 days)
        start_time = timestamp
        end_time = timestamp + timedelta(days=30)

        try:
            response = await self.hass.services.async_call(
                "calendar",
                "get_events",
                {
                    "entity_id": self.entity_id,
                    "start_date_time": start_time.isoformat(),
                    "end_date_time": end_time.isoformat(),
                },
                blocking=True,
                return_response=True,
            )

            # Extract events from response
            events = response.get(self.entity_id, {}).get("events", [])

            if not events:
                # No future events, return default
                return timestamp + timedelta(hours=24)

            # Collect all event boundaries (start and end times) after timestamp
            boundaries = []
            for event in events:
                # Parse event times
                start_str = event.get("start")
                end_str = event.get("end")

                if start_str:
                    # Handle both datetime strings and date-only strings
                    if "T" in start_str:
                        event_start = datetime.fromisoformat(
                            start_str.replace("Z", "+00:00")
                        )
                    else:
                        # Date-only event (all-day)
                        event_start = datetime.fromisoformat(start_str + "T00:00:00")

                    if event_start > timestamp:
                        boundaries.append(event_start)

                if end_str:
                    # Handle both datetime strings and date-only strings
                    if "T" in end_str:
                        event_end = datetime.fromisoformat(
                            end_str.replace("Z", "+00:00")
                        )
                    else:
                        # Date-only event (all-day)
                        event_end = datetime.fromisoformat(end_str + "T00:00:00")

                    if event_end > timestamp:
                        boundaries.append(event_end)

            # Return the nearest boundary
            if boundaries:
                return min(boundaries)

            return timestamp + timedelta(hours=24)

        except Exception:  # noqa: BLE001
            # If calendar service fails, return default
            return timestamp + timedelta(hours=24)
