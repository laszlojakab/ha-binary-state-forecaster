"""Module of the Binary State Forecaster Coordinator."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Self, cast

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_BINARY_SENSOR,
    STORING_TIME_PATTERN,
    TIME_BLOCK_PERIOD_IN_MINUTES,
)

if TYPE_CHECKING:
    from . import BinaryStateForecasterConfigEntry


@dataclass
class _StatePeriod:
    period_length: float
    state: str


class BinaryStateForecasterCoordinatorState:
    """The internal state of binary state forecaster coordinator."""

    def __init__(
        self: Self, state: dict[tuple[Any, ...], list[float | None]] | None = None
    ) -> None:
        """Initialize the coordinator state."""
        self._state: dict[tuple[Any, ...], list[float | None]] = state or {}

    def get_probability(
        self: Self, features: tuple[Any, ...], time_block_index: int
    ) -> float | None:
        """Get the probability for the given features and time block index."""
        probabilities = self._state.get(features)
        if probabilities is None:
            return None

        value_at_time_block_index = probabilities[time_block_index]

        if value_at_time_block_index is None:
            # Find closest non-None values before and after
            left_value = None
            left_index = None
            for i in range(time_block_index - 1, -1, -1):
                if probabilities[i] is not None:
                    left_value = probabilities[i]
                    left_index = i
                    break

            right_value = None
            right_index = None
            for i in range(time_block_index + 1, len(probabilities)):
                if probabilities[i] is not None:
                    right_value = probabilities[i]
                    right_index = i
                    break

            # Interpolate if both left and right values exist
            if left_value is not None and right_value is not None:
                # Linear interpolation
                total_distance = right_index - left_index
                distance_from_left = time_block_index - left_index
                weight = distance_from_left / total_distance
                value_at_time_block_index = (
                    left_value + (right_value - left_value) * weight
                )

        return value_at_time_block_index

    def update_probability(
        self: Self, features: tuple[Any, ...], timestamp_index: int, probability: float
    ) -> None:
        """Update the probability for the given features and timestamp index."""
        if features not in self._state:
            self._state[features] = [None] * int(
                timedelta(hours=24).total_seconds()
                / (TIME_BLOCK_PERIOD_IN_MINUTES * 60)
            )

        self._state[features][timestamp_index] = probability

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any],
    ) -> Self:
        """Create an instance from a dictionary."""
        state_dict: dict = data.get("state", {})
        # Convert JSON string keys back to tuples
        restored_state: dict[tuple[Any, ...], list[float]] = {
            tuple(json.loads(key)): value for key, value in state_dict.items()
        }
        return cls(state=restored_state)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        # Convert tuple keys to JSON strings for serialization
        serializable_state = {
            json.dumps(key): value for key, value in self._state.items()
        }
        return {"state": serializable_state}


class BinaryStateForecasterCoordinator(
    DataUpdateCoordinator[BinaryStateForecasterCoordinatorState]
):
    """Coordinator for managing binary state forecasting updates."""

    def __init__(
        self: Self,
        hass: HomeAssistant,
        config_entry: "BinaryStateForecasterConfigEntry",
        logger: logging.Logger,
    ) -> None:
        """
        Initializes a new instance of HungaroMet10MinDataUpdateCoordinator class.

        Args:
          hass: The Home Assistant instance.
          config_entry: The configuration entry for the integration.
          logger: The logger instance.
        """
        super().__init__(
            hass,
            logger,
            name="binary state forecaster coordinator",
            update_interval=None,
        )
        self._config_entry = config_entry
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_state"
        )
        self._unsubscribe_scheduled_update_interval: None | CALLBACK_TYPE = None
        self._unsubscribe_scheduled_store_interval: None | CALLBACK_TYPE = None
        self._unsubscribe_state_change: None | CALLBACK_TYPE = None
        self._on_state: Final = "on"
        self._off_state: Final = "off"
        self._latest_time_block_last_state_change_event: Event | None = None
        self._saved_change_events: list[Event] = []

    async def async_start(self: Self) -> None:
        """
        Starts the coordinator.

        It starts periodic state updates, state storage, and listens for state changes.
        """
        await self._async_restore_state()

        async def _scheduled_update(now: datetime) -> None:
            self.logger.debug(
                "Scheduled update triggered for %s.",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
            )

            self.logger.debug(
                "Latest time block last state stored for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                self._latest_time_block_last_state_change_event,
            )

            self.logger.debug(
                "Updating internal state for time block for %s.",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
            )

            self.logger.debug(
                "Saved change events for the current time block for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                self._saved_change_events,
            )

            state_intervals: list[_StatePeriod] = []
            time_block_start = now.timestamp() - TIME_BLOCK_PERIOD_IN_MINUTES * 60

            if not self._saved_change_events:
                # No state changes during this time block - state remained constant
                # Use the last known state from the previous block, or default to off
                constant_state = self._off_state
                if self._latest_time_block_last_state_change_event is not None:
                    new_state = (
                        self._latest_time_block_last_state_change_event.data.get(
                            "new_state"
                        )
                    )
                    if new_state is not None:
                        constant_state = cast(State, new_state).state

                state_intervals.append(
                    _StatePeriod(
                        TIME_BLOCK_PERIOD_IN_MINUTES * 60,
                        constant_state,
                    )
                )
            else:
                # We have state changes - build intervals
                # Start with the state at the beginning of the time block
                initial_state = self._off_state
                if self._latest_time_block_last_state_change_event is not None:
                    new_state = (
                        self._latest_time_block_last_state_change_event.data.get(
                            "new_state"
                        )
                    )
                    if new_state is not None:
                        initial_state = cast(State, new_state).state

                # Create interval from time block start to first event
                first_event_old_state = self._saved_change_events[0].data.get(
                    "old_state"
                )
                if first_event_old_state is not None:
                    initial_state = cast(State, first_event_old_state).state

                state_intervals.append(
                    _StatePeriod(
                        self._saved_change_events[0].time_fired_timestamp
                        - time_block_start,
                        initial_state,
                    )
                )

                # Create intervals for each state change
                for index, event in enumerate(self._saved_change_events):
                    new_state_obj = event.data.get("new_state")
                    if new_state_obj is None:
                        continue

                    state = cast(State, new_state_obj).state

                    # Calculate duration until next event or now
                    if index < len(self._saved_change_events) - 1:
                        duration = (
                            self._saved_change_events[index + 1].time_fired_timestamp
                            - event.time_fired_timestamp
                        )
                    else:
                        # Last event - duration until now
                        duration = now.timestamp() - event.time_fired_timestamp

                    state_intervals.append(_StatePeriod(duration, state))

            # Filter out intervals with states other than on/off
            state_intervals = [
                interval
                for interval in state_intervals
                if interval.state in [self._on_state, self._off_state]
            ]

            self.logger.debug(
                "State intervals for the current time block for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                state_intervals,
            )

            # We calculate the weighted probability of being in the "on" state
            total_time = sum(interval.period_length for interval in state_intervals)
            on_time = sum(
                interval.period_length
                for interval in state_intervals
                if interval.state == self._on_state
            )
            probability = on_time / total_time if total_time > 0 else 0.0

            total_measured_time_ratio = total_time / (TIME_BLOCK_PERIOD_IN_MINUTES * 60)

            self.logger.debug(
                "Calculated probability for the current time block for %s: %s, "
                "total measured time ratio: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                probability,
                total_measured_time_ratio,
            )

            # TODO: update state probabilities

            # We save the latest event for the next time block processing
            self._latest_time_block_last_state_change_event = next(
                (
                    evt
                    for evt in reversed(self._saved_change_events)
                    if cast(State, evt.data.get("new_state")).state == self._on_state
                    or cast(State, evt.data.get("new_state")).state == self._off_state
                ),
                None,
            )

            # We clear saved events after processing
            self._saved_change_events.clear()

            await self.async_refresh()

        async def _scheduled_store_state(now: datetime) -> None:  # noqa: ARG001
            await self._async_store_state()

        async def _state_changed_listener(event: Event) -> None:
            """Handle state changes of the predicted entity."""
            self.logger.debug(
                "Predicted entity state change detected for %s, saving state: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                event.data,
            )
            self._saved_change_events.append(event)

        self._unsubscribe_scheduled_update_interval = async_track_time_change(
            self.hass,
            _scheduled_update,
            minute=f"/{TIME_BLOCK_PERIOD_IN_MINUTES}",
            second=0,
        )

        self._unsubscribe_scheduled_store_interval = async_track_time_change(
            self.hass,
            _scheduled_store_state,
            minute=0,
            second=0,
            hour="*",
        )

        self._unsubscribe_state_change = async_track_state_change_event(
            self.hass,
            self.config_entry.data[CONF_BINARY_SENSOR],
            _state_changed_listener,
        )

        self.hass.bus.async_listen_once(
            self.hass,
            EVENT_HOMEASSISTANT_STOP,
            self._async_store_state,
        )

        await self.async_refresh()

    async def async_stop(self) -> None:
        """Stops the coordinator."""
        if self._unsubscribe_scheduled_update_interval:
            self._unsubscribe_scheduled_update_interval()
            self._unsubscribe_scheduled_update_interval = None

        if self._unsubscribe_scheduled_store_interval:
            self._unsubscribe_scheduled_store_interval()
            self._unsubscribe_scheduled_store_interval = None

        if self._unsubscribe_state_change:
            self._unsubscribe_state_change()
            self._unsubscribe_state_change = None

        await self._async_store_state()

    async def _async_update_data(self) -> BinaryStateForecasterCoordinatorState:
        return BinaryStateForecasterCoordinatorState()

    async def _async_restore_state(self) -> None:
        data: dict[str, Any] | None = await self._store.async_load()
        state: None | BinaryStateForecasterCoordinatorState = None

        if data:
            try:
                state = BinaryStateForecasterCoordinatorState.from_dict(data)
            except (KeyError, TypeError) as err:
                self.logger.warning(
                    "Invalid stored state for %s: %s",
                    self._config_entry.data.get(CONF_BINARY_SENSOR),
                    err,
                )
        else:
            self.logger.debug(
                "No stored state found for %s.",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
            )

        self.data = state or BinaryStateForecasterCoordinatorState()

    async def _async_store_state(self, *_args) -> None:
        await self._store.async_save(self.data.to_dict())

        self.logger.debug(
            "Coordinator state stored for %s.",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
        )
