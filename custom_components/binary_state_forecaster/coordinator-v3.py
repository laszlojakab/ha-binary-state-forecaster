"""Module of the Binary State Forecaster Coordinator."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    CONF_CALENDAR_FEATURES,
    CONF_FORECASTER_FEATURES,
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

        self._unsubscribe_callbacks: list[CALLBACK_TYPE] = []
        self._on_state: Final = "on"
        self._off_state: Final = "off"
        self._change_events: list[Event] = []
        self._previous_state_update_at: float | None = None

    async def async_start(self: Self) -> None:
        """
        Starts the coordinator.

        It starts periodic state updates, state storage, and listens for state changes.
        """
        await self._async_restore_state()

        async def _update_internal_state(now: float) -> None:
            self.logger.debug(
                "Updating internal state for %s at %s. Change events: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                datetime.fromtimestamp(now, tz=UTC),
                self._change_events,
            )

            state_intervals: list[_StatePeriod] = []

            # We take all periods from the saved change events
            # and calculate state intervals from them
            for index, period in enumerate(
                zip(self._change_events, self._change_events[1:], strict=False)
            ):
                curr_event, next_event = period
                state_intervals.append(
                    _StatePeriod(
                        (
                            next_event.time_fired_timestamp
                            - (
                                # If there was a previous state update and
                                # we are calculating the first interval,
                                # then we need to use that time as the start
                                # of the interval.
                                self._previous_state_update_at
                                if self._previous_state_update_at is not None
                                and index == 0
                                else curr_event.time_fired_timestamp
                            )
                        ),
                        cast(State, curr_event.data.get("new_state")).state,
                    )
                )

            # We also need to account for the last event until now interval
            if len(self._change_events):
                state_intervals.append(
                    _StatePeriod(
                        now - self._change_events[-1].time_fired_timestamp,
                        cast(
                            State, self._change_events[-1].data.get("new_state")
                        ).state,
                    )
                )

            # We filter only valid binary states which is defined in _on_state and _off_state.
            state_intervals = [
                interval
                for interval in state_intervals
                if interval.state in (self._on_state, self._off_state)
                and interval.period_length > 0
            ]

            self.logger.debug(
                "State intervals for the current update for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                state_intervals,
            )

            # We calculate the total time and the "on" time
            total_time = sum(interval.period_length for interval in state_intervals)
            on_time = sum(
                interval.period_length
                for interval in state_intervals
                if interval.state == self._on_state
            )

            self.logger.debug(
                "Total measured time for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                total_time,
            )

            self.logger.debug(
                "%s state time for %s: %s",
                self._on_state,
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                on_time,
            )

            if total_time == 0:
                self.logger.debug(
                    "No valid state intervals for %s, skipping probability calculation.",
                    self._config_entry.data.get(CONF_BINARY_SENSOR),
                )
                return

            probability = on_time / total_time
            total_measured_time_ratio = min(
                1.0, total_time / (TIME_BLOCK_PERIOD_IN_MINUTES * 60)
            )

            self.logger.debug(
                "Calculated probability for %s: %s, " "total measured time ratio: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                probability,
                total_measured_time_ratio,
            )

            # TODO: we need to determine features and timestamp_index
            # We also need to update state for features based on probability and total_measured_time_ratio

            # We remove all processed events except the last one (to keep the latest state)
            if len(self._change_events) > 1:
                self._change_events = [self._change_events[-1]]

            # We save the update time of internal state
            # This will be used to calculate the next period start.
            self._previous_state_update_at = now

            await self.async_refresh()

        async def _scheduled_update(now: datetime) -> None:
            self.logger.debug(
                "Scheduled update triggered for %s.",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
            )

            await _update_internal_state(now.timestamp())

        async def _scheduled_store_state(now: datetime) -> None:  # noqa: ARG001
            await self._async_store_state()

        async def _state_changed_listener(event: Event) -> None:
            """Handle state changes of the forecasted entity."""
            self.logger.debug(
                "Forecasted entity state change detected for %s, Event: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                event,
            )

            # We store the event ...
            self._change_events.append(event)

            # ... and perform an immediate update of internal state
            await _update_internal_state(event.time_fired_timestamp)

        async def _feature_state_changed_listener(event: Event) -> None:
            """Handle state changes of the feature entities."""
            self.logger.debug(
                "Feature entity state change detected for %s, Event: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                event,
            )

            # # We perform an immediate update of internal state
            # await _update_internal_state(event.time_fired_timestamp)

        # TODO: get forecaster features target and listen to their state changes too

        for calendar_entity in self._config_entry.data.get(CONF_CALENDAR_FEATURES, []):
            self.logger.debug(
                "Setting up calendar feature entity state change listener for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                calendar_entity,
            )

            self._unsubscribe_callbacks.append(
                async_track_state_change_event(
                    self.hass,
                    calendar_entity,
                    _feature_state_changed_listener,
                )
            )

        for entry_id in self._config_entry.data.get(CONF_FORECASTER_FEATURES, []):
            entry = self.hass.config_entries.async_get_entry(entry_id)
            entity_id = entry.data.get(CONF_BINARY_SENSOR)

            self.logger.debug(
                "Setting up feature entity state change listener for %s: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                entity_id,
            )

            self._unsubscribe_callbacks.append(
                async_track_state_change_event(
                    self.hass,
                    entity_id,
                    _feature_state_changed_listener,
                )
            )

        self._unsubscribe_callbacks.append(
            async_track_time_change(
                self.hass,
                _scheduled_update,
                minute=f"/{TIME_BLOCK_PERIOD_IN_MINUTES}",
                second=0,
            )
        )

        self._unsubscribe_callbacks.append(
            async_track_time_change(
                self.hass,
                _scheduled_store_state,
                **STORING_TIME_PATTERN,
            )
        )

        self._unsubscribe_callbacks.append(
            async_track_state_change_event(
                self.hass,
                self.config_entry.data[CONF_BINARY_SENSOR],
                _state_changed_listener,
            )
        )

        self.hass.bus.async_listen_once(
            self.hass,
            EVENT_HOMEASSISTANT_STOP,
            self._async_store_state,
        )

        await self.async_refresh()

    async def async_stop(self) -> None:
        """Stops the coordinator."""
        for unsubscribe in self._unsubscribe_callbacks:
            unsubscribe()

        self._unsubscribe_callbacks.clear()

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
