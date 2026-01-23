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

from .binary_state_forecaster import BinaryStateForecaster, BinaryStateForecasterState
from .const import (
    CONF_BINARY_SENSOR,
    CONF_CALENDAR_FEATURES,
    CONF_FADING,
    CONF_FORECASTER_FEATURES,
    CONF_USE_DAY_OF_WEEK_FEATURE,
    DAY_OF_WEEK_FEATURE,
    STORING_TIME_PATTERN,
    TIME_BLOCK_PERIOD_IN_MINUTES,
)

if TYPE_CHECKING:
    from . import BinaryStateForecasterConfigEntry


@dataclass
class _StatePeriod:
    period_length: float
    state: str


class BinaryStateForecasterCoordinator(
    DataUpdateCoordinator[BinaryStateForecasterState]
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
        self._forecaster = BinaryStateForecaster
        self._config_entry = config_entry
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_state"
        )

        self._unsubscribe_callbacks: list[CALLBACK_TYPE] = []
        self._change_events: list[Event] = []
        self._previous_state_update_at: float | None = None
        self._current_time_block_index: int = self._calculate_time_block_index()
        self._all_features = sorted(
            config_entry.data.get(CONF_FORECASTER_FEATURES, [])
            + config_entry.data.get(CONF_CALENDAR_FEATURES, [])
            + (
                [DAY_OF_WEEK_FEATURE]
                if config_entry.data.get(CONF_USE_DAY_OF_WEEK_FEATURE, True)
                else []
            )
        )
        self._current_feature: tuple[bool | None, ...] = tuple(
            [None] * len(self._all_features)
        )
        self._forecasted_entity_current_state: str | None = None
        self._day_of_week: int | None

    async def async_start(self: Self) -> None:
        """
        Starts the coordinator.

        It starts periodic state updates, state storage, and listens for state changes.
        """
        # Restore stored state
        await self._async_restore_state()

        # Initialize current feature states from Home Assistant state
        for feature_entity in self._all_features:
            if feature_entity == DAY_OF_WEEK_FEATURE:
                self._set_day_of_week_feature(self._day_of_week)
                self._day_of_week = datetime.now(UTC).weekday()
            else:
                state = self.hass.states.get(feature_entity)
                self._set_feature_state(feature_entity, state, "on", "off")

        # Initialize the forecasted entity current state
        current_state = cast(
            State,
            self.hass.states.get(self.config_entry.data[CONF_BINARY_SENSOR]),
        )
        self._forecasted_entity_current_state = (
            current_state.state if current_state else None
        )

        # Initialize the internal state
        await self._update_internal_state(datetime.now(UTC).timestamp())

        async def _scheduled_update(now: datetime) -> None:
            self.logger.debug(
                "Scheduled update triggered for %s. Current time block index: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                self._current_time_block_index,
            )

            await self._update_internal_state(now.timestamp())

            self._current_time_block_index = self._calculate_time_block_index()

        async def _scheduled_store_state(now: datetime) -> None:  # noqa: ARG001
            await self._async_store_state()

        async def _day_of_week_changed(now: datetime) -> None:
            """Handle day-of-week changes."""
            if not self.config_entry.data.get(CONF_USE_DAY_OF_WEEK_FEATURE, True):
                return

            new_day = now.weekday()
            if new_day == self._day_of_week:
                return

            self.logger.debug(
                "Day of week changed for %s: %s -> %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                self._day_of_week,
                new_day,
            )

            self._day_of_week = new_day
            self._set_day_of_week_feature(new_day)

            # Recalculate internal state with the new day-of-week feature
            await self._update_internal_state(now.timestamp())

        async def _state_changed_listener(event: Event) -> None:
            """Handle state changes of the forecasted entity."""
            self.logger.debug(
                "Forecasted entity state change detected for %s, Event: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                event,
            )

            # We store the state.
            self._forecasted_entity_current_state = cast(
                State, event.data.get("new_state")
            ).state

            # We store the event ...
            self._change_events.append(event)

            # ... and perform an immediate update of internal state
            await self._update_internal_state(event.time_fired_timestamp)

        async def _calendar_feature_state_changed_listener(event: Event) -> None:
            """Handle state changes of the feature entities."""
            self.logger.debug(
                "Feature entity state change detected for %s, Event: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                event,
            )

            # We perform an immediate update of internal state
            await self._update_internal_state(event.time_fired_timestamp)

            # We update the current features
            state = self.hass.states.get(event.data.get("entity_id"))

            self._set_feature_state(event.data.get("entity_id"), state, "on", "off")
            feature_index = self._all_features.index(event.data.get("entity_id"))
            new_state = cast(State, event.data.get("new_state")).state

            self._current_feature = tuple(
                (
                    new_state == "on"
                    if index == feature_index
                    else self._current_feature[index]
                )
                for index in range(len(self._all_features))
            )

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
                    _calendar_feature_state_changed_listener,
                )
            )

        # for entry_id in self._config_entry.data.get(CONF_FORECASTER_FEATURES, []):
        #     entry = self.hass.config_entries.async_get_entry(entry_id)
        #     entity_id = entry.data.get(CONF_BINARY_SENSOR)

        #     self.logger.debug(
        #         "Setting up feature entity state change listener for %s: %s",
        #         self._config_entry.data.get(CONF_BINARY_SENSOR),
        #         entity_id,
        #     )

        #     state = self.hass.states.get(entry_id)
        #     self._set_feature_state(
        #         entry_id, state, "on", "off" # TODO: replace "on"/"off" with proper params
        #     )

        #     self._unsubscribe_callbacks.append(
        #         async_track_state_change_event(
        #             self.hass,
        #             entity_id,
        #             _feature_state_changed_listener,
        #         )
        #     )

        self._unsubscribe_callbacks.append(
            async_track_time_change(
                self.hass,
                _scheduled_update,
                minute=f"/{TIME_BLOCK_PERIOD_IN_MINUTES}",
                second=0,
            )
        )

        if self.config_entry.data.get(CONF_USE_DAY_OF_WEEK_FEATURE, True):
            # Listen for day changes at local midnight
            self._unsubscribe_callbacks.append(
                async_track_time_change(
                    self.hass,
                    _day_of_week_changed,
                    hour=0,
                    minute=0,
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

    async def _async_restore_state(self) -> None:
        data: dict[str, Any] | None = await self._store.async_load()
        forecaster: None | BinaryStateForecaster = None

        if data:
            try:
                forecaster = BinaryStateForecaster.from_dict(data)
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

        # TODO: load on/off state from config entry...
        self._forecaster = forecaster or BinaryStateForecaster()

    async def _async_store_state(self, *_args) -> None:
        await self._store.async_save(self._forecaster.to_dict())

        self.logger.debug(
            "Forecaster state stored for %s. Config entry id: %s",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
            self._config_entry.entry_id,
        )

    async def _async_update_data(self: Self) -> BinaryStateForecasterState:
        """Fetches the latest data."""
        return self._forecaster.state

    def _calculate_time_block_index(self: Self) -> int:
        """Calculate the current time block index based on the current time."""
        now = datetime.now(UTC)
        total_minutes = now.hour * 60 + now.minute
        return total_minutes // TIME_BLOCK_PERIOD_IN_MINUTES

    def _set_feature_state(
        self: Self,
        feature_entity_id: str,
        state: State | None,
        on_state: str,
        off_state: str,
    ) -> None:
        """Set the state of a feature entity."""
        feature_index = self._all_features.index(feature_entity_id)
        feature = (
            True
            if state and state.state == on_state
            else (False if state and state.state == off_state else None)
        )

        self._current_feature = tuple(
            (feature if index == feature_index else self._current_feature[index])
            for index in range(len(self._all_features))
        )

    def _set_day_of_week_feature(self: Self, day_of_week: int) -> None:
        """Set the day of week feature."""
        feature_index = self._all_features.index(DAY_OF_WEEK_FEATURE)

        self._current_feature = tuple(
            (day_of_week if index == feature_index else self._current_feature[index])
            for index in range(len(self._all_features))
        )

    async def _update_internal_state(self: Self, now: float) -> None:
        self.logger.debug(
            "Updating internal state for %s at %s. Features: %s, Change events: %s",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
            datetime.fromtimestamp(now, tz=UTC),
            [
                f"{x}: {y}"
                for x, y in zip(self._all_features, self._current_feature, strict=True)
            ],
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
                            if self._previous_state_update_at is not None and index == 0
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
                    cast(State, self._change_events[-1].data.get("new_state")).state,
                )
            )
        # We calculate total measured time
        total_measured_time = sum(
            interval.period_length for interval in state_intervals
        )

        # We filter only valid binary states which is defined in _on_state and _off_state.
        state_intervals = [
            interval
            for interval in state_intervals
            if interval.state
            in (self._forecaster._state._on_state, self._forecaster._state._off_state)
            and interval.period_length > 0
        ]

        self.logger.debug(
            "State intervals for the current update for %s: %s",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
            state_intervals,
        )

        if len(state_intervals) == 0:
            self.logger.debug(
                "No valid state intervals for %s, skipping probability calculation.",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
            )
        else:
            # We calculate the total time and the "on" time
            total_known_time = sum(
                interval.period_length for interval in state_intervals
            )

            # We calculate the effective fading based on the measured time
            effective_decay = self._config_entry.data.get(CONF_FADING, 0.0) * (
                total_known_time / total_measured_time
            )

            on_state_time = sum(
                interval.period_length
                for interval in state_intervals
                if interval.state == self._forecaster._state._on_state
            )

            self.logger.debug(
                "Total measured time for %s: %s, known time: %s, effective decay: %s",
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                total_measured_time,
                total_known_time,
                effective_decay,
            )

            self.logger.debug(
                "%s state time for %s: %s",
                self._forecaster._state._on_state,
                self._config_entry.data.get(CONF_BINARY_SENSOR),
                on_state_time,
            )

            if total_known_time == 0:
                self.logger.debug(
                    "Total known time is 0 for %s, skipping probability calculation.",
                    self._config_entry.data.get(CONF_BINARY_SENSOR),
                )
            else:
                probability = on_state_time / total_known_time

                self.logger.debug(
                    "Calculated probability for %s: %s",
                    self._config_entry.data.get(CONF_BINARY_SENSOR),
                    probability,
                )

                if None in self._current_feature:
                    self.logger.debug(
                        "Incomplete feature set for %s, skipping state update. Features: %s",
                        self._config_entry.data.get(CONF_BINARY_SENSOR),
                        [
                            f"{x}: {y}"
                            for x, y in zip(
                                self._all_features, self._current_feature, strict=True
                            )
                        ],
                    )
                else:
                    # We also need to update state for features based on probability
                    # and total_measured_time_ratio
                    previous_probability = self.data.get_probability(
                        self._current_feature, self._current_time_block_index
                    )

                    updated_probability = round(
                        (
                            (1 - effective_decay) * previous_probability
                            + effective_decay * probability
                            if previous_probability is not None
                            else probability
                        ),
                        5,
                    )

                    self.logger.debug(
                        "Updating probability for %s with features %s at time block index %s: "
                        "previous probability: %s, new probability: %s",
                        self._config_entry.data.get(CONF_BINARY_SENSOR),
                        [
                            f"{x}: {y}"
                            for x, y in zip(
                                self._all_features, self._current_feature, strict=True
                            )
                        ],
                        self._current_time_block_index,
                        previous_probability,
                        updated_probability,
                    )

                    self.data.update_probability(
                        self._current_feature,
                        self._current_time_block_index,
                        updated_probability,
                    )

                    # TODO: remove saving
                    await self._async_store_state()

                # We remove all processed events except the last one (to keep the latest state)
                if len(self._change_events) > 1:
                    self._change_events = [self._change_events[-1]]

                # We save the update time of internal state
                # This will be used to calculate the next period start.
                self._previous_state_update_at = now

        await self.async_refresh()

    def get_current_probability(self: Self) -> float | None:
        """Get the current probability based on current features and time block index."""
        self.logger.debug(
            "Getting current probability for %s with features %s at time block index %s",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
            [
                f"{x}: {y}"
                for x, y in zip(self._all_features, self._current_feature, strict=True)
            ],
            self._current_time_block_index,
        )

        return self.data.get_probability(
            self._current_feature, self._current_time_block_index
        )

    def get_current_state(self: Self) -> str | None:
        """Get the current state based on current features."""
        self.logger.debug(
            "Getting current state for %s with features %s: %s",
            self._config_entry.data.get(CONF_BINARY_SENSOR),
            [
                f"{x}: {y}"
                for x, y in zip(self._all_features, self._current_feature, strict=True)
            ],
            self._forecasted_entity_current_state,
        )

        return self._forecasted_entity_current_state
