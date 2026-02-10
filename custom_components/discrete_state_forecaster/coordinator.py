"""Module of the Discrete State Forecaster Coordinator."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self, cast

from homeassistant.const import (
    EVENT_HOMEASSISTANT_STOP,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.event_type import EventType

from custom_components.discrete_state_forecaster.old_model.prediction import Prediction
from custom_components.discrete_state_forecaster.old_model.time_aware_forecaster import (
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.old_model.time_indexers.calendar_indexer import (
    CalendarIndexer,
)
from custom_components.discrete_state_forecaster.old_model.time_indexers.season_indexer import (
    SeasonIndexer,
)

from .const import (
    CONF_ADAPTIVE_PERSISTENCE,
    CONF_CALENDAR_FEATURES,
    CONF_HALF_LIFE_HOURS,
    CONF_STATE_PERSISTENCE_FACTOR,
    CONF_TARGET_ENTITY_ID,
    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
    CONF_USE_DAY_OF_WEEK,
    CONF_USE_MONTH_OF_YEAR,
    CONF_USE_SEASON,
    DEFAULT_ADAPTIVE_PERSISTENCE,
    DEFAULT_HALF_LIFE_HOURS,
    DEFAULT_STATE_PERSISTENCE_FACTOR,
    DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
    DEFAULT_USE_DAY_OF_WEEK,
    DEFAULT_USE_MONTH_OF_YEAR,
    DEFAULT_USE_SEASON,
    STORING_TIME_PATTERN,
)
from .old_model.state_tracker import StateTracker
from .old_model.time_indexers import (
    CompositeIndexer,
    DayOfWeekIndexer,
    MonthIndexer,
    TimeOfDayIndexer,
)

if TYPE_CHECKING:
    from . import DiscreteStateForecasterConfigEntry


@dataclass
class DiscreteStateForecasterCoordinatorState:
    """State of the Discrete State Forecaster Coordinator."""

    prediction: Prediction
    current_state: str | None
    timestamp: datetime
    next_transition_timestamp: datetime | None = None


class DiscreteStateForecasterCoordinator(
    DataUpdateCoordinator[DiscreteStateForecasterCoordinatorState]
):
    def __init__(
        self: Self,
        hass: HomeAssistant,
        config_entry: "DiscreteStateForecasterConfigEntry",
        logger: logging.Logger,
    ) -> None:
        super().__init__(
            hass,
            logger,
            name="discrete state forecaster coordinator",
            update_interval=None,
        )

        self._config_entry = config_entry
        # Get time bucket size from configuration
        time_bucket_minutes = int(
            config_entry.options.get(
                CONF_TIME_BUCKET_SIZE_IN_MINUTES, DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES
            )
        )

        # Build indexers based on configuration
        self._time_indexers = self._build_indexers(time_bucket_minutes, config_entry.options)
        self._composite_indexer = CompositeIndexer(self._time_indexers)

        # Get persistence settings from options
        state_persistence_factor = config_entry.options.get(
            CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
        )
        adaptive_persistence = config_entry.options.get(
            CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
        )
        half_life_hours = config_entry.options.get(CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS)

        self._forecaster = TimeAwareForecaster(
            self._composite_indexer,
            half_life=half_life_hours * 3600,  # Convert hours to seconds
            state_persistence_factor=state_persistence_factor,
            adaptive_persistence=adaptive_persistence,
        )
        self._state_tracker = StateTracker(self._forecaster)

        # Initialize storage for model persistence
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_forecaster.json"
        )
        self._unsubscribe_callbacks: list[CALLBACK_TYPE] = []
        self._unsubscribe_time_indexer_change_callback: CALLBACK_TYPE | None = None

        # Track current indexer configuration for detecting changes
        self._current_use_day_of_week = config_entry.options.get(
            CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        )
        self._current_use_month = config_entry.options.get(
            CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
        )
        self._current_use_season = config_entry.options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        self._current_calendar_features = config_entry.options.get(CONF_CALENDAR_FEATURES, [])
        self._current_time_bucket_size = time_bucket_minutes

        # Listen for config entry updates
        config_entry.async_on_unload(
            config_entry.add_update_listener(self._async_config_entry_updated)
        )

    async def async_start(self: Self) -> None:
        """
        Starts the coordinator.

        It starts periodic state updates, state storage, and listens for state changes.
        """
        # Restore stored state
        await self._async_restore_state()

        now = dt_util.now()

        # Initiate target entity state
        await self._handle_target_entity_state_change(
            Event(
                event_type=EventType("state_changed"),
                data={
                    "new_state": self.hass.states.get(self.config_entry.data[CONF_TARGET_ENTITY_ID])
                },
                time_fired_timestamp=now.timestamp(),
            )
        )

        # Start to listen to change of target entity
        self._unsubscribe_callbacks.append(
            async_track_state_change_event(
                self.hass,
                self.config_entry.data[CONF_TARGET_ENTITY_ID],
                self._handle_target_entity_state_change,
            )
        )

        # Initiate time index changes
        await self._track_next_time_indexer_change(now)

        async def _scheduled_store_state(now: datetime) -> None:  # noqa: ARG001
            await self._async_store_state()

        self._unsubscribe_callbacks.append(
            async_track_time_change(
                self.hass,
                _scheduled_store_state,
                **STORING_TIME_PATTERN,
            )
        )

        self.hass.bus.async_listen_once(
            self.hass,
            EVENT_HOMEASSISTANT_STOP,
            self._async_store_state,
        )

        await self.async_refresh()

    async def async_stop(self: Self) -> None:
        """Stops the coordinator."""
        for unsubscribe in self._unsubscribe_callbacks:
            unsubscribe()

        self._unsubscribe_callbacks.clear()

        if self._unsubscribe_time_indexer_change_callback:
            self._unsubscribe_time_indexer_change_callback()
            self._unsubscribe_time_indexer_change_callback = None

        await self._async_store_state()

    async def _async_restore_state(self: Self) -> None:
        """Restore forecaster state from storage."""
        data: dict[str, Any] | None = await self._store.async_load()

        if data:
            try:
                # Restore forecaster from saved state
                self._forecaster = TimeAwareForecaster.from_dict(data, self._composite_indexer)
                # Reconnect state tracker to restored forecaster
                self._state_tracker = StateTracker(self._forecaster)

                self.logger.info(
                    "Restored forecaster state for %s from storage.",
                    self._config_entry.data[CONF_TARGET_ENTITY_ID],
                )
            except (KeyError, TypeError, ValueError) as err:
                self.logger.warning(
                    "Invalid stored state for %s: %s. Starting with fresh model.",
                    self._config_entry.data[CONF_TARGET_ENTITY_ID],
                    err,
                )
        else:
            self.logger.debug(
                "No stored state found for %s. Starting with fresh model.",
                self._config_entry.data[CONF_TARGET_ENTITY_ID],
            )

    async def _async_store_state(self: Self, *_args) -> None:
        """Store forecaster state to persistent storage."""
        try:
            await self._store.async_save(self._forecaster.to_dict())
            self.logger.debug(
                "Forecaster state stored for %s. Config entry id: %s",
                self._config_entry.data[CONF_TARGET_ENTITY_ID],
                self._config_entry.entry_id,
            )
        except Exception:
            self.logger.exception(
                "Failed to store forecaster state for %s.",
                self._config_entry.data[CONF_TARGET_ENTITY_ID],
            )

    async def _async_update_data(self: Self) -> DiscreteStateForecasterCoordinatorState:
        """Fetches the latest prediction data."""
        self.logger.debug(
            "Updating data for %s. Current data: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            self.data,
        )

        # Get current time and make prediction
        now = dt_util.now()

        # Get current state from Home Assistant
        target_entity = self.hass.states.get(self._config_entry.data[CONF_TARGET_ENTITY_ID])
        current_state = self._get_state_to_store(target_entity)

        # Make prediction for current time
        prediction = await self._forecaster.predict(
            now,
            current_state=current_state,
        )

        # Calculate next transition time, using a binary search approach
        interval_minutes = 1440
        while interval_minutes > 1:
            next_transition_timestamp = await self._forecaster.find_next_transition(
                now, max_horizon_minutes=1440, interval_minutes=interval_minutes
            )
            if next_transition_timestamp is None:
                break
            interval_minutes = interval_minutes // 2

        return DiscreteStateForecasterCoordinatorState(
            prediction=prediction,
            current_state=current_state,
            timestamp=now,
            next_transition_timestamp=next_transition_timestamp,
        )

    async def _handle_target_entity_state_change(self: Self, event: Event) -> None:
        """Handle state changes of the forecasted entity."""
        self.logger.debug(
            "Forecasted entity state change detected for %s, Event: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            event,
        )

        new_state = cast("State", event.data.get("new_state"))
        new_state_to_store = self._get_state_to_store(new_state)

        if new_state_to_store is None:
            self.logger.debug(
                "Ignoring state change for %s as new state is None.",
                self._config_entry.data[CONF_TARGET_ENTITY_ID],
            )
            return

        self.logger.debug(
            "Updating state tracker for %s with new state: %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            new_state_to_store,
            event.time_fired,
        )

        await self._state_tracker.update(event.time_fired, new_state_to_store)

        await self.async_refresh()

    async def _handle_time_key_change(self: Self, now: datetime) -> None:
        key = await self._composite_indexer.key(now)

        self.logger.debug(
            "Time key change detected for %s at %s. Key: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            now,
            key,
        )

        await self._track_next_time_indexer_change(now)

        await self._state_tracker.update(now)

        await self.async_refresh()

    def _get_state_to_store(self: Self, state: State | None) -> str | None:
        """Get the stored state for an entity."""
        if state is None:
            return None

        return state.state if state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE) else None

    def _build_indexers(
        self: Self,
        time_bucket_minutes: int,
        options: dict[str, Any],
    ) -> list:
        """Build time indexers based on configuration."""
        indexers = []

        # Calendar feature indexers (optional)
        for calendar_feature in options.get(CONF_CALENDAR_FEATURES, []):
            indexers.append(CalendarIndexer(self.hass, calendar_feature))  # noqa: PERF401

        # Day of week indexer (optional)
        use_day_of_week = options.get(CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK)
        if use_day_of_week:
            indexers.append(DayOfWeekIndexer())

        # Month of year indexer (optional)
        use_month = options.get(CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR)
        if use_month:
            indexers.append(MonthIndexer())

        use_season = options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        if use_season:
            indexers.append(SeasonIndexer())

        # Time of day indexer (always included)
        indexers.append(TimeOfDayIndexer(time_bucket_minutes))

        return indexers

    async def _async_config_entry_updated(
        self: Self,
        hass: HomeAssistant,
        config_entry: "DiscreteStateForecasterConfigEntry",
    ) -> None:
        """Handle config entry updates (options flow changes)."""
        new_time_bucket_size = int(
            config_entry.options.get(
                CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
            )
        )
        new_use_day_of_week = config_entry.options.get(
            CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        )
        new_use_month = config_entry.options.get(CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR)
        new_use_season = config_entry.options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        new_calendar_features = config_entry.options.get(CONF_CALENDAR_FEATURES, [])

        # Check if indexer configuration changed
        indexers_changed = (
            new_use_day_of_week != self._current_use_day_of_week
            or new_use_month != self._current_use_month
            or new_use_season != self._current_use_season
            or new_calendar_features != self._current_calendar_features
            or new_time_bucket_size != self._current_time_bucket_size
        )

        if indexers_changed:
            self.logger.info(
                "Indexer configuration changed - resetting model. "
                "Time bucket size: %s -> %s, Day of week: %s -> %s, Month: %s -> %s, "
                "Season: %s -> %s, Calendar features: %s -> %s",
                self._current_time_bucket_size,
                new_time_bucket_size,
                self._current_use_day_of_week,
                new_use_day_of_week,
                self._current_use_month,
                new_use_month,
                self._current_use_season,
                new_use_season,
                self._current_calendar_features,
                new_calendar_features,
            )

            # Update tracked configuration
            self._current_use_day_of_week = new_use_day_of_week
            self._current_use_month = new_use_month
            self._current_calendar_features = new_calendar_features
            self._current_time_bucket_size = new_time_bucket_size

            # Rebuild indexers
            self._time_indexers = self._build_indexers(new_time_bucket_size, config_entry.options)
            self._composite_indexer = CompositeIndexer(self._time_indexers)

            # Get persistence settings from options
            state_persistence_factor = config_entry.options.get(
                CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
            )
            adaptive_persistence = config_entry.options.get(
                CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
            )
            half_life_hours = config_entry.options.get(
                CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
            )

            # Create new forecaster (resets the model)
            self._forecaster = TimeAwareForecaster(
                self._composite_indexer,
                half_life=half_life_hours * 3600,  # Convert hours to seconds
                state_persistence_factor=state_persistence_factor,
                adaptive_persistence=adaptive_persistence,
            )
            self._state_tracker = StateTracker(self._forecaster)

            # Delete old stored state
            await self._store.async_remove()

            self.logger.info("Model reset complete - learning from scratch")

            # Trigger update to refresh all entities
            await self.async_refresh()

    async def _track_next_time_indexer_change(self: Self, now: datetime) -> None:
        next_boundary = await self._composite_indexer.next_boundary(now)

        self.logger.debug(
            "Scheduling next time indexer change tracking for %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            next_boundary,
        )

        self._unsubscribe_time_indexer_change_callback = async_track_point_in_time(
            self.hass,
            self._handle_time_key_change,
            next_boundary,
        )
