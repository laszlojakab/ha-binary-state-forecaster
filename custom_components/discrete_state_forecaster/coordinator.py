"""Module of the Discrete State Forecaster Coordinator."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self, cast

from homeassistant.const import (
    EVENT_HOMEASSISTANT_STOP,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.event_type import EventType

from .const import (
    CONF_CALENDAR_FEATURES,
    CONF_DECAY_SECONDS,
    CONF_FORECASTER_FEATURES,
    CONF_TARGET_ENTITY_ID,
    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
    CONF_USE_DAY_OF_WEEK_FEATURE,
    FEATURE_DAY_OF_WEEK,
    FEATURE_TIME_BUCKET,
    STORING_TIME_PATTERN,
)
from .discrete_conditional_model import (
    DiscreteConditionalModel,
    FeatureLabel,
    FeatureName,
    TargetLabel,
)

if TYPE_CHECKING:
    from . import DiscreteStateForecasterConfigEntry


@dataclass
class _StatePeriod:
    timestamp: float
    state: str | None


class DiscreteStateForecasterCoordinator(
    DataUpdateCoordinator[None]  # TODO: state type
):
    """Coordinator for managing discrete state forecasting updates."""

    def __init__(
        self: Self,
        hass: HomeAssistant,
        config_entry: "DiscreteStateForecasterConfigEntry",
        logger: logging.Logger,
    ) -> None:
        """
        Initializes a new instance of DiscreteStateForecasterCoordinator class.

        Args:
          hass: The Home Assistant instance.
          config_entry: The configuration entry for the integration.
          logger: The logger instance.
        """
        super().__init__(
            hass,
            logger,
            name="discrete state forecaster coordinator",
            update_interval=None,
        )
        self._config_entry = config_entry
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_forecaster.json"
        )
        self._states: dict[FeatureLabel | TargetLabel, _StatePeriod] = {}
        self._unsubscribe_callbacks: list[CALLBACK_TYPE] = []
        self._t: None | int = None
        self._all_features: list[FeatureName] = sorted(
            [FEATURE_TIME_BUCKET]
            + (
                [FEATURE_DAY_OF_WEEK]
                if config_entry.data.get(CONF_USE_DAY_OF_WEEK_FEATURE, False)
                else []
            )
            + config_entry.data.get(CONF_CALENDAR_FEATURES, [])
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
                    "new_state": self.hass.states.get(
                        self.config_entry.data[CONF_TARGET_ENTITY_ID]
                    )
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

        # Initiate calendar entity states
        for calendar_entity in self.config_entry.data.get(CONF_CALENDAR_FEATURES, []):
            self._handle_state_change(
                now.timestamp(),
                calendar_entity,
                self.get_state_to_store(
                    self.hass.states.get(calendar_entity),
                ),
            )

            # Start to listen to state changes of calendar entities
            self._unsubscribe_callbacks.append(
                async_track_state_change_event(
                    self.hass,
                    calendar_entity,
                    self._handle_calendar_feature_state_change,
                )
            )

        # Initiate time bucket feature
        bucket_size_in_minutes: int = self.config_entry.data.get(
            CONF_TIME_BUCKET_SIZE_IN_MINUTES, 60
        )
        hours = bucket_size_in_minutes // 60
        minutes = bucket_size_in_minutes % 60

        time_bucket_config = {
            "second": "0",
            "minute": f"/{minutes}" if minutes > 0 else ("0" if hours > 0 else "*"),
            "hour": f"/{hours}" if hours > 0 else "*",
        }

        self.logger.debug(
            "Setting up time bucket change listener for %s with bucket size %s. Configuration: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            bucket_size_in_minutes,
            time_bucket_config,
        )

        await self._handle_time_bucket_change(now)

        # Start to listen to time bucket changes
        self._unsubscribe_callbacks.append(
            async_track_time_change(
                self.hass,
                self._handle_time_bucket_change,
                **time_bucket_config,
            )
        )

        if self.config_entry.data.get(CONF_USE_DAY_OF_WEEK_FEATURE, False):
            # Initiate day of week feature
            await self._handle_day_of_week_change(now)

            # Start to listen to day changes at local midnight
            self._unsubscribe_callbacks.append(
                async_track_time_change(
                    self.hass,
                    self._handle_day_of_week_change,
                    hour=0,
                    minute=0,
                    second=0,
                )
            )

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

    async def async_stop(self) -> None:
        """Stops the coordinator."""
        for unsubscribe in self._unsubscribe_callbacks:
            unsubscribe()

        self._unsubscribe_callbacks.clear()

        await self._async_store_state()

    async def _async_restore_state(self) -> None:
        data: dict[str, Any] | None = await self._store.async_load()
        model: None | DiscreteConditionalModel = None

        if data:
            try:
                model = DiscreteConditionalModel.from_dict(data)
            except (KeyError, TypeError) as err:
                self.logger.warning(
                    "Invalid stored state for %s: %s",
                    self._config_entry.data[CONF_TARGET_ENTITY_ID],
                    err,
                )
        else:
            self.logger.debug(
                "No stored state found for %s.",
                self._config_entry.data[CONF_TARGET_ENTITY_ID],
            )

        self._model = model or DiscreteConditionalModel()
        self._model.decay = self.config_entry.data.get(CONF_DECAY_SECONDS, 3600.0)

    async def _async_store_state(self, *_args) -> None:
        await self._store.async_save(self._model.to_dict())

        self.logger.debug(
            "Forecaster state stored for %s. Config entry id: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            self._config_entry.entry_id,
        )

    async def _async_update_data(self: Self) -> None:  # BinaryStateForecasterState:
        """Fetches the latest data."""
        return None  # self._forecaster.state

    def _store_state(
        self: Self, entity_id: str, timestamp: float, state: State
    ) -> None:
        self._t = timestamp
        self._states[entity_id] = _StatePeriod(
            timestamp,
            self.get_state_to_store(state),
        )

    async def _handle_target_entity_state_change(self, event: Event) -> None:
        """Handle state changes of the forecasted entity."""
        self.logger.debug(
            "Forecasted entity state change detected for %s, Event: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            event,
        )

        new_state = cast(State, event.data.get("new_state"))

        self._handle_state_change(
            event.time_fired_timestamp,
            event.data.get("entity_id"),
            self.get_state_to_store(new_state),
        )

    async def _handle_calendar_feature_state_change(self, event: Event) -> None:
        """Handle state changes of the calendar feature entities."""
        self.logger.debug(
            "Calendar feature entity state change detected for %s, Event: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            event,
        )

        new_state = cast(State, event.data.get("new_state"))

        self._handle_state_change(
            event.time_fired_timestamp,
            event.data.get("entity_id"),
            self.get_state_to_store(new_state),
        )

    async def _handle_time_bucket_change(self, now: datetime) -> None:
        self.logger.debug(
            "Time bucket change detected for %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            now,
        )

        bucket_size_in_minutes: int = self.config_entry.data.get(
            CONF_TIME_BUCKET_SIZE_IN_MINUTES, 60
        )

        current_bucket_index = (now.hour * 60 + now.minute) // bucket_size_in_minutes

        self._handle_state_change(
            now.timestamp(), FEATURE_TIME_BUCKET, current_bucket_index
        )

    async def _handle_day_of_week_change(self, now: datetime) -> None:
        self.logger.debug(
            "Day of week change detected for %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            now,
        )

        self._handle_state_change(now.timestamp(), FEATURE_DAY_OF_WEEK, now.weekday())

    def _handle_state_change(
        self, timestamp: float, feature: str, state: int | str | None
    ) -> None:
        self.logger.debug(
            "State change detected for %s: feature or target=%s, state=%s at %s.",
            self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
            feature,
            state,
            datetime.fromtimestamp(timestamp, tz=UTC),
        )

        current_target_label = self._states.get(
            self.config_entry.data[CONF_TARGET_ENTITY_ID], None
        )

        if self._t is not None and current_target_label is not None:
            duration = timestamp - self._t
            features = {
                feature_name: self._states.get(feature_name).state
                for feature_name in self._all_features
                if self._states.get(feature_name, None) is not None
            }
            self.logger.debug(
                "Updating model for %s with features: %s, target label: %s, duration: %s",
                self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
                features,
                current_target_label.state,
                duration,
            )

            self._model.update(features, current_target_label.state, duration)

            confidence = self._model.confidence(features)

            self.logger.debug(
                "Model updated for %s. Confidence: %s",
                self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
                confidence,
            )
        else:
            self.logger.debug(
                "Not updating model for %s as previous time or target label is missing. "
                "Previous time: %s, target label: %s",
                self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
                self._t,
                current_target_label.state if current_target_label else None,
            )

        self._t = timestamp
        self._states[feature] = _StatePeriod(self._t, state)

    # def get_current_probability(self: Self) -> float | None:
    #     """Get the current probability based on current features and time block index."""
    #     self.logger.debug(
    #         "Getting current probability for %s with features %s at time block index %s",
    #         self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
    #         [
    #             f"{x}: {y}"
    #             for x, y in zip(self._all_features, self._current_feature, strict=True)
    #         ],
    #         self._current_time_block_index,
    #     )

    #     return self.data.get_probability(
    #         self._current_feature, self._current_time_block_index
    #     )

    # def get_current_state(self: Self) -> str | None:
    #     """Get the current state based on current features."""
    #     self.logger.debug(
    #         "Getting current state for %s with features %s: %s",
    #         self._config_entry.data.get(CONF_TARGET_ENTITY_ID),
    #         [
    #             f"{x}: {y}"
    #             for x, y in zip(self._all_features, self._current_feature, strict=True)
    #         ],
    #         self._forecasted_entity_current_state,
    #     )

    #     return self._forecasted_entity_current_state

    def get_state_to_store(self: Self, state: State | None) -> str | None:
        """Get the stored state for an entity."""
        if state is None:
            return None

        return (
            state.state
            if state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)
            else None
        )
