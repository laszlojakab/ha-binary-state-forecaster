# """Module of the Discrete State Forecaster Coordinator."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# from datetime import datetime
from typing import TYPE_CHECKING, Any, Hashable, Self, cast

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

from custom_components.discrete_state_forecaster.model.forecaster_engine_runtime_parameters import (
    ForecasterEngineRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_runtime_parameters import (
    DriftStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.duration_weighted_baseline_runtime_parameters import (
    DurationWeightedBaselineRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller import (
    AdaptationMode,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller_runtime_parameters import (
    AdaptationConfig,
    HyperParameterControllerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_runtime_parameters import (
    StatePersistenceTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_runtime_parameters import (
    OnlineErrorTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_runtime_parameters import (
    HierarchicalStateStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.prediction_result import (
    PredictionResult,
)
from custom_components.discrete_state_forecaster.model.structural_parameters import (
    StructuralParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.composite_indexer import (
    CompositeIndexer,
)

# from custom_components.discrete_state_forecaster.model.temporal.calendar_indexer import (
#     CalendarIndexer,
# )
# from custom_components.discrete_state_forecaster.model.temporal.season_indexer import (
#     SeasonIndexer,
# )
# from custom_components.discrete_state_forecaster.model.temporal.composite_indexer import (
#     CompositeIndexer,
# )
# from custom_components.discrete_state_forecaster.model.temporal.day_of_week_indexer import (
#     DayOfWeekIndexer,
# )
# from custom_components.discrete_state_forecaster.model.temporal.month_indexer import (
#     MonthIndexer,
# )
# from custom_components.discrete_state_forecaster.old_model.prediction import Prediction
# from custom_components.discrete_state_forecaster.old_model.time_aware_forecaster import (
#     TimeAwareForecaster,
# )
from custom_components.discrete_state_forecaster.model.temporal.time_of_day_indexer import (
    TimeOfDayIndexer,
)
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)

from .const import (
    #     CONF_ADAPTIVE_PERSISTENCE,
    #     CONF_CALENDAR_FEATURES,
    CONF_HALF_LIFE_HOURS,
    #     CONF_STATE_PERSISTENCE_FACTOR,
    CONF_TARGET_ENTITY_ID,
    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
    #     CONF_USE_DAY_OF_WEEK,
    #     CONF_USE_MONTH_OF_YEAR,
    #     CONF_USE_SEASON,
    #     DEFAULT_ADAPTIVE_PERSISTENCE,
    DEFAULT_HALF_LIFE_HOURS,
    #     DEFAULT_STATE_PERSISTENCE_FACTOR,
    DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
    #     DEFAULT_USE_DAY_OF_WEEK,
    #     DEFAULT_USE_MONTH_OF_YEAR,
    #     DEFAULT_USE_SEASON,
    STORING_TIME_PATTERN,
)

# from .old_model.state_tracker import StateTracker

if TYPE_CHECKING:
    from . import DiscreteStateForecasterConfigEntry


@dataclass
class DiscreteStateForecasterCoordinatorState:
    """State of the Discrete State Forecaster Coordinator."""

    prediction: PredictionResult
    current_state: str | None
    timestamp: datetime
    next_transition_timestamp: datetime | None = None
    adaption_mode: AdaptationMode | None = None


class DiscreteStateForecasterCoordinator(
    DataUpdateCoordinator[DiscreteStateForecasterCoordinatorState]
):

    _config_entry: "DiscreteStateForecasterConfigEntry"
    """Configuration entry for the coordinator."""

    _runtime_parameters: ForecasterEngineRuntimeParameters
    """Runtime parameters for the forecaster engine."""

    _current_state: Hashable | None
    """Current state of the target entity."""

    _current_state_last_reported_to_engine_at: datetime | None
    """Timestamp of the last time the current state was reported to the forecaster engine."""

    _current_time_bucket_start_at: datetime | None
    """Timestamp of the current time bucket start."""

    _next_time_bucket_start_at: datetime | None
    """Timestamp of the next time bucket start."""

    _composite_indexer: CompositeIndexer
    """Composite indexer that combines multiple time indexers based on configuration."""

    _forecaster: TimeAwareForecaster
    """The core forecaster engine that maintains state statistics and makes predictions."""

    _store: Store
    """Storage for persisting the forecaster state across restarts."""

    _unsubscribe_callbacks: list[CALLBACK_TYPE]
    """List of callbacks to unsubscribe from events when the coordinator is stopped."""

    _unsubscribe_time_indexer_change_callback: CALLBACK_TYPE | None
    """Callback to unsubscribe from time indexer change tracking."""

    _current_time_bucket_size: int
    """
    Currently used time bucket size for the time of day indexer,
    tracked for detecting configuration changes.
    """

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

        self._runtime_parameters = ForecasterEngineRuntimeParameters(
            hierarchical_state_stats=HierarchicalStateStatsRuntimeParameters(
                min_support_factor=0.00000001
            ),
            short_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
                error_half_life_factor=4.0
            ),
            long_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
                error_half_life_factor=40.0
            ),
            state_persistence_tracker=StatePersistenceTrackerRuntimeParameters(
                persistence_half_life_factor=5
            ),
            drift_monitor=DriftMonitorRuntimeParameters(
                slow_baseline=DurationWeightedBaselineRuntimeParameters(
                    half_life_factor=20.0,
                    prune_threshold=1e-6,
                    epsilon=1e-9,
                ),
                fast_baseline=DurationWeightedBaselineRuntimeParameters(
                    half_life_factor=1.5,
                    prune_threshold=1e-6,
                    epsilon=1e-9,
                ),
                drift_stats=DriftStatsRuntimeParameters(half_life_factor=30.0),
                tau_enter=0.1,
                tau_exit=0.05,
                adaptive_tau=False,
                n_enter=3,
                n_exit=5,
            ),
            hyper_parameter_controller=HyperParameterControllerRuntimeParameters(
                min_prune_interval_factor=5.0,
                base_half_life=config_entry.options.get(
                    CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
                )
                * 3600,
                base_persistence_strength=0.5,
                adaptation_config=AdaptationConfig(
                    adapt_half_life=False,
                    adapt_prune_interval=False,
                    adapt_persistence=False,
                ),
            ),
        )

        self._current_state = None
        self._current_state_last_reported_to_engine_at = None

        # Build indexers based on configuration
        self._composite_indexer = CompositeIndexer(
            self._build_indexers(time_bucket_minutes, config_entry.options)
        )

        self._next_time_bucket_start_at = None
        self._current_time_bucket_start_at = None

        #         # Get persistence settings from options
        #         state_persistence_factor = config_entry.options.get(
        #             CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
        #         )
        #         adaptive_persistence = config_entry.options.get(
        #             CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
        #         )

        self._forecaster = TimeAwareForecaster(
            StructuralParameters(
                indexer=self._composite_indexer,
            ),
            runtime_parameters=self._runtime_parameters,
        )

        # Initialize storage for model persistence
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_forecaster.json"
        )

        self._unsubscribe_callbacks = []
        self._unsubscribe_time_indexer_change_callback = None

        #         # Track current indexer configuration for detecting changes
        #         self._current_use_day_of_week = config_entry.options.get(
        #             CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        #         )
        #         self._current_use_month = config_entry.options.get(
        #             CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
        #         )
        #         self._current_use_season = config_entry.options.get(
        #             CONF_USE_SEASON, DEFAULT_USE_SEASON
        #         )
        #         self._current_calendar_features = config_entry.options.get(
        #             CONF_CALENDAR_FEATURES, []
        #         )
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

        now_local = dt_util.as_local(dt_util.now())

        # Initialize time bucket boundaries
        self._next_time_bucket_start_at = await self._composite_indexer.next_boundary(
            dt_util.as_local(dt_util.now())
        )
        self._current_time_bucket_start_at = (
            self._next_time_bucket_start_at
            - timedelta(minutes=self._current_time_bucket_size)
        )

        # Initiate target entity state
        await self._handle_target_entity_state_change(
            Event(
                event_type=EventType("state_changed"),
                data={
                    "new_state": self.hass.states.get(
                        self.config_entry.data[CONF_TARGET_ENTITY_ID]
                    )
                },
                time_fired_timestamp=now_local.timestamp(),
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
        await self._track_next_time_indexer_change(now_local)

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
                self._forecaster = TimeAwareForecaster.from_dict(
                    data,
                    StructuralParameters(indexer=self._composite_indexer),
                    runtime_parameters=self._runtime_parameters,
                )

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

    async def _async_update_data(
        self: Self,
    ) -> DiscreteStateForecasterCoordinatorState:
        """Fetches the latest prediction data."""
        self.logger.debug(
            "Updating data for %s.", self._config_entry.data[CONF_TARGET_ENTITY_ID]
        )

        # Get current time and make prediction
        now_local = dt_util.as_local(dt_util.now())

        # Get calculate the current state duration for persistence adjustment
        current_state_duration = (
            now_local.timestamp()
            - self._current_state_last_reported_to_engine_at.timestamp()
            if self._current_state_last_reported_to_engine_at
            else None
        )

        # Make prediction for current time
        prediction = await self._forecaster.predict_with_persistence(
            now_local,
            current_state=self._current_state,
            current_state_duration=current_state_duration,
        )

        # Calculate next transition time
        next_transition_timestamp = await self._forecaster.predict_next_transition(
            timestamp=now_local,
            current_state=self._current_state,
            current_state_duration=current_state_duration,
        )

        state = DiscreteStateForecasterCoordinatorState(
            prediction=prediction,
            current_state=self._current_state,
            timestamp=now_local,
            next_transition_timestamp=next_transition_timestamp,
            adaption_mode=self._forecaster._engine._hyper_parameter_controller.mode,
        )

        self.logger.debug(
            "Data updated for %s. New state: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            state,
        )

        return state

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

        time_fired_in_local = dt_util.as_local(event.time_fired)
        self.logger.debug(
            "Updating forecaster for %s with new state: %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            new_state_to_store,
            time_fired_in_local,
        )

        # We store the state and the timestamp when it was reported.
        # We use the timestamp to calculate the state duration for persistence adjustment
        # when making predictions.
        self._current_state = new_state_to_store
        self._current_state_last_reported_to_engine_at = time_fired_in_local
        await self._forecaster.update(new_state_to_store, time_fired_in_local)

        # TODO: remove
        await self._async_store_state()

        await self.async_refresh()

    async def _handle_time_key_change(self: Self, now: datetime) -> None:
        new_key_now_local = dt_util.as_local(now)

        self.logger.debug(
            "Time key change detected for %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            new_key_now_local,
        )

        if self._current_state is not None:
            self.logger.debug(
                "Updating forecaster for time key change with current state: %s at %s. "
                "Time key start: %s",
                self._current_state,
                new_key_now_local,
                self._current_time_bucket_start_at,
            )

            # _next_time_bucket_start_at is the boundary that just fired (e.g. 09:15:00).
            # Subtracting 1 microsecond places the timestamp firmly inside the finishing
            # bucket so the engine indexes it into the correct time key.
            just_before_boundary = self._next_time_bucket_start_at - timedelta(microseconds=1)
            await self._forecaster.update(self._current_state, just_before_boundary)
            self._current_state_last_reported_to_engine_at = self._next_time_bucket_start_at

            # TODO: remove
            await self._async_store_state()

        await self._track_next_time_indexer_change(new_key_now_local)

        await self.async_refresh()

    def _get_state_to_store(self: Self, state: State | None) -> str | None:
        """Get the stored state for an entity."""
        if state is None:
            return None

        return (
            state.state
            if state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)
            else None
        )

    def _build_indexers(
        self: Self,
        time_bucket_minutes: int,
        options: dict[str, Any],
    ) -> list:
        """Build time indexers based on configuration."""
        indexers = []

        #         # Calendar feature indexers (optional)
        #         for calendar_feature in options.get(CONF_CALENDAR_FEATURES, []):
        #             indexers.append(
        #                 CalendarIndexer(self.hass, calendar_feature)
        #             )  # noqa: PERF401

        #         # Day of week indexer (optional)
        #         use_day_of_week = options.get(CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK)
        #         if use_day_of_week:
        #             indexers.append(DayOfWeekIndexer())

        #         # Month of year indexer (optional)
        #         use_month = options.get(CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR)
        #         if use_month:
        #             indexers.append(MonthIndexer())

        #         use_season = options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        #         if use_season:
        #             indexers.append(SeasonIndexer())

        # Time of day indexer (always included)
        indexers.append(TimeOfDayIndexer(time_bucket_minutes * 60))

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
        #         new_use_day_of_week = config_entry.options.get(
        #             CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        #         )
        #         new_use_month = config_entry.options.get(
        #             CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
        #         )
        #         new_use_season = config_entry.options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        #         new_calendar_features = config_entry.options.get(CONF_CALENDAR_FEATURES, [])

        #         # Check if indexer configuration changed
        indexers_changed = (
            #             new_use_day_of_week != self._current_use_day_of_week
            #             or new_use_month != self._current_use_month
            #             or new_use_season != self._current_use_season
            #             or new_calendar_features != self._current_calendar_features
            # or
            new_time_bucket_size
            != self._current_time_bucket_size
        )

        half_life_hours = config_entry.options.get(
            CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
        )

        if indexers_changed:
            self.logger.info(
                "Indexer configuration changed - resetting model. "
                "Time bucket size: %s -> %s",
                # ", Day of week: %s -> %s, Month: %s -> %s, ",
                # "Season: %s -> %s, Calendar features: %s -> %s",
                self._current_time_bucket_size,
                new_time_bucket_size,
                # self._current_use_day_of_week,
                # new_use_day_of_week,
                # self._current_use_month,
                # new_use_month,
                # self._current_use_season,
                # new_use_season,
                # self._current_calendar_features,
                # new_calendar_features,
            )

            # Unsubscribe from old time indexer change tracking
            if self._unsubscribe_time_indexer_change_callback:
                self._unsubscribe_time_indexer_change_callback()
                self._unsubscribe_time_indexer_change_callback = None

            # Update tracked configuration
            #             self._current_use_day_of_week = new_use_day_of_week
            #             self._current_use_month = new_use_month
            #             self._current_calendar_features = new_calendar_features
            self._current_time_bucket_size = new_time_bucket_size

            # Reset bucket boundaries to avoid stale values from old indexer
            now_local = dt_util.as_local(dt_util.now())
            self._next_time_bucket_start_at = (
                await self._composite_indexer.next_boundary(now_local)
            )
            self._current_time_bucket_start_at = self._next_time_bucket_start_at - (
                timedelta(minutes=self._current_time_bucket_size)
            )

            # Rebuild indexers
            self._composite_indexer = CompositeIndexer(
                self._build_indexers(new_time_bucket_size, config_entry.options)
            )

            #             # Get persistence settings from options
            #             state_persistence_factor = config_entry.options.get(
            #                 CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
            #             )
            #             adaptive_persistence = config_entry.options.get(
            #                 CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
            #             )

            # Create new forecaster (resets the model)
            self._forecaster = TimeAwareForecaster(
                StructuralParameters(self._composite_indexer),
                self._runtime_parameters,
            )

            now_local = dt_util.as_local(dt_util.now())

            # If the entity has a known state, update the new forecaster with it
            # to avoid losing the current state information
            if self._current_state is not None:
                await self._forecaster.update(self._current_state, now_local)

            # Start tracking time indexer changes with new indexer configuration
            await self._track_next_time_indexer_change(now_local)

            # Store state immediately to persist the reset model and new configuration
            await self._async_store_state()

            self.logger.info("Model reset complete - learning from scratch")

        self._runtime_parameters.hyper_parameter_controller.base_half_life = (
            half_life_hours * 3600
        )

        # Trigger update to refresh all entities
        await self.async_refresh()

    async def _track_next_time_indexer_change(self: Self, now: datetime) -> None:
        # Previously I have used `now` parameter but sometimes it contained values which generates
        # next boundary for an elapsed time which caused immediate triggering of the callback.
        # Finally it has run into an infinite loop. To avoid this issue, I have changed to use
        # the current time which ensures that the next boundary is always in the future.
        now_local = dt_util.as_local(dt_util.now())

        self._current_time_bucket_start_at = self._next_time_bucket_start_at
        self._next_time_bucket_start_at = await self._composite_indexer.next_boundary(
            now_local
        )

        self.logger.debug(
            "Scheduling next time indexer change tracking for %s at %s.",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            self._next_time_bucket_start_at,
        )

        self.logger.debug(
            "Updated time indexer boundaries for %s. Current: %s, Next: %s",
            self._config_entry.data[CONF_TARGET_ENTITY_ID],
            self._current_time_bucket_start_at,
            self._next_time_bucket_start_at,
        )

        self._unsubscribe_time_indexer_change_callback = async_track_point_in_time(
            self.hass,
            self._handle_time_key_change,
            self._next_time_bucket_start_at,
        )
