"""The configuration flow for Discrete State Forecaster integration."""

from typing import Any, Self

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ADAPTIVE_PERSISTENCE,
    CONF_ADAPTIVE_PRUNE_INTERVAL,
    CONF_ADAPTIVE_TAU,
    CONF_ADVANCED_CONFIGURATION,
    CONF_BASE_STATE_INERTIA_STRENGTH,
    CONF_BASE_STATE_INERTIA_STRENGTH_MAX,
    CONF_BASE_STATE_INERTIA_STRENGTH_MIN,
    # CONF_ADAPTIVE_PERSISTENCE,
    # CONF_CALENDAR_FEATURES,
    CONF_ENABLE_ADAPTIVE_HALF_LIFE,
    CONF_FAST_BASELINE_HALF_LIFE_FACTOR,
    CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MAX,
    CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MIN,
    CONF_HALF_LIFE_HOURS,
    CONF_HALF_LIFE_HOURS_MAX,
    CONF_HALF_LIFE_HOURS_MIN,
    CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR,
    CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MAX,
    CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MIN,
    CONF_MIN_PRUNE_INTERVAL_FACTOR,
    CONF_MIN_PRUNE_INTERVAL_FACTOR_MAX,
    CONF_MIN_PRUNE_INTERVAL_FACTOR_MIN,
    CONF_PERSISTENCE_HALF_LIFE_FACTOR,
    CONF_PERSISTENCE_HALF_LIFE_FACTOR_MAX,
    CONF_PERSISTENCE_HALF_LIFE_FACTOR_MIN,
    CONF_PRESET,
    CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR,
    CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MAX,
    CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MIN,
    CONF_SLOW_BASELINE_HALF_LIFE_FACTOR,
    CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MAX,
    CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MIN,
    # CONF_STATE_PERSISTENCE_FACTOR,
    CONF_TARGET_ENTITY_ID,
    CONF_TAU_ENTER,
    CONF_TAU_ENTER_MAX,
    CONF_TAU_ENTER_MIN,
    CONF_TAU_EXIT,
    CONF_TAU_EXIT_MAX,
    CONF_TAU_EXIT_MIN,
    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
    CONF_USE_DAY_OF_WEEK,
    CONF_USE_MONTH_OF_YEAR,
    CONF_USE_SEASON,
    # DEFAULT_ADAPTIVE_PERSISTENCE,
    DEFAULT_HALF_LIFE_HOURS,
    DEFAULT_PRESET,
    # DEFAULT_STATE_PERSISTENCE_FACTOR,
    DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
    DEFAULT_USE_DAY_OF_WEEK,
    DEFAULT_USE_MONTH_OF_YEAR,
    DEFAULT_USE_SEASON,
    DOMAIN,
    LOGGER,
    PRESET_CONFIGURATIONS,
    PRESET_CUSTOM,
    PRESET_MODERATE,
    PRESETS,
    SUPPORTED_BUCKET_SIZES,
    SUPPORTED_TARGET_DOMAINS,
)

config_schema = {
    vol.Required(CONF_TARGET_ENTITY_ID): selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain=SUPPORTED_TARGET_DOMAINS,
        )
    ),
}


def get_basic_options_schema(options_user_input: dict[str, Any] | None = None):
    options_user_input = options_user_input or {}

    return {
        vol.Required(
            CONF_TIME_BUCKET_SIZE_IN_MINUTES,
            default=options_user_input.get(
                CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                str(DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES),
            ),
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                mode=selector.SelectSelectorMode.DROPDOWN,
                options=SUPPORTED_BUCKET_SIZES,
                translation_key=CONF_TIME_BUCKET_SIZE_IN_MINUTES,
            )
        ),
        vol.Required(
            CONF_USE_DAY_OF_WEEK,
            default=options_user_input.get(
                CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
            ),
        ): bool,
        vol.Required(
            CONF_USE_MONTH_OF_YEAR,
            default=options_user_input.get(
                CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
            ),
        ): bool,
        vol.Required(
            CONF_USE_SEASON,
            default=options_user_input.get(CONF_USE_SEASON, DEFAULT_USE_SEASON),
        ): bool,
        # vol.Optional(CONF_CALENDAR_FEATURES): selector.EntitySelector(
        #     selector.EntitySelectorConfig(
        #         multiple=True,
        #         domain=["calendar"],
        #     )
        # ),
        vol.Required(
            CONF_ADVANCED_CONFIGURATION,
            default=options_user_input.get(CONF_ADVANCED_CONFIGURATION, False),
        ): bool,
    }


def get_advanced_options_schema(options_user_input: dict[str, Any] | None = None):
    options_user_input = options_user_input or {}

    return {
        vol.Required(
            CONF_PRESET,
            default=options_user_input.get(CONF_PRESET, PRESET_MODERATE),
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                mode=selector.SelectSelectorMode.DROPDOWN,
                options=PRESETS,
                translation_key=CONF_PRESET,
            )
        ),
        vol.Required(
            CONF_ENABLE_ADAPTIVE_HALF_LIFE,
            default=options_user_input.get(CONF_ENABLE_ADAPTIVE_HALF_LIFE, False),
        ): bool,
        vol.Required(
            CONF_ADAPTIVE_PRUNE_INTERVAL,
            default=options_user_input.get(CONF_ADAPTIVE_PRUNE_INTERVAL, False),
        ): bool,
        vol.Required(
            CONF_ADAPTIVE_PERSISTENCE,
            default=options_user_input.get(CONF_ADAPTIVE_PERSISTENCE, False),
        ): bool,
    }


def get_custom_preset_schema(options_user_input: dict[str, Any] | None = None):
    options_user_input = options_user_input or {}

    current_preset = PRESET_CONFIGURATIONS.get(
        options_user_input.get(CONF_PRESET, PRESET_MODERATE), {}
    )

    return {
        vol.Required(
            CONF_HALF_LIFE_HOURS,
            default=options_user_input.get(
                CONF_HALF_LIFE_HOURS,
                current_preset.get(
                    CONF_HALF_LIFE_HOURS,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][CONF_HALF_LIFE_HOURS],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_HALF_LIFE_HOURS_MIN,
                max=CONF_HALF_LIFE_HOURS_MAX,
            ),
        ),
        vol.Required(
            CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR,
            default=options_user_input.get(
                CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR,
                current_preset.get(
                    CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MIN,
                max=CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MAX,
            ),
        ),
        vol.Required(
            CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR,
            default=options_user_input.get(
                CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR,
                current_preset.get(
                    CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MIN,
                max=CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MAX,
            ),
        ),
        vol.Required(
            CONF_BASE_STATE_INERTIA_STRENGTH,
            default=options_user_input.get(
                CONF_BASE_STATE_INERTIA_STRENGTH,
                current_preset.get(
                    CONF_BASE_STATE_INERTIA_STRENGTH,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_BASE_STATE_INERTIA_STRENGTH
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_BASE_STATE_INERTIA_STRENGTH_MIN,
                max=CONF_BASE_STATE_INERTIA_STRENGTH_MAX,
            ),
        ),
        vol.Required(
            CONF_PERSISTENCE_HALF_LIFE_FACTOR,
            default=options_user_input.get(
                CONF_PERSISTENCE_HALF_LIFE_FACTOR,
                current_preset.get(
                    CONF_PERSISTENCE_HALF_LIFE_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_PERSISTENCE_HALF_LIFE_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_PERSISTENCE_HALF_LIFE_FACTOR_MIN,
                max=CONF_PERSISTENCE_HALF_LIFE_FACTOR_MAX,
            ),
        ),
        vol.Required(
            CONF_FAST_BASELINE_HALF_LIFE_FACTOR,
            default=options_user_input.get(
                CONF_FAST_BASELINE_HALF_LIFE_FACTOR,
                current_preset.get(
                    CONF_FAST_BASELINE_HALF_LIFE_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_FAST_BASELINE_HALF_LIFE_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MIN,
                max=CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MAX,
            ),
        ),
        vol.Required(
            CONF_SLOW_BASELINE_HALF_LIFE_FACTOR,
            default=options_user_input.get(
                CONF_SLOW_BASELINE_HALF_LIFE_FACTOR,
                current_preset.get(
                    CONF_SLOW_BASELINE_HALF_LIFE_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_SLOW_BASELINE_HALF_LIFE_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MIN,
                max=CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MAX,
            ),
        ),
        vol.Required(
            CONF_TAU_ENTER,
            default=options_user_input.get(
                CONF_TAU_ENTER,
                current_preset.get(
                    CONF_TAU_ENTER,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][CONF_TAU_ENTER],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_TAU_ENTER_MIN,
                max=CONF_TAU_ENTER_MAX,
            ),
        ),
        vol.Required(
            CONF_TAU_EXIT,
            default=options_user_input.get(
                CONF_TAU_EXIT,
                current_preset.get(
                    CONF_TAU_EXIT,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][CONF_TAU_EXIT],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_TAU_EXIT_MIN,
                max=CONF_TAU_EXIT_MAX,
            ),
        ),
        vol.Required(
            CONF_ADAPTIVE_TAU,
            default=options_user_input.get(
                CONF_ADAPTIVE_TAU,
                current_preset.get(
                    CONF_ADAPTIVE_TAU,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][CONF_ADAPTIVE_TAU],
                ),
            ),
        ): bool,
        vol.Required(
            CONF_MIN_PRUNE_INTERVAL_FACTOR,
            default=options_user_input.get(
                CONF_MIN_PRUNE_INTERVAL_FACTOR,
                current_preset.get(
                    CONF_MIN_PRUNE_INTERVAL_FACTOR,
                    PRESET_CONFIGURATIONS[DEFAULT_PRESET][
                        CONF_MIN_PRUNE_INTERVAL_FACTOR
                    ],
                ),
            ),
        ): vol.All(
            vol.Coerce(float),
            vol.Range(
                min=CONF_MIN_PRUNE_INTERVAL_FACTOR_MIN,
                max=CONF_MIN_PRUNE_INTERVAL_FACTOR_MAX,
            ),
        ),
    }


@config_entries.HANDLERS.register(DOMAIN)
class DiscreteStateForecasterConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configuration flow handler for Discrete state forecaster integration."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return DiscreteStateForecasterOptionsFlow(config_entry)

    async def async_step_user(
        self: Self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step when integration is added from the UI."""
        if user_input is not None:
            user_input[CONF_PRESET] = PRESET_MODERATE

            if user_input.get(CONF_ADVANCED_CONFIGURATION):
                self._user_input = user_input
                return await self.async_step_advanced_configuration()

            return await self._create_entry(user_input)

        # Show form to select target entity and configuration
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    **config_schema,
                    **get_basic_options_schema(),
                }
            ),
        )

    async def async_step_advanced_configuration(
        self: Self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show advanced configuration options."""
        if user_input is not None:
            self._user_input.update(user_input)
            if user_input.get(CONF_PRESET) == PRESET_CUSTOM:
                # Custom preset - save all options from the form
                return await self.async_step_custom_preset()

            return await self._create_entry(self._user_input)

        # Show form to select advanced configuration options
        return self.async_show_form(
            step_id="advanced_configuration",
            data_schema=vol.Schema(get_advanced_options_schema()),
        )

    async def _create_entry(
        self: Self, user_input: dict[str, Any]
    ) -> config_entries.ConfigFlowResult:
        # Get the entity friendly name for the title
        target_entity = self.hass.states.get(user_input[CONF_TARGET_ENTITY_ID])
        entity_name = (
            target_entity.attributes.get("friendly_name")
            if target_entity
            else user_input[CONF_TARGET_ENTITY_ID]
        )
        title = f"{entity_name} Forecast"

        config_data = {
            CONF_TARGET_ENTITY_ID: user_input[CONF_TARGET_ENTITY_ID],
        }

        # Store initial options (indexers and prediction settings)
        options_data = {
            CONF_PRESET: user_input.get(CONF_PRESET, PRESET_MODERATE),
            CONF_TIME_BUCKET_SIZE_IN_MINUTES: int(
                user_input.get(
                    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                    DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
                )
            ),
            CONF_USE_DAY_OF_WEEK: user_input.get(
                CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
            ),
            CONF_USE_MONTH_OF_YEAR: user_input.get(
                CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
            ),
            CONF_USE_SEASON: user_input.get(CONF_USE_SEASON, DEFAULT_USE_SEASON),
            CONF_HALF_LIFE_HOURS: user_input.get(
                CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
            ),
            #     CONF_CALENDAR_FEATURES: user_input.get(CONF_CALENDAR_FEATURES, []),
        }

        LOGGER.info(
            "Creating Discrete State Forecaster for entity: %s. Config data: %s",
            config_data[CONF_TARGET_ENTITY_ID],
            config_data,
        )

        return self.async_create_entry(
            title=title, data=config_data, options=options_data
        )

    async def async_step_custom_preset(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show form to configure custom preset parameters."""
        if user_input is not None:
            self._user_input.update(user_input)

            return await self._create_entry(self._user_input)

        # Show form to configure custom preset parameters
        return self.async_show_form(
            step_id="custom_preset",
            data_schema=vol.Schema(get_custom_preset_schema(self._user_input)),
        )


class DiscreteStateForecasterOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Discrete State Forecaster."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._user_input: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            # Get current indexer configuration
            current_use_day_of_week = self.config_entry.options.get(
                CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
            )
            current_use_month = self.config_entry.options.get(
                CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
            )
            current_use_season = self.config_entry.options.get(
                CONF_USE_SEASON, DEFAULT_USE_SEASON
            )
            # current_calendar_features = self.config_entry.options.get(
            #     CONF_CALENDAR_FEATURES, []
            # )
            current_time_bucket_size_in_minutes = self.config_entry.options.get(
                CONF_TIME_BUCKET_SIZE_IN_MINUTES, DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES
            )

            # Check if indexer configuration changed
            indexers_changed = (
                str(user_input[CONF_TIME_BUCKET_SIZE_IN_MINUTES])
                != str(current_time_bucket_size_in_minutes)
                or user_input[CONF_USE_DAY_OF_WEEK] != current_use_day_of_week
                or user_input[CONF_USE_MONTH_OF_YEAR] != current_use_month
                or user_input[CONF_USE_SEASON] != current_use_season
                # or user_input.get(CONF_CALENDAR_FEATURES, []) != current_calendar_features
            )

            self._user_input = user_input
            if indexers_changed:
                # Store new options and show confirmation warning
                return await self.async_step_confirm_reset()

            if user_input.get(CONF_ADVANCED_CONFIGURATION):
                return await self.async_step_advanced_configuration()

            # No indexer change - save directly
            return self.async_create_entry(title="", data=self._user_input)

        # Get current values with defaults
        current_use_day_of_week = self.config_entry.options.get(
            CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        )
        current_use_month = self.config_entry.options.get(
            CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
        )
        current_use_season = self.config_entry.options.get(
            CONF_USE_SEASON, DEFAULT_USE_SEASON
        )
        # current_half_life = self.config_entry.options.get(
        #     CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
        # )
        # current_calendar_features = self.config_entry.options.get(
        #     CONF_CALENDAR_FEATURES, []
        # )
        current_time_bucket_size_in_minutes = self.config_entry.options.get(
            CONF_TIME_BUCKET_SIZE_IN_MINUTES, DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES
        )

        data_schema = vol.Schema(get_basic_options_schema(self.config_entry.options))

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
        )

    async def async_step_confirm_reset(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm model reset when indexers change."""
        if user_input is not None:
            if user_input.get("confirm"):
                # User confirmed - save options, the coordinator will detect the change
                # and reset the model
                if user_input.get(CONF_ADVANCED_CONFIGURATION):
                    return await self.async_step_advanced_configuration()

                return self.async_create_entry(title="", data=self._user_input)

            # User cancelled - go back to options
            return await self.async_step_init()

        return self.async_show_form(
            step_id="confirm_reset",
            data_schema=vol.Schema(
                {
                    vol.Required("confirm", default=False): bool,
                }
            ),
        )

    async def async_step_advanced_configuration(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show advanced configuration options."""
        if user_input is not None:
            self._user_input.update(user_input)
            if user_input.get(CONF_PRESET) == PRESET_CUSTOM:
                # Custom preset - save all options from the form
                return await self.async_step_custom_preset()

            return self.async_create_entry(title="", data=self._user_input)

        """Show advanced configuration options."""
        return self.async_show_form(
            step_id="advanced_configuration",
            data_schema=vol.Schema(
                get_advanced_options_schema(self.config_entry.options)
            ),
        )

    async def async_step_custom_preset(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show form to configure custom preset parameters."""
        if user_input is not None:
            self._user_input.update(user_input)
            return self.async_create_entry(title="", data=self._user_input)

        # Show form to configure custom preset parameters
        return self.async_show_form(
            step_id="custom_preset",
            data_schema=vol.Schema(get_custom_preset_schema(self.config_entry.options)),
        )
