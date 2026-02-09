"""The configuration flow for Discrete State Forecaster integration."""

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

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
    DOMAIN,
    LOGGER,
    SUPPORTED_BUCKET_SIZES,
    SUPPORTED_TARGET_DOMAINS,
)


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

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step when integration is added from the UI."""
        if user_input is not None:
            # Get the entity friendly name for the title
            target_entity = self.hass.states.get(user_input[CONF_TARGET_ENTITY_ID])
            entity_name = (
                target_entity.attributes.get("friendly_name")
                if target_entity
                else user_input[CONF_TARGET_ENTITY_ID]
            )
            title = f"{entity_name} Forecast"

            # Convert time bucket from string to int
            config_data = {
                CONF_TARGET_ENTITY_ID: user_input[CONF_TARGET_ENTITY_ID],
            }

            # Store initial options (indexers and prediction settings)
            options_data = {
                CONF_TIME_BUCKET_SIZE_IN_MINUTES: int(
                    user_input.get(
                        CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                        DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES,
                    )
                ),
                CONF_USE_DAY_OF_WEEK: user_input.get(CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK),
                CONF_USE_MONTH_OF_YEAR: user_input.get(
                    CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
                ),
                CONF_USE_SEASON: user_input.get(CONF_USE_SEASON, DEFAULT_USE_SEASON),
                CONF_STATE_PERSISTENCE_FACTOR: user_input.get(
                    CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
                ),
                CONF_ADAPTIVE_PERSISTENCE: user_input.get(
                    CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
                ),
                CONF_HALF_LIFE_HOURS: user_input.get(CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS),
                CONF_CALENDAR_FEATURES: user_input.get(CONF_CALENDAR_FEATURES, []),
            }

            LOGGER.info(
                "Creating Discrete State Forecaster for entity: %s. Config data: %s",
                config_data[CONF_TARGET_ENTITY_ID],
                config_data,
            )

            return self.async_create_entry(title=title, data=config_data, options=options_data)

        # Show form to select target entity and configuration
        data_schema = vol.Schema(
            {
                vol.Required(CONF_TARGET_ENTITY_ID): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SUPPORTED_TARGET_DOMAINS,
                    )
                ),
                vol.Required(
                    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                    default=str(DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        options=SUPPORTED_BUCKET_SIZES,
                        translation_key=CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                    )
                ),
                vol.Required(CONF_USE_DAY_OF_WEEK, default=DEFAULT_USE_DAY_OF_WEEK): bool,
                vol.Required(CONF_USE_MONTH_OF_YEAR, default=DEFAULT_USE_MONTH_OF_YEAR): bool,
                vol.Required(CONF_USE_SEASON, default=DEFAULT_USE_SEASON): bool,
                vol.Required(
                    CONF_STATE_PERSISTENCE_FACTOR,
                    default=DEFAULT_STATE_PERSISTENCE_FACTOR,
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Required(CONF_ADAPTIVE_PERSISTENCE, default=DEFAULT_ADAPTIVE_PERSISTENCE): bool,
                vol.Required(CONF_HALF_LIFE_HOURS, default=DEFAULT_HALF_LIFE_HOURS): vol.All(
                    vol.Coerce(float), vol.Range(min=0.0, max=8760.0)
                ),
                vol.Optional(CONF_CALENDAR_FEATURES): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        multiple=True,
                        domain=["calendar"],
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
        )


class DiscreteStateForecasterOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Discrete State Forecaster."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._new_options: dict[str, Any] = {}

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            # Get current indexer configuration
            current_use_day_of_week = self.config_entry.options.get(
                CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
            )
            current_use_month = self.config_entry.options.get(
                CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
            )
            current_use_season = self.config_entry.options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
            current_calendar_features = self.config_entry.options.get(CONF_CALENDAR_FEATURES, [])
            current_time_bucket_size_in_minutes = self.config_entry.options.get(
                CONF_TIME_BUCKET_SIZE_IN_MINUTES, DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES
            )

            # Check if indexer configuration changed
            indexers_changed = (
                user_input[CONF_USE_DAY_OF_WEEK] != current_use_day_of_week
                or user_input[CONF_USE_MONTH_OF_YEAR] != current_use_month
                or user_input[CONF_USE_SEASON] != current_use_season
                or user_input.get(CONF_CALENDAR_FEATURES, []) != current_calendar_features
                or user_input[CONF_TIME_BUCKET_SIZE_IN_MINUTES]
                != current_time_bucket_size_in_minutes
            )

            if indexers_changed:
                # Store new options and show confirmation warning
                self._new_options = user_input
                return await self.async_step_confirm_reset()

            # No indexer change - save directly
            return self.async_create_entry(title="", data=user_input)

        # Get current values with defaults
        current_use_day_of_week = self.config_entry.options.get(
            CONF_USE_DAY_OF_WEEK, DEFAULT_USE_DAY_OF_WEEK
        )
        current_use_month = self.config_entry.options.get(
            CONF_USE_MONTH_OF_YEAR, DEFAULT_USE_MONTH_OF_YEAR
        )
        current_use_season = self.config_entry.options.get(CONF_USE_SEASON, DEFAULT_USE_SEASON)
        current_persistence_factor = self.config_entry.options.get(
            CONF_STATE_PERSISTENCE_FACTOR, DEFAULT_STATE_PERSISTENCE_FACTOR
        )
        current_adaptive = self.config_entry.options.get(
            CONF_ADAPTIVE_PERSISTENCE, DEFAULT_ADAPTIVE_PERSISTENCE
        )
        current_half_life = self.config_entry.options.get(
            CONF_HALF_LIFE_HOURS, DEFAULT_HALF_LIFE_HOURS
        )
        current_calendar_features = self.config_entry.options.get(CONF_CALENDAR_FEATURES, [])
        current_time_bucket_size_in_minutes = self.config_entry.options.get(
            CONF_TIME_BUCKET_SIZE_IN_MINUTES, DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES
        )

        data_schema = vol.Schema(
            {
                vol.Required(
                    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                    default=str(current_time_bucket_size_in_minutes),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        options=SUPPORTED_BUCKET_SIZES,
                        translation_key=CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                    )
                ),
                vol.Required(
                    CONF_USE_DAY_OF_WEEK,
                    default=current_use_day_of_week,
                ): bool,
                vol.Required(
                    CONF_USE_MONTH_OF_YEAR,
                    default=current_use_month,
                ): bool,
                vol.Required(
                    CONF_USE_SEASON,
                    default=current_use_season,
                ): bool,
                vol.Required(
                    CONF_STATE_PERSISTENCE_FACTOR,
                    default=current_persistence_factor,
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Required(
                    CONF_ADAPTIVE_PERSISTENCE,
                    default=current_adaptive,
                ): bool,
                vol.Required(
                    CONF_HALF_LIFE_HOURS,
                    default=current_half_life,
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=8760.0)),
                vol.Optional(
                    CONF_CALENDAR_FEATURES, default=current_calendar_features
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        multiple=True,
                        domain=["calendar"],
                    ),
                ),
            }
        )

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
                # User confirmed - save options
                # The coordinator will detect the change and reset the model
                return self.async_create_entry(title="", data=self._new_options)

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
