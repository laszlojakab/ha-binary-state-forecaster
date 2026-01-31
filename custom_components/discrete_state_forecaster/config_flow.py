"""The configuration flow for Helios Easy Controls integration."""

from typing import Any

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CALENDAR_FEATURES,
    CONF_DECAY_SECONDS,
    CONF_FORECASTER_FEATURES,
    CONF_STABILITY,
    CONF_TARGET_ENTITY_ID,
    CONF_TIME_BUCKET_SIZE_IN_MINUTES,
    CONF_USE_DAY_OF_WEEK_FEATURE,
    CONF_USE_MONTH_OF_YEAR_FEATURE,
    DOMAIN,
    LOGGER,
    SUPPORTED_BUCKET_SIZES,
    SUPPORTED_STABILITY_OPTIONS,
    SUPPORTED_TARGET_DOMAINS,
)


@config_entries.HANDLERS.register(DOMAIN)
class DiscreteStateForecasterConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configuration flow handler for Discrete state forecaster integration."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handles the step when integration added from the UI."""
        data_schema = vol.Schema(
            {
                vol.Required(CONF_TARGET_ENTITY_ID): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SUPPORTED_TARGET_DOMAINS,
                    )
                ),
                vol.Required(CONF_TIME_BUCKET_SIZE_IN_MINUTES): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=SUPPORTED_BUCKET_SIZES,
                        translation_key=CONF_TIME_BUCKET_SIZE_IN_MINUTES,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                # vol.Required(CONF_STABILITY): selector.SelectSelector(
                #     selector.SelectSelectorConfig(
                #         options=SUPPORTED_STABILITY_OPTIONS,
                #         translation_key=CONF_STABILITY,
                #         mode=selector.SelectSelectorMode.DROPDOWN,
                #     )
                # ),
                vol.Required(CONF_USE_DAY_OF_WEEK_FEATURE, default=False): cv.boolean,
                vol.Required(CONF_USE_MONTH_OF_YEAR_FEATURE, default=False): cv.boolean,
                # vol.Optional(CONF_CALENDAR_FEATURES): selector.EntitySelector(
                #     selector.EntitySelectorConfig(
                #         multiple=True,
                #         domain=["calendar"],
                #     )
                # ),
                # vol.Optional(CONF_FORECASTER_FEATURES): selector.EntitySelector(
                #     selector.EntitySelectorConfig(
                #         multiple=True,
                #         integration=DOMAIN,
                #     )
                # ),
            }
        )

        if user_input is not None:
            # Get the binary sensor friendly name for the title
            binary_sensor_entity = self.hass.states.get(user_input[CONF_TARGET_ENTITY_ID])
            binary_sensor_name = (
                binary_sensor_entity.attributes.get("friendly_name")
                if binary_sensor_entity
                else user_input[CONF_TARGET_ENTITY_ID]
            )
            name = f"{binary_sensor_name}"

            def stability_to_decay_seconds(stability: str) -> int:
                if stability == "stable":
                    return 3600 * 24 * 14  # 2 weeks
                if stability == "semi_stable":
                    return 3600 * 24  # 24 hours
                if stability == "quick_changing":
                    return 3600  # 1 hour

                return 3600 * 24 * 7  # default to 1 week

            data = {
                CONF_NAME: name,
                CONF_TARGET_ENTITY_ID: user_input[CONF_TARGET_ENTITY_ID],
                # CONF_DECAY_SECONDS: stability_to_decay_seconds(
                #     user_input[CONF_STABILITY]
                # ),
                CONF_TIME_BUCKET_SIZE_IN_MINUTES: int(user_input[CONF_TIME_BUCKET_SIZE_IN_MINUTES]),
                CONF_USE_DAY_OF_WEEK_FEATURE: user_input.get(CONF_USE_DAY_OF_WEEK_FEATURE, False),
                CONF_USE_MONTH_OF_YEAR_FEATURE: user_input.get(
                    CONF_USE_MONTH_OF_YEAR_FEATURE, False
                ),
                # CONF_CALENDAR_FEATURES: user_input.get(CONF_CALENDAR_FEATURES, []),
                # CONF_FORECASTER_FEATURES: user_input.get(CONF_FORECASTER_FEATURES, []),
            }

            LOGGER.info("Creating Discrete State Forecaster with data: %s", data)

            return self.async_create_entry(title=name + " forecaster", data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
        )
