"""The configuration flow for Helios Easy Controls integration."""

from typing import Any

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_BINARY_SENSOR,
    CONF_CALENDAR_FEATURES,
    CONF_FADING,
    CONF_FORECASTER_FEATURES,
    CONF_PERIOD,
    CONF_THRESHOLD,
    CONF_TIME_BLOCK_PERIOD,
    CONF_USE_DAY_OF_WEEK_FEATURE,
    DOMAIN,
)


@config_entries.HANDLERS.register(DOMAIN)
class BinaryStateForecasterConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configuration flow handler for Binary state forecaster integration."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handles the step when integration added from the UI."""
        data_schema = vol.Schema(
            {
                vol.Required(CONF_BINARY_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["binary_sensor", "light", "switch", "input_boolean"]
                    )
                ),
                vol.Required(CONF_FADING): cv.small_float,
                vol.Required(CONF_THRESHOLD): cv.small_float,
                vol.Required(CONF_USE_DAY_OF_WEEK_FEATURE, default=False): cv.boolean,
                vol.Optional(CONF_CALENDAR_FEATURES): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        multiple=True,
                        domain=["calendar"],
                    )
                ),
                vol.Optional(CONF_FORECASTER_FEATURES): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        multiple=True,
                        integration=DOMAIN,
                    )
                ),
            }
        )

        if user_input is not None:
            # Get the binary sensor friendly name for the title
            binary_sensor_entity = self.hass.states.get(user_input[CONF_BINARY_SENSOR])
            binary_sensor_name = (
                binary_sensor_entity.attributes.get("friendly_name")
                if binary_sensor_entity
                else user_input[CONF_BINARY_SENSOR]
            )
            name = f"{binary_sensor_name}"

            data = {
                CONF_NAME: name,
                CONF_BINARY_SENSOR: user_input[CONF_BINARY_SENSOR],
                CONF_FADING: user_input[CONF_FADING],
                CONF_THRESHOLD: user_input[CONF_THRESHOLD],
                CONF_PERIOD: 1440,
                CONF_TIME_BLOCK_PERIOD: 5,
                CONF_USE_DAY_OF_WEEK_FEATURE: user_input.get(CONF_USE_DAY_OF_WEEK_FEATURE, False),
                CONF_CALENDAR_FEATURES: user_input.get(CONF_CALENDAR_FEATURES, []),
                CONF_FORECASTER_FEATURES: user_input.get(CONF_FORECASTER_FEATURES, []),
            }

            return self.async_create_entry(title=name + " Forecaster", data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
        )
