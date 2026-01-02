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
    CONF_FADING,
    CONF_PERIOD,
    CONF_THRESHOLD,
    CONF_TIME_BLOCK_PERIOD,
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
                vol.Required(CONF_NAME): cv.string,
                vol.Required(CONF_BINARY_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["binary_sensor", "light", "switch", "input_boolean"]
                    )
                ),
                vol.Required(CONF_FADING): cv.small_float,
                vol.Required(CONF_THRESHOLD): cv.small_float,
            }
        )

        if user_input is not None:
            data = {
                CONF_NAME: user_input[CONF_NAME],
                CONF_BINARY_SENSOR: user_input[CONF_BINARY_SENSOR],
                CONF_FADING: user_input[CONF_FADING],
                CONF_THRESHOLD: user_input[CONF_THRESHOLD],
                CONF_PERIOD: 1440,
                CONF_TIME_BLOCK_PERIOD: 5,
            }

            return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
        )
