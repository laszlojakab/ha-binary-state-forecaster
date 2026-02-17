"""Discrete state forecaster integration module."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.core_config import Config

from .const import LOGGER
from .coordinator import DiscreteStateForecasterCoordinator


@dataclass
class DiscreteStateForecasterRuntimeData:
    """The runtime data for the discrete state forecaster integration."""

    coordinator: DiscreteStateForecasterCoordinator


type DiscreteStateForecasterConfigEntry = ConfigEntry[
    DiscreteStateForecasterRuntimeData
]
"""The config entry for the discrete state forecaster integration."""


async def async_setup(hass: HomeAssistant, config: Config) -> bool:
    """
    Set up the discrete state forecaster integration.

    Args:
      hass:
        The Home Assistant instance.
      config:
        The configuration.

    Returns:
        The value indicates whether the setup succeeded.
    """
    return True


async def async_setup_entry(
    hass: HomeAssistant, config_entry: "DiscreteStateForecasterConfigEntry"
) -> bool:
    """
    Initialize the forecaster sensor based on the config entry.

    Args:
      hass:
        The Home Assistant instance.
      config_entry:
        The config entry which contains information gathered by the config flow.

    Returns:
        The value indicates whether the setup succeeded.
    """
    coordinator = DiscreteStateForecasterCoordinator(hass, config_entry, LOGGER)
    config_entry.runtime_data = DiscreteStateForecasterRuntimeData(coordinator)

    await coordinator.async_start()

    # # Forward setup to sensor platform
    # await hass.config_entries.async_forward_entry_setups(config_entry, ["sensor"])

    return True


async def async_unload_entry(
    hass: HomeAssistant, config_entry: "DiscreteStateForecasterConfigEntry"
) -> bool:
    """
    Executed when a config entry unloaded by Home Assistant.

    Args:
      hass:
        The Home Assistant instance.
      config_entry:
        The config entry being unloaded.

    Returns:
      The value indicates whether the unloading succeeded.
    """
    await config_entry.runtime_data.coordinator.async_stop()

    # # Unload sensor platform
    # await hass.config_entries.async_unload_platforms(config_entry, ["sensor"])

    return True
