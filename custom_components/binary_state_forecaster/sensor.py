"""Sensor platform for Binary State Forecaster integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.const import CONF_NAME, PERCENTAGE
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import BinaryStateForecasterCoordinator

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from . import BinaryStateForecasterConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    config_entry: BinaryStateForecasterConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    coordinator: BinaryStateForecasterCoordinator = (
        config_entry.runtime_data.coordinator
    )

    async_add_entities(
        [
            BinaryStateForecasterProbabilitySensor(coordinator, config_entry),
        ]
    )


class BinaryStateForecasterProbabilitySensor(
    CoordinatorEntity[BinaryStateForecasterCoordinator], SensorEntity
):
    """Sensor for binary state forecaster probability."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE

    def __init__(
        self,
        coordinator: BinaryStateForecasterCoordinator,
        config_entry: BinaryStateForecasterConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._config_entry = config_entry
        self._attr_name = f"{config_entry.data.get(CONF_NAME)} Probability"
        self._attr_unique_id = f"{config_entry.entry_id}_probability"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose additional attributes for the probability sensor."""
        return {
            "forecasted_entity_state": self.coordinator.get_current_state(),
        }

    @property
    def native_value(self) -> float | None:
        """Return the current probability value."""
        probability = self.coordinator.get_current_probability()
        if probability is None:
            return None
        # Convert to percentage (0-100)
        return round(probability * 100, 2)
