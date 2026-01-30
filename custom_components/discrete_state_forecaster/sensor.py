"""Sensor platform for Discrete State Forecaster."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import DiscreteStateForecasterCoordinator

if TYPE_CHECKING:
    from . import DiscreteStateForecasterConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DiscreteStateForecasterConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    coordinator = config_entry.runtime_data.coordinator

    async_add_entities(
        [DiscreteStateForecasterSensor(coordinator, config_entry)],
        True,
    )


class DiscreteStateForecasterSensor(
    CoordinatorEntity[DiscreteStateForecasterCoordinator], SensorEntity
):
    """Sensor that shows the most likely state from the distribution."""

    def __init__(
        self: Self,
        coordinator: DiscreteStateForecasterCoordinator,
        config_entry: DiscreteStateForecasterConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_has_entity_name = True
        self._attr_name = "Predicted State"
        self._attr_unique_id = f"{config_entry.entry_id}_predicted_state"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": f"Discrete State Forecaster {config_entry.title}",
            "manufacturer": "Discrete State Forecaster",
        }

    @property
    def native_value(self) -> str | None:
        """Return the state with maximum probability from distribution."""
        if self.coordinator.data is None:
            return None

        distribution, _ = self.coordinator.data.distribution

        if not distribution:
            return None

        # Find the key with maximum value
        max_state = max(distribution.items(), key=lambda x: x[1])
        return str(max_state[0])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        if self.coordinator.data is None:
            return {}

        attributes: dict[str, Any] = {}

        # Add confidence attributes
        confidence = self.coordinator.data.confidence
        attributes["max_probability"] = confidence.max_probability
        attributes["entropy_confidence"] = confidence.entropy_confidence
        attributes["support_time"] = confidence.support_time

        # Add used features
        attributes["used_features"] = dict(confidence.used_features)

        # Add drift level
        attributes["drift_level"] = self.coordinator.data.drift_level

        # Add full distribution for reference
        distribution, feature_key = self.coordinator.data.distribution
        attributes["distribution"] = {str(k): v for k, v in distribution.items()}
        attributes["feature_key"] = [
            {"feature": name, "value": str(label)} for name, label in feature_key
        ]

        return attributes

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
