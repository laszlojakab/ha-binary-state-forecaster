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
        """Return the predicted state."""
        if self.coordinator.data is None:
            return "unknown"

        prediction = self.coordinator.data.prediction
        if prediction.state is None:
            return "unknown"  # No data yet

        return str(prediction.state)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        if self.coordinator.data is None:
            return {}

        prediction = self.coordinator.data.prediction
        attributes: dict[str, Any] = {}

        # Add predicted state and probability
        if prediction.state is not None:
            attributes["predicted_state"] = str(prediction.state)
            attributes["probability"] = prediction.distribution.get(
                prediction.state, 0.0
            )

        # Add current actual state for comparison
        if self.coordinator.data.current_state:
            attributes["current_state"] = self.coordinator.data.current_state

        # Add confidence metrics
        confidence = prediction.confidence
        attributes["key"] = str(prediction.key) if prediction.key else None
        attributes["confidence"] = {
            "max_probability": round(confidence.max_probability, 3),
            "entropy_confidence": round(confidence.entropy_confidence, 3),
            "support_time": round(confidence.support_time),
            "depth": confidence.depth,
        }

        # Add full probability distribution
        attributes["distribution"] = {
            str(state): round(prob, 3)
            for state, prob in prediction.distribution.items()
        }

        # Add timestamp of prediction
        attributes["timestamp"] = self.coordinator.data.timestamp.isoformat()

        # Add next transition time if available
        attributes["next_transition_time"] = (
            self.coordinator.data.next_transition_timestamp.isoformat()
            if self.coordinator.data.next_transition_timestamp
            else None
        )

        # Add learned persistence factors if available
        try:
            learned_persistence = self.coordinator._forecaster.get_learned_persistence()
            if learned_persistence:
                attributes["learned_persistence"] = {
                    str(state): factor for state, factor in learned_persistence.items()
                }
        except Exception:  # noqa: S110
            pass  # Not critical if this fails

        return attributes

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
