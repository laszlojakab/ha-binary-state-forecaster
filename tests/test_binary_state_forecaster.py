"""Tests for the Binary State Forecaster."""

import sys
from pathlib import Path
import json

# Ensure project root is on sys.path so ``custom_components`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from custom_components.binary_state_forecaster.binary_state_forecaster import (  # noqa: E402
    BinaryStateForecaster,
    BinaryStateForecasterState,
)


def test_forecaster_initialization() -> None:
    """Tests for the Binary State Forecaster initialization."""
    forecaster = BinaryStateForecaster()

    assert forecaster is not None


def test_forecaster_to_dict_serializes_state_and_on_off() -> None:
    """BinaryStateForecaster.to_dict should serialize internal state correctly."""
    features = ("sensor.living_room", "weekday", 10)
    probabilities = [0.1, 0.5, None, 0.9]

    internal_state = BinaryStateForecasterState(
        on_state="custom_on",
        off_state="custom_off",
        state={features: probabilities},
    )

    forecaster = BinaryStateForecaster(
        on_state="custom_on",
        off_state="custom_off",
        state=internal_state,
    )

    data = forecaster.to_dict()

    assert data["on_state"] == "custom_on"
    assert data["off_state"] == "custom_off"
    assert "state" in data

    state_dict = data["state"]
    expected_key = json.dumps(features)
    assert expected_key in state_dict
    assert state_dict[expected_key] == probabilities


def test_forecaster_from_dict_roundtrip() -> None:
    """BinaryStateForecaster.from_dict should restore a forecaster from dict data."""
    features = ("sensor.kitchen", "weekend", 5)
    probabilities = [0.0, 0.25, 0.75]

    state_dict = {json.dumps(features): probabilities}
    original_data = {
        "state": state_dict,
        "on_state": "on",
        "off_state": "off",
    }

    forecaster = BinaryStateForecaster.from_dict(original_data)
    restored_data = forecaster.to_dict()

    assert restored_data["on_state"] == original_data["on_state"]
    assert restored_data["off_state"] == original_data["off_state"]
    assert restored_data["state"] == original_data["state"]


def test_state_property_returns_internal_state_instance() -> None:
    """The state property should expose the same internal state instance."""
    internal_state = BinaryStateForecasterState()
    forecaster = BinaryStateForecaster(state=internal_state)

    assert forecaster.state is internal_state


def test_state_property_default_creates_state_instance() -> None:
    """The state property on default forecaster should be a BinaryStateForecasterState."""
    forecaster = BinaryStateForecaster()

    assert isinstance(forecaster.state, BinaryStateForecasterState)
