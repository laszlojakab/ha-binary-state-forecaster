"""Module of the BinaryStateForecaster class."""

import json
from datetime import timedelta
from typing import Any, Final, Self

from .const import TIME_BLOCK_PERIOD_IN_MINUTES

Probability = float | None

class DiscreteConditionalProbabilityModel:
    ...


class BinaryStateForecasterMatrix:
    def __init__(
        self: Self,
        state: dict[tuple[Any, ...], list[float | None]] | None = None,
    ) -> None:
        self._state: dict[tuple[Any, ...], list[float | None]] = state or {}

    def get_probability(
        self: Self, features: tuple[Any, ...], time_block_index: int
    ) -> float | None:
        """Get the probability for the given features and time block index."""
        probabilities = self._state.get(features)
        if probabilities is None:
            return None

        value_at_time_block_index = probabilities[time_block_index]

        if value_at_time_block_index is None:
            # Find closest non-None values before and after
            left_value = None
            left_index = None
            for i in range(time_block_index - 1, -1, -1):
                if probabilities[i] is not None:
                    left_value = probabilities[i]
                    left_index = i
                    break

            right_value = None
            right_index = None
            for i in range(time_block_index + 1, len(probabilities)):
                if probabilities[i] is not None:
                    right_value = probabilities[i]
                    right_index = i
                    break

            # Interpolate if both left and right values exist
            if left_value is not None and right_value is not None:
                # Linear interpolation
                total_distance = right_index - left_index
                distance_from_left = time_block_index - left_index
                weight = distance_from_left / total_distance
                value_at_time_block_index = (
                    left_value + (right_value - left_value) * weight
                )

        return value_at_time_block_index

    def update_probability(
        self: Self, features: tuple[Any, ...], timestamp_index: int, probability: float
    ) -> None:
        """Update the probability for the given features and timestamp index."""
        if features not in self._state:
            self._state[features] = [None] * int(
                timedelta(hours=24).total_seconds()
                / (TIME_BLOCK_PERIOD_IN_MINUTES * 60)
            )

        self._state[features][timestamp_index] = probability

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any],
    ) -> Self:
        """Create an instance from a dictionary."""
        state_dict: dict = data.get("state", {})
        on_state: str = data.get("on_state", "on")
        off_state: str = data.get("off_state", "off")
        # Convert JSON string keys back to tuples
        restored_state: dict[tuple[Any, ...], list[float]] = {
            tuple(json.loads(key)): value for key, value in state_dict.items()
        }
        return cls(state=restored_state, on_state=on_state, off_state=off_state)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        # Convert tuple keys to JSON strings for serialization
        serializable_state = {
            json.dumps(key): value for key, value in self._state.items()
        }
        return {
            "state": serializable_state,
            "on_state": self._on_state,
            "off_state": self._off_state,
        }


class BinaryStateForecasterState:
    """The internal state of binary state forecaster coordinator."""

    def __init__(
        self: Self,
        on_state: str = "on",
        off_state: str = "off",
        state: dict[tuple[Any, ...], list[float | None]] | None = None,
    ) -> None:
        """Initialize the coordinator state."""
        self._state: dict[tuple[Any, ...], list[float | None]] = state or {}
        self._on_state: Final = on_state
        self._off_state: Final = off_state

    def get_probability(
        self: Self, features: tuple[Any, ...], time_block_index: int
    ) -> float | None:
        """Get the probability for the given features and time block index."""
        probabilities = self._state.get(features)
        if probabilities is None:
            return None

        value_at_time_block_index = probabilities[time_block_index]

        if value_at_time_block_index is None:
            # Find closest non-None values before and after
            left_value = None
            left_index = None
            for i in range(time_block_index - 1, -1, -1):
                if probabilities[i] is not None:
                    left_value = probabilities[i]
                    left_index = i
                    break

            right_value = None
            right_index = None
            for i in range(time_block_index + 1, len(probabilities)):
                if probabilities[i] is not None:
                    right_value = probabilities[i]
                    right_index = i
                    break

            # Interpolate if both left and right values exist
            if left_value is not None and right_value is not None:
                # Linear interpolation
                total_distance = right_index - left_index
                distance_from_left = time_block_index - left_index
                weight = distance_from_left / total_distance
                value_at_time_block_index = (
                    left_value + (right_value - left_value) * weight
                )

        return value_at_time_block_index

    def update_probability(
        self: Self, features: tuple[Any, ...], timestamp_index: int, probability: float
    ) -> None:
        """Update the probability for the given features and timestamp index."""
        if features not in self._state:
            self._state[features] = [None] * int(
                timedelta(hours=24).total_seconds()
                / (TIME_BLOCK_PERIOD_IN_MINUTES * 60)
            )

        self._state[features][timestamp_index] = probability

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any],
    ) -> Self:
        """Create an instance from a dictionary."""
        state_dict: dict = data.get("state", {})
        on_state: str = data.get("on_state", "on")
        off_state: str = data.get("off_state", "off")
        # Convert JSON string keys back to tuples
        restored_state: dict[tuple[Any, ...], list[float]] = {
            tuple(json.loads(key)): value for key, value in state_dict.items()
        }
        return cls(state=restored_state, on_state=on_state, off_state=off_state)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        # Convert tuple keys to JSON strings for serialization
        serializable_state = {
            json.dumps(key): value for key, value in self._state.items()
        }
        return {
            "state": serializable_state,
            "on_state": self._on_state,
            "off_state": self._off_state,
        }


class BinaryStateForecaster:
    """The binary state forecaster."""

    def __init__(
        self: Self,
        on_state: str = "on",
        off_state: str = "off",
        state: BinaryStateForecasterState | None = None,
    ) -> None:
        """Initialize the BinaryStateForecaster."""
        self._state = state or BinaryStateForecasterState(on_state, off_state)

    @property
    def state(self: Self) -> BinaryStateForecasterState:
        """Get the internal state of the forecaster."""
        return self._state

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        """Create a BinaryStateForecaster from a serialized dict."""
        internal_state = BinaryStateForecasterState.from_dict(data)
        return cls(
            on_state=data.get("on_state", "on"),
            off_state=data.get("off_state", "off"),
            state=internal_state,
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """Convert the forecaster to a serialized dict."""
        return self._state.to_dict()
