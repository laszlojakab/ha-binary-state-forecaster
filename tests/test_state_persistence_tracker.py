"""
Unit tests for StatePersistenceTracker.

Comprehensive tests for state persistence tracking and hazard-style
persistence boost computation.
"""

import math
from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker import (
    StatePersistenceTracker,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_hyper_parameters import (  # noqa: E501
    StatePersistenceTrackerHyperParameters,
)


def create_test_hp(
    half_life: float = 50.0,
    persistence_half_life_factor: float = 1.0,
) -> StatePersistenceTrackerHyperParameters:
    """Create test hyper-parameters."""
    base_hp = HyperParameters(
        half_life=half_life,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    return StatePersistenceTrackerHyperParameters(
        hyper_parameters=base_hp,
        persistence_half_life_factor=persistence_half_life_factor,
    )


class TestStatePersistenceTrackerInitialization:
    """Tests for StatePersistenceTracker initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating tracker with default configuration."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        assert tracker.current_state is None
        assert tracker.current_duration(100.0) == 0.0

    def test_instance_isolation(self: Self) -> None:
        """Test that multiple instances don't share state."""
        hp = create_test_hp()
        tracker1 = StatePersistenceTracker(hp)
        tracker2 = StatePersistenceTracker(hp)

        tracker1.update("on", 100.0)
        tracker2.update("off", 100.0)

        assert tracker1.current_state == "on"
        assert tracker2.current_state == "off"
        assert tracker1.current_state != tracker2.current_state

    def test_mean_duration_isolation(self: Self) -> None:
        """Test that mean duration dicts are not shared between instances."""
        hp = create_test_hp()
        tracker1 = StatePersistenceTracker(hp)
        tracker2 = StatePersistenceTracker(hp)

        tracker1.update("on", 100.0)
        tracker1.update("off", 200.0)

        # tracker2 should have no mean durations
        assert tracker2.expected_duration("on", 999.0) == 999.0
        assert tracker1.expected_duration("on", 999.0) != 999.0


class TestStatePersistenceTrackerUpdate:
    """Tests for StatePersistenceTracker.update method."""

    def test_first_update(self: Self) -> None:
        """Test first update sets current state."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        assert tracker.current_state == "on"
        assert tracker.current_duration(100.0) == 0.0

    def test_same_state_update(self: Self) -> None:
        """Test that updating with same state doesn't record duration."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        tracker.update("on", 150.0)
        tracker.update("on", 200.0)

        assert tracker.current_state == "on"
        # No state change means no duration recorded
        assert tracker.expected_duration("on", 999.0) == 999.0

    def test_state_change_records_duration(self: Self) -> None:
        """Test that state change records duration of previous state."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        tracker.update("off", 200.0)  # "on" lasted 100 time units

        assert tracker.current_state == "off"
        # First observation sets mean to the observed duration
        assert abs(tracker.expected_duration("on") - 100.0) < 1e-6

    def test_multiple_state_changes(self: Self) -> None:
        """Test multiple state changes update mean durations."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        tracker.update("off", 200.0)  # "on" lasted 100
        tracker.update("on", 250.0)  # "off" lasted 50
        tracker.update("off", 400.0)  # "on" lasted 150

        # Check both states have recorded durations
        assert tracker.expected_duration("on", 999.0) != 999.0
        assert tracker.expected_duration("off", 999.0) != 999.0

    def test_exponential_weighting(self: Self) -> None:
        """Test that durations are exponentially weighted."""
        hp = create_test_hp(half_life=50.0)
        tracker = StatePersistenceTracker(hp)

        # First observation
        tracker.update("on", 0.0)
        tracker.update("off", 100.0)  # "on" duration = 100, dt = 100
        mean1 = tracker.expected_duration("on")
        assert abs(mean1 - 100.0) < 1e-6

        # Second observation after time gap
        tracker.update("on", 200.0)  # dt = 100 (from last update at 100)
        tracker.update("off", 250.0)  # "on" duration = 50, dt = 50
        mean2 = tracker.expected_duration("on")

        # dt = 50, half_life = 50, so decay = exp(-ln(2) * 50/50) = 0.5
        # New mean = 0.5 * 100 + 0.5 * 50 = 75
        assert abs(mean2 - 75.0) < 1.0


class TestStatePersistenceTrackerCurrentState:
    """Tests for current_state property."""

    def test_no_state_initially(self: Self) -> None:
        """Test that current_state is None initially."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        assert tracker.current_state is None

    def test_current_state_after_update(self: Self) -> None:
        """Test current_state reflects latest update."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        assert tracker.current_state == "on"

        tracker.update("off", 200.0)
        assert tracker.current_state == "off"


class TestStatePersistenceTrackerCurrentDuration:
    """Tests for current_duration method."""

    def test_zero_when_no_state(self: Self) -> None:
        """Test current_duration returns 0.0 when no state active."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        assert tracker.current_duration(100.0) == 0.0

    def test_duration_calculation(self: Self) -> None:
        """Test current_duration calculates correctly."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        assert tracker.current_duration(100.0) == 0.0
        assert tracker.current_duration(150.0) == 50.0
        assert tracker.current_duration(200.0) == 100.0

    def test_duration_resets_on_state_change(self: Self) -> None:
        """Test duration resets when state changes."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        assert tracker.current_duration(200.0) == 100.0

        tracker.update("off", 200.0)
        assert tracker.current_duration(200.0) == 0.0
        assert tracker.current_duration(250.0) == 50.0

    def test_duration_never_negative(self: Self) -> None:
        """Test current_duration handles backwards time."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        # Even with backwards time, duration should be >= 0
        assert tracker.current_duration(50.0) == 0.0


class TestStatePersistenceTrackerExpectedDuration:
    """Tests for expected_duration method."""

    def test_default_for_unseen_state(self: Self) -> None:
        """Test expected_duration returns default for unseen states."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        assert tracker.expected_duration("unknown") == 60.0
        assert tracker.expected_duration("unknown", 123.0) == 123.0

    def test_expected_duration_after_observation(self: Self) -> None:
        """Test expected_duration returns mean after observations."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 0.0)
        tracker.update("off", 100.0)

        # First observation sets mean to observed value
        assert abs(tracker.expected_duration("on") - 100.0) < 1e-6

    def test_expected_duration_multiple_states(self: Self) -> None:
        """Test expected_duration tracks different states independently."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 0.0)
        tracker.update("off", 100.0)  # "on" lasted 100
        tracker.update("on", 120.0)  # "off" lasted 20
        tracker.update("off", 170.0)  # "on" lasted 50

        # Different states should have different expected durations
        on_duration = tracker.expected_duration("on")
        off_duration = tracker.expected_duration("off")
        assert on_duration != off_duration


class TestStatePersistenceTrackerPersistenceBoost:
    """Tests for persistence_boost method."""

    def test_zero_for_different_state(self: Self) -> None:
        """Test persistence_boost returns 0 for non-current state."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        assert tracker.persistence_boost("off", 100.0) == 0.0

    def test_max_at_start(self: Self) -> None:
        """Test persistence_boost is highest when state just started."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        # At t=100 (duration=0), boost should be exp(-0) = 1.0
        assert abs(tracker.persistence_boost("on", 100.0) - 1.0) < 1e-9

    def test_decreases_with_time(self: Self) -> None:
        """Test persistence_boost decreases as state duration increases."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 0.0)
        tracker.update("off", 100.0)  # Establish expected duration of 100
        tracker.update("on", 200.0)

        boost_0 = tracker.persistence_boost("on", 200.0)  # duration = 0
        boost_50 = tracker.persistence_boost("on", 250.0)  # duration = 50
        boost_100 = tracker.persistence_boost("on", 300.0)  # duration = 100
        boost_200 = tracker.persistence_boost("on", 400.0)  # duration = 200

        # Boost should decay monotonically
        assert boost_0 > boost_50 > boost_100 > boost_200

    def test_hazard_decay_formula(self: Self) -> None:
        """Test persistence_boost uses correct hazard-style decay."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 0.0)
        tracker.update("off", 100.0)  # "on" expected duration = 100
        tracker.update("on", 200.0)

        # At duration = expected, boost should be exp(-1) ≈ 0.368
        boost = tracker.persistence_boost("on", 300.0)
        expected = math.exp(-1.0)
        assert abs(boost - expected) < 1e-6

        # At duration = 2 * expected, boost should be exp(-2) ≈ 0.135
        boost = tracker.persistence_boost("on", 400.0)
        expected = math.exp(-2.0)
        assert abs(boost - expected) < 1e-6

    def test_boost_with_custom_default(self: Self) -> None:
        """Test persistence_boost uses custom default for unseen states."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("new_state", 100.0)

        # Using custom default of 50.0
        # At duration = 50, boost should be exp(-50/50) = exp(-1)
        boost = tracker.persistence_boost("new_state", 150.0, default_expected=50.0)
        expected = math.exp(-1.0)
        assert abs(boost - expected) < 1e-6

    def test_boost_handles_very_short_expected(self: Self) -> None:
        """Test persistence_boost handles near-zero expected durations."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)

        # Should not raise error even with very small default
        boost = tracker.persistence_boost("on", 100.0, default_expected=1e-10)
        assert 0.0 <= boost <= 1.0


class TestStatePersistenceTrackerSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self: Self) -> None:
        """Test serialization to dictionary."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        tracker.update("off", 200.0)

        data = tracker.to_dict()

        assert "hyper_parameters" in data
        assert "mean_duration" in data
        assert "last_ts" in data
        assert "current_state" in data
        assert "current_state_start" in data

    def test_from_dict(self: Self) -> None:
        """Test deserialization from dictionary."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )

        data = {
            "hyper_parameters": {"persistence_half_life_factor": 2.0},
            "mean_duration": {"on": 100.0, "off": 50.0},
            "last_ts": 300.0,
            "current_state": "off",
            "current_state_start": 200.0,
        }

        tracker = StatePersistenceTracker.from_dict(data, base_hp)

        assert tracker.current_state == "off"
        assert tracker.expected_duration("on") == 100.0
        assert tracker.expected_duration("off") == 50.0
        assert tracker.current_duration(300.0) == 100.0

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves state."""
        hp = create_test_hp(half_life=60.0, persistence_half_life_factor=1.5)
        original = StatePersistenceTracker(hp)

        original.update("on", 100.0)
        original.update("off", 200.0)
        original.update("on", 250.0)

        data = original.to_dict()

        base_hp = HyperParameters(
            half_life=60.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        restored = StatePersistenceTracker.from_dict(data, base_hp)

        assert restored.current_state == original.current_state
        assert restored.current_duration(250.0) == original.current_duration(250.0)
        assert restored.expected_duration("on") == original.expected_duration("on")
        assert restored.expected_duration("off") == original.expected_duration("off")

    def test_serialization_with_no_state(self: Self) -> None:
        """Test serialization of tracker with no observations."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        data = tracker.to_dict()

        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        restored = StatePersistenceTracker.from_dict(data, base_hp)

        assert restored.current_state is None
        assert restored.current_duration(100.0) == 0.0

    def test_serialization_preserves_mean_duration_dict(self: Self) -> None:
        """Test that serialization creates a copy of mean_duration dict."""
        hp = create_test_hp()
        tracker = StatePersistenceTracker(hp)

        tracker.update("on", 100.0)
        tracker.update("off", 200.0)

        data = tracker.to_dict()

        # Modify the serialized dict
        data["mean_duration"]["on"] = 999.0

        # Original tracker should be unchanged
        assert tracker.expected_duration("on") != 999.0
