"""
Unit tests for StatePersistenceTrackerHyperParameters.

Comprehensive tests for the hyper-parameter configuration used by
StatePersistenceTracker.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_hyper_parameters import (  # noqa: E501
    StatePersistenceTrackerHyperParameters,
)


def create_base_hp(half_life: float = 50.0) -> HyperParameters:
    """Create base hyper-parameters for testing."""
    return HyperParameters(
        half_life=half_life,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )


class TestStatePersistenceTrackerHyperParametersInitialization:
    """Tests for StatePersistenceTrackerHyperParameters initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating hyper-parameters with default factor."""
        base_hp = create_base_hp()
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=1.0,
        )
        assert hp.persistence_half_life == 50.0

    def test_create_with_custom_factor(self: Self) -> None:
        """Test creating hyper-parameters with custom factor."""
        base_hp = create_base_hp(half_life=100.0)
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=2.0,
        )
        assert hp.persistence_half_life == 200.0

    def test_zero_factor(self: Self) -> None:
        """Test creating hyper-parameters with zero factor."""
        base_hp = create_base_hp()
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=0.0,
        )
        assert hp.persistence_half_life == 0.0

    def test_fractional_factor(self: Self) -> None:
        """Test creating hyper-parameters with fractional factor."""
        base_hp = create_base_hp(half_life=60.0)
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=0.5,
        )
        assert hp.persistence_half_life == 30.0


class TestStatePersistenceTrackerHyperParametersPersistenceHalfLife:
    """Tests for persistence_half_life property."""

    def test_persistence_half_life_calculation(self: Self) -> None:
        """Test that persistence half-life is correctly calculated."""
        base_hp = create_base_hp(half_life=80.0)
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=1.5,
        )
        assert hp.persistence_half_life == 120.0

    def test_persistence_half_life_updates_with_base(self: Self) -> None:
        """Test that persistence half-life reflects changes in base half-life."""
        base_hp = create_base_hp(half_life=50.0)
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=2.0,
        )

        # Initial value
        assert hp.persistence_half_life == 100.0

        # Update base half-life
        base_hp.update(half_life=100.0)

        # Persistence half-life should reflect the change
        assert hp.persistence_half_life == 200.0


class TestStatePersistenceTrackerHyperParametersSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self: Self) -> None:
        """Test serialization to dictionary."""
        base_hp = create_base_hp()
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=1.5,
        )

        data = hp.to_dict()

        assert "persistence_half_life_factor" in data
        assert data["persistence_half_life_factor"] == 1.5

    def test_from_dict(self: Self) -> None:
        """Test deserialization from dictionary."""
        base_hp = create_base_hp(half_life=60.0)
        data = {"persistence_half_life_factor": 2.5}

        hp = StatePersistenceTrackerHyperParameters.from_dict(data, base_hp)

        assert hp.persistence_half_life == 150.0

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves values."""
        base_hp = create_base_hp(half_life=75.0)
        original_hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=1.8,
        )

        data = original_hp.to_dict()
        restored_hp = StatePersistenceTrackerHyperParameters.from_dict(data, base_hp)

        assert restored_hp.persistence_half_life == original_hp.persistence_half_life

    def test_serialization_with_zero_factor(self: Self) -> None:
        """Test serialization with zero factor."""
        base_hp = create_base_hp()
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=0.0,
        )

        data = hp.to_dict()
        restored_hp = StatePersistenceTrackerHyperParameters.from_dict(data, base_hp)

        assert restored_hp.persistence_half_life == 0.0

    def test_serialization_with_large_factor(self: Self) -> None:
        """Test serialization with large factor."""
        base_hp = create_base_hp(half_life=10.0)
        hp = StatePersistenceTrackerHyperParameters(
            hyper_parameters=base_hp,
            persistence_half_life_factor=100.0,
        )

        data = hp.to_dict()
        restored_hp = StatePersistenceTrackerHyperParameters.from_dict(data, base_hp)

        assert restored_hp.persistence_half_life == 1000.0
