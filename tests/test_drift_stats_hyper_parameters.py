"""
Unit tests for DriftStatsHyperParameters.

Comprehensive tests for the hyper-parameter configuration used by
DriftStats.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_hyper_parameters import (
    DriftStatsHyperParameters,
)


def create_drift_monitor_hp(half_life: float = 50.0) -> DriftMonitorHyperParameters:
    """Create drift monitor hyper-parameters for testing."""
    base_hp = HyperParameters(
        half_life=half_life,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    return DriftMonitorHyperParameters(hyper_parameters=base_hp)


class TestDriftStatsHyperParametersInitialization:
    """Tests for DriftStatsHyperParameters initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating hyper-parameters with default factor."""
        drift_hp = create_drift_monitor_hp()
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
        )
        assert hp.drift_half_life == 50.0

    def test_create_with_custom_factor(self: Self) -> None:
        """Test creating hyper-parameters with custom factor."""
        drift_hp = create_drift_monitor_hp(half_life=100.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=2.0,
        )
        assert hp.drift_half_life == 200.0

    def test_zero_factor(self: Self) -> None:
        """Test creating hyper-parameters with zero factor."""
        drift_hp = create_drift_monitor_hp()
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.0,
        )
        assert hp.drift_half_life == 0.0

    def test_fractional_factor(self: Self) -> None:
        """Test creating hyper-parameters with fractional factor."""
        drift_hp = create_drift_monitor_hp(half_life=60.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.5,
        )
        assert hp.drift_half_life == 30.0

    def test_large_factor(self: Self) -> None:
        """Test creating hyper-parameters with large factor."""
        drift_hp = create_drift_monitor_hp(half_life=10.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=100.0,
        )
        assert hp.drift_half_life == 1000.0


class TestDriftStatsHyperParametersDriftHalfLife:
    """Tests for drift_half_life property."""

    def test_drift_half_life_calculation(self: Self) -> None:
        """Test that drift half-life is correctly calculated."""
        drift_hp = create_drift_monitor_hp(half_life=80.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.5,
        )
        assert hp.drift_half_life == 120.0

    def test_drift_half_life_updates_with_base(self: Self) -> None:
        """Test that drift half-life reflects changes in base half-life."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=2.0,
        )
        
        # Initial value
        assert hp.drift_half_life == 100.0
        
        # Update base half-life
        base_hp.update(half_life=100.0)
        
        # Drift half-life should reflect the change
        assert hp.drift_half_life == 200.0

    def test_multiple_factors(self: Self) -> None:
        """Test drift half-life with different base and factor combinations."""
        test_cases = [
            (50.0, 1.0, 50.0),
            (100.0, 0.5, 50.0),
            (25.0, 4.0, 100.0),
            (75.0, 2.0, 150.0),
        ]
        
        for base_half_life, factor, expected in test_cases:
            drift_hp = create_drift_monitor_hp(half_life=base_half_life)
            hp = DriftStatsHyperParameters(
                hyper_parameters=drift_hp,
                half_life_factor=factor,
            )
            assert hp.drift_half_life == expected


class TestDriftStatsHyperParametersSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self: Self) -> None:
        """Test serialization to dictionary."""
        drift_hp = create_drift_monitor_hp()
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.5,
        )
        
        data = hp.to_dict()
        
        assert "half_life_factor" in data
        assert data["half_life_factor"] == 1.5

    def test_to_dict_structure(self: Self) -> None:
        """Test that to_dict returns only the expected keys."""
        drift_hp = create_drift_monitor_hp()
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=2.0,
        )
        
        data = hp.to_dict()
        
        assert len(data) == 1
        assert "half_life_factor" in data

    def test_from_dict(self: Self) -> None:
        """Test deserialization from dictionary."""
        drift_hp = create_drift_monitor_hp(half_life=60.0)
        data = {"half_life_factor": 2.5}
        
        hp = DriftStatsHyperParameters.from_dict(data, drift_hp)
        
        assert hp.drift_half_life == 150.0

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves values."""
        drift_hp = create_drift_monitor_hp(half_life=75.0)
        original_hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.8,
        )
        
        data = original_hp.to_dict()
        restored_hp = DriftStatsHyperParameters.from_dict(data, drift_hp)
        
        assert restored_hp.drift_half_life == original_hp.drift_half_life

    def test_serialization_with_zero_factor(self: Self) -> None:
        """Test serialization with zero factor."""
        drift_hp = create_drift_monitor_hp()
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.0,
        )
        
        data = hp.to_dict()
        restored_hp = DriftStatsHyperParameters.from_dict(data, drift_hp)
        
        assert restored_hp.drift_half_life == 0.0

    def test_serialization_with_large_factor(self: Self) -> None:
        """Test serialization with large factor."""
        drift_hp = create_drift_monitor_hp(half_life=10.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=100.0,
        )
        
        data = hp.to_dict()
        restored_hp = DriftStatsHyperParameters.from_dict(data, drift_hp)
        
        assert restored_hp.drift_half_life == 1000.0

    def test_serialization_with_fractional_factor(self: Self) -> None:
        """Test serialization with fractional factor."""
        drift_hp = create_drift_monitor_hp(half_life=80.0)
        hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.25,
        )
        
        data = hp.to_dict()
        restored_hp = DriftStatsHyperParameters.from_dict(data, drift_hp)
        
        assert restored_hp.drift_half_life == 20.0

    def test_multiple_round_trips(self: Self) -> None:
        """Test multiple serialization/deserialization cycles."""
        drift_hp = create_drift_monitor_hp(half_life=50.0)
        original_hp = DriftStatsHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=3.0,
        )
        
        # First round trip
        data1 = original_hp.to_dict()
        hp1 = DriftStatsHyperParameters.from_dict(data1, drift_hp)
        
        # Second round trip
        data2 = hp1.to_dict()
        hp2 = DriftStatsHyperParameters.from_dict(data2, drift_hp)
        
        assert hp2.drift_half_life == original_hp.drift_half_life
        assert data1 == data2
