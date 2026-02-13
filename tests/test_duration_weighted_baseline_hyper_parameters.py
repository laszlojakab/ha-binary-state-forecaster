"""
Unit tests for DurationWeightedBaselineHyperParameters.

Comprehensive tests for the hyper-parameter configuration used by
DurationWeightedBaseline.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (  # noqa: E501
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.duration_weighted_baseline_hyper_parameters import (  # noqa: E501
    DurationWeightedBaselineHyperParameters,
)


def create_drift_hp(half_life: float = 50.0) -> DriftMonitorHyperParameters:
    """Create drift monitor hyper-parameters for testing."""
    base_hp = HyperParameters(
        half_life=half_life,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    return DriftMonitorHyperParameters(hyper_parameters=base_hp)


class TestDurationWeightedBaselineHyperParametersInitialization:
    """Tests for DurationWeightedBaselineHyperParameters initialization."""

    def test_create_with_defaults(self: Self) -> None:
        """Test creating hyper-parameters with default values."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
        )

        assert hp.baseline_half_life == 50.0
        assert hp.prune_threshold == 1e-6
        assert hp.epsilon == 1e-9

    def test_create_with_custom_values(self: Self) -> None:
        """Test creating hyper-parameters with custom values."""
        drift_hp = create_drift_hp(half_life=100.0)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=2.0,
            prune_threshold=1e-5,
            epsilon=1e-8,
        )

        assert hp.baseline_half_life == 200.0
        assert hp.prune_threshold == 1e-5
        assert hp.epsilon == 1e-8

    def test_zero_half_life_factor(self: Self) -> None:
        """Test creating hyper-parameters with zero half-life factor."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.0,
        )

        assert hp.baseline_half_life == 0.0

    def test_fractional_half_life_factor(self: Self) -> None:
        """Test creating hyper-parameters with fractional half-life factor."""
        drift_hp = create_drift_hp(half_life=60.0)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.5,
        )

        assert hp.baseline_half_life == 30.0


class TestDurationWeightedBaselineHyperParametersBaselineHalfLife:
    """Tests for baseline_half_life property."""

    def test_baseline_half_life_calculation(self: Self) -> None:
        """Test that baseline half-life is correctly calculated."""
        drift_hp = create_drift_hp(half_life=80.0)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.5,
        )

        assert hp.baseline_half_life == 120.0

    def test_baseline_half_life_updates_with_base(self: Self) -> None:
        """Test that baseline half-life reflects changes in base half-life."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=2.0,
        )

        # Initial value
        assert hp.baseline_half_life == 100.0

        # Update base half-life
        base_hp.update(half_life=100.0)

        # Baseline half-life should reflect the change
        assert hp.baseline_half_life == 200.0

    def test_large_half_life_factor(self: Self) -> None:
        """Test baseline half-life with large factor."""
        drift_hp = create_drift_hp(half_life=10.0)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=100.0,
        )

        assert hp.baseline_half_life == 1000.0


class TestDurationWeightedBaselineHyperParametersPruneThreshold:
    """Tests for prune_threshold property."""

    def test_default_prune_threshold(self: Self) -> None:
        """Test default prune threshold value."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
        )

        assert hp.prune_threshold == 1e-6

    def test_custom_prune_threshold(self: Self) -> None:
        """Test custom prune threshold value."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
            prune_threshold=1e-4,
        )

        assert hp.prune_threshold == 1e-4

    def test_zero_prune_threshold(self: Self) -> None:
        """Test zero prune threshold (no pruning)."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
            prune_threshold=0.0,
        )

        assert hp.prune_threshold == 0.0


class TestDurationWeightedBaselineHyperParametersEpsilon:
    """Tests for epsilon property."""

    def test_default_epsilon(self: Self) -> None:
        """Test default epsilon value."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
        )

        assert hp.epsilon == 1e-9

    def test_custom_epsilon(self: Self) -> None:
        """Test custom epsilon value."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
            epsilon=1e-6,
        )

        assert hp.epsilon == 1e-6

    def test_zero_epsilon(self: Self) -> None:
        """Test zero epsilon (no smoothing)."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
            epsilon=0.0,
        )

        assert hp.epsilon == 0.0


class TestDurationWeightedBaselineHyperParametersSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self: Self) -> None:
        """Test serialization to dictionary."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.5,
            prune_threshold=1e-5,
            epsilon=1e-8,
        )

        data = hp.to_dict()

        assert "half_life_factor" in data
        assert "prune_threshold" in data
        assert "epsilon" in data
        assert data["half_life_factor"] == 1.5
        assert data["prune_threshold"] == 1e-5
        assert data["epsilon"] == 1e-8

    def test_from_dict(self: Self) -> None:
        """Test deserialization from dictionary."""
        drift_hp = create_drift_hp(half_life=60.0)
        data = {
            "half_life_factor": 2.5,
            "prune_threshold": 1e-4,
            "epsilon": 1e-7,
        }

        hp = DurationWeightedBaselineHyperParameters.from_dict(data, drift_hp)

        assert hp.baseline_half_life == 150.0
        assert hp.prune_threshold == 1e-4
        assert hp.epsilon == 1e-7

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves values."""
        drift_hp = create_drift_hp(half_life=75.0)
        original_hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.8,
            prune_threshold=2e-5,
            epsilon=3e-8,
        )

        data = original_hp.to_dict()
        restored_hp = DurationWeightedBaselineHyperParameters.from_dict(data, drift_hp)

        assert restored_hp.baseline_half_life == original_hp.baseline_half_life
        assert restored_hp.prune_threshold == original_hp.prune_threshold
        assert restored_hp.epsilon == original_hp.epsilon

    def test_serialization_with_defaults(self: Self) -> None:
        """Test serialization with default values."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=1.0,
        )

        data = hp.to_dict()
        restored_hp = DurationWeightedBaselineHyperParameters.from_dict(data, drift_hp)

        assert restored_hp.baseline_half_life == 50.0
        assert restored_hp.prune_threshold == 1e-6
        assert restored_hp.epsilon == 1e-9

    def test_serialization_with_zero_values(self: Self) -> None:
        """Test serialization with zero values."""
        drift_hp = create_drift_hp()
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=0.0,
            prune_threshold=0.0,
            epsilon=0.0,
        )

        data = hp.to_dict()
        restored_hp = DurationWeightedBaselineHyperParameters.from_dict(data, drift_hp)

        assert restored_hp.baseline_half_life == 0.0
        assert restored_hp.prune_threshold == 0.0
        assert restored_hp.epsilon == 0.0

    def test_serialization_with_large_values(self: Self) -> None:
        """Test serialization with large values."""
        drift_hp = create_drift_hp(half_life=10.0)
        hp = DurationWeightedBaselineHyperParameters(
            hyper_parameters=drift_hp,
            half_life_factor=100.0,
            prune_threshold=0.1,
            epsilon=0.01,
        )

        data = hp.to_dict()
        restored_hp = DurationWeightedBaselineHyperParameters.from_dict(data, drift_hp)

        assert restored_hp.baseline_half_life == 1000.0
        assert restored_hp.prune_threshold == 0.1
        assert restored_hp.epsilon == 0.01
