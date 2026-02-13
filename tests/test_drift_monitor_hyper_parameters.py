"""
Unit tests for DriftMonitorHyperParameters.

Comprehensive tests for the hyper-parameter configuration used by
DriftMonitor.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (  # noqa: E501
    DriftMonitorHyperParameters,
)


def create_base_hp(half_life: float = 50.0) -> HyperParameters:
    """Create base hyper-parameters for testing."""
    return HyperParameters(
        half_life=half_life,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )


class TestDriftMonitorHyperParametersInitialization:
    """Tests for DriftMonitorHyperParameters initialization."""

    def test_create_with_defaults(self: Self) -> None:
        """Test creating hyper-parameters with default values."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)

        assert hp.slow_half_life_factor == 20
        assert hp.fast_half_life_factor == 1.5
        assert hp.drift_half_life_factor == 30
        assert hp.tau_enter == 0.1
        assert hp.tau_exit == 0.05
        assert hp.adaptive_tau is True
        assert hp.n_enter == 3
        assert hp.n_exit == 5

    def test_create_with_custom_values(self: Self) -> None:
        """Test creating hyper-parameters with custom values."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_half_life_factor=30.0,
            fast_half_life_factor=2.0,
            drift_half_life_factor=40.0,
            tau_enter=0.2,
            tau_exit=0.1,
            adaptive_tau=False,
            n_enter=5,
            n_exit=10,
        )

        assert hp.slow_half_life_factor == 30.0
        assert hp.fast_half_life_factor == 2.0
        assert hp.drift_half_life_factor == 40.0
        assert hp.tau_enter == 0.2
        assert hp.tau_exit == 0.1
        assert hp.adaptive_tau is False
        assert hp.n_enter == 5
        assert hp.n_exit == 10

    def test_pruning_and_smoothing_parameters(self: Self) -> None:
        """Test creating with custom pruning and smoothing parameters."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_prune_threshold=1e-5,
            slow_epsilon=1e-8,
            fast_prune_threshold=1e-4,
            fast_epsilon=1e-7,
        )

        assert hp.slow_prune_threshold == 1e-5
        assert hp.slow_epsilon == 1e-8
        assert hp.fast_prune_threshold == 1e-4
        assert hp.fast_epsilon == 1e-7


class TestDriftMonitorHyperParametersProperties:
    """Tests for property accessors."""

    def test_half_life_property(self: Self) -> None:
        """Test that half_life property returns base half-life."""
        base_hp = create_base_hp(half_life=75.0)
        hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)

        assert hp.half_life == 75.0

    def test_half_life_updates_with_base(self: Self) -> None:
        """Test that half_life reflects changes in base hyper-parameters."""
        base_hp = create_base_hp(half_life=50.0)
        hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)

        assert hp.half_life == 50.0

        base_hp.update(half_life=100.0)

        assert hp.half_life == 100.0

    def test_all_factor_properties(self: Self) -> None:
        """Test all factor properties."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_half_life_factor=25.0,
            fast_half_life_factor=2.5,
            drift_half_life_factor=35.0,
        )

        assert hp.slow_half_life_factor == 25.0
        assert hp.fast_half_life_factor == 2.5
        assert hp.drift_half_life_factor == 35.0

    def test_threshold_properties(self: Self) -> None:
        """Test threshold properties."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            tau_enter=0.15,
            tau_exit=0.08,
        )

        assert hp.tau_enter == 0.15
        assert hp.tau_exit == 0.08

    def test_adaptive_tau_property(self: Self) -> None:
        """Test adaptive_tau property."""
        base_hp = create_base_hp()

        hp_adaptive = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            adaptive_tau=True,
        )
        assert hp_adaptive.adaptive_tau is True

        hp_fixed = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            adaptive_tau=False,
        )
        assert hp_fixed.adaptive_tau is False

    def test_counter_properties(self: Self) -> None:
        """Test n_enter and n_exit properties."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            n_enter=7,
            n_exit=12,
        )

        assert hp.n_enter == 7
        assert hp.n_exit == 12


class TestDriftMonitorHyperParametersSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_structure(self: Self) -> None:
        """Test that to_dict returns all expected keys."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)

        data = hp.to_dict()

        expected_keys = {
            "half_life",
            "slow_half_life_factor",
            "slow_prune_threshold",
            "slow_epsilon",
            "fast_half_life_factor",
            "fast_prune_threshold",
            "fast_epsilon",
            "drift_half_life_factor",
            "tau_enter",
            "tau_exit",
            "adaptive_tau",
            "n_enter",
            "n_exit",
        }

        assert set(data.keys()) == expected_keys

    def test_to_dict_values(self: Self) -> None:
        """Test that to_dict returns correct values."""
        base_hp = create_base_hp(half_life=60.0)
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_half_life_factor=25.0,
            tau_enter=0.2,
            adaptive_tau=False,
            n_enter=4,
        )

        data = hp.to_dict()

        assert data["half_life"] == 60.0
        assert data["slow_half_life_factor"] == 25.0
        assert data["tau_enter"] == 0.2
        assert data["adaptive_tau"] is False
        assert data["n_enter"] == 4

    def test_from_dict(self: Self) -> None:
        """Test deserialization from dictionary."""
        base_hp = create_base_hp(half_life=50.0)

        data = {
            "slow_half_life_factor": 22.0,
            "slow_prune_threshold": 1e-5,
            "slow_epsilon": 1e-8,
            "fast_half_life_factor": 2.0,
            "fast_prune_threshold": 1e-4,
            "fast_epsilon": 1e-7,
            "drift_half_life_factor": 35.0,
            "tau_enter": 0.15,
            "tau_exit": 0.07,
            "adaptive_tau": False,
            "n_enter": 4,
            "n_exit": 6,
        }

        hp = DriftMonitorHyperParameters.from_dict(data, base_hp)

        assert hp.slow_half_life_factor == 22.0
        assert hp.slow_prune_threshold == 1e-5
        assert hp.slow_epsilon == 1e-8
        assert hp.fast_half_life_factor == 2.0
        assert hp.fast_prune_threshold == 1e-4
        assert hp.fast_epsilon == 1e-7
        assert hp.drift_half_life_factor == 35.0
        assert hp.tau_enter == 0.15
        assert hp.tau_exit == 0.07
        assert hp.adaptive_tau is False
        assert hp.n_enter == 4
        assert hp.n_exit == 6

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves values."""
        base_hp = create_base_hp(half_life=75.0)
        original = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_half_life_factor=18.0,
            fast_half_life_factor=1.2,
            drift_half_life_factor=28.0,
            tau_enter=0.12,
            tau_exit=0.06,
            adaptive_tau=True,
            n_enter=2,
            n_exit=4,
        )

        data = original.to_dict()
        restored = DriftMonitorHyperParameters.from_dict(data, base_hp)

        assert restored.slow_half_life_factor == original.slow_half_life_factor
        assert restored.fast_half_life_factor == original.fast_half_life_factor
        assert restored.drift_half_life_factor == original.drift_half_life_factor
        assert restored.tau_enter == original.tau_enter
        assert restored.tau_exit == original.tau_exit
        assert restored.adaptive_tau == original.adaptive_tau
        assert restored.n_enter == original.n_enter
        assert restored.n_exit == original.n_exit

    def test_serialization_with_defaults(self: Self) -> None:
        """Test serialization with default values."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)

        data = hp.to_dict()
        restored = DriftMonitorHyperParameters.from_dict(data, base_hp)

        assert restored.slow_half_life_factor == 20
        assert restored.fast_half_life_factor == 1.5
        assert restored.drift_half_life_factor == 30
        assert restored.tau_enter == 0.1
        assert restored.tau_exit == 0.05
        assert restored.adaptive_tau is True
        assert restored.n_enter == 3
        assert restored.n_exit == 5

    def test_multiple_round_trips(self: Self) -> None:
        """Test multiple serialization/deserialization cycles."""
        base_hp = create_base_hp(half_life=50.0)
        original = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_half_life_factor=25.0,
            n_enter=7,
        )

        # First round trip
        data1 = original.to_dict()
        hp1 = DriftMonitorHyperParameters.from_dict(data1, base_hp)

        # Second round trip
        data2 = hp1.to_dict()
        hp2 = DriftMonitorHyperParameters.from_dict(data2, base_hp)

        assert hp2.slow_half_life_factor == original.slow_half_life_factor
        assert hp2.n_enter == original.n_enter
        assert data1 == data2


class TestDriftMonitorHyperParametersEdgeCases:
    """Tests for edge cases."""

    def test_zero_thresholds(self: Self) -> None:
        """Test with zero threshold values."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            tau_enter=0.0,
            tau_exit=0.0,
        )

        assert hp.tau_enter == 0.0
        assert hp.tau_exit == 0.0

    def test_large_counters(self: Self) -> None:
        """Test with large counter values."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            n_enter=100,
            n_exit=200,
        )

        assert hp.n_enter == 100
        assert hp.n_exit == 200

    def test_zero_pruning_threshold(self: Self) -> None:
        """Test with zero pruning threshold (no pruning)."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_prune_threshold=0.0,
            fast_prune_threshold=0.0,
        )

        assert hp.slow_prune_threshold == 0.0
        assert hp.fast_prune_threshold == 0.0

    def test_zero_epsilon(self: Self) -> None:
        """Test with zero epsilon (no smoothing)."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            slow_epsilon=0.0,
            fast_epsilon=0.0,
        )

        assert hp.slow_epsilon == 0.0
        assert hp.fast_epsilon == 0.0

    def test_single_counter_values(self: Self) -> None:
        """Test with counter values of 1."""
        base_hp = create_base_hp()
        hp = DriftMonitorHyperParameters(
            hyper_parameters=base_hp,
            n_enter=1,
            n_exit=1,
        )

        assert hp.n_enter == 1
        assert hp.n_exit == 1
