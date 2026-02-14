"""
Unit tests for DriftStats.

Comprehensive tests for exponentially weighted statistics tracking used
for adaptive drift thresholds.
"""

import math
from typing import Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (  # noqa: E501
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats import (
    DriftStats,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_hyper_parameters import (  # noqa: E501
    DriftStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_runtime_parameters import (
    DriftStatsRuntimeParameters,
)


def create_test_hp() -> DriftStatsHyperParameters:
    """Create test hyper-parameters."""
    base_hp = ForecasterEngineHyperParameters(
        half_life=50.0,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
    return DriftStatsHyperParameters(hyper_parameters=drift_hp)


def create_test_rp(half_life_factor: float = 1.0) -> DriftStatsRuntimeParameters:
    """Create test runtime parameters."""
    return DriftStatsRuntimeParameters(half_life_factor=half_life_factor)


class TestDriftStatsInitialization:
    """Tests for DriftStats initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating DriftStats with default configuration."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        assert stats.mean == 0.0
        assert stats.var == 0.0
        assert stats.std >= 0.0

    def test_initial_values(self: Self) -> None:
        """Test initial values are zero."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        assert stats.mean == 0.0
        assert stats.var == 0.0


class TestDriftStatsUpdate:
    """Tests for DriftStats.update method."""

    def test_first_update(self: Self) -> None:
        """Test first update sets mean to value."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)

        assert abs(stats.mean - 0.5) < 1e-9
        assert stats.var == 0.0

    def test_multiple_updates_same_value(self: Self) -> None:
        """Test multiple updates with same value stabilize mean."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        for i in range(10):
            stats.update(0.3, 100.0 + i * 10.0)

        assert abs(stats.mean - 0.3) < 0.01
        assert stats.var < 0.01

    def test_update_with_zero_time_delta(self: Self) -> None:
        """Test that zero time delta is ignored."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)
        mean1 = stats.mean

        stats.update(0.9, 100.0)
        mean2 = stats.mean

        assert mean1 == mean2

    def test_update_with_negative_time_delta(self: Self) -> None:
        """Test that negative time delta is ignored."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)
        mean1 = stats.mean

        stats.update(0.9, 50.0)
        mean2 = stats.mean

        assert mean1 == mean2

    def test_varying_updates_compute_variance(self: Self) -> None:
        """Test that varying values produce non-zero variance."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        values = [0.1, 0.5, 0.2, 0.8, 0.3]
        for i, v in enumerate(values):
            stats.update(v, 100.0 + i * 10.0)

        assert stats.var > 0


class TestDriftStatsMean:
    """Tests for mean property."""

    def test_mean_after_updates(self: Self) -> None:
        """Test mean computation after updates."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        stats.update(0.4, 100.0)
        stats.update(0.6, 110.0)

        # Mean should be between 0.4 and 0.6
        assert 0.4 <= stats.mean <= 0.6

    def test_mean_multiple_calls(self: Self) -> None:
        """Test that mean returns consistent value."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)

        mean1 = stats.mean
        mean2 = stats.mean

        assert mean1 == mean2


class TestDriftStatsVariance:
    """Tests for variance and std properties."""

    def test_variance_after_first_update(self: Self) -> None:
        """Test variance is zero after first update."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)

        assert stats.var == 0.0

    def test_std_computation(self: Self) -> None:
        """Test that std is sqrt of variance."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        stats.update(0.5, 100.0)
        stats.update(0.3, 110.0)
        stats.update(0.7, 120.0)

        expected_std = math.sqrt(stats.var)
        assert abs(stats.std - expected_std) < 1e-9

    def test_std_has_floor(self: Self) -> None:
        """Test that std has a minimum floor value."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)
        stats.update(0.5, 100.0)

        # Even with zero variance, std should be >= sqrt(1e-12)
        assert stats.std >= math.sqrt(1e-12) * 0.99


class TestDriftStatsDecay:
    """Tests for exponential decay behavior."""

    def test_decay_with_different_half_lives(self: Self) -> None:
        """Test that different half-lives affect decay rate."""
        hp_fast = create_test_hp()
        hp_slow = create_test_hp()
        rp_fast = create_test_rp(half_life_factor=0.5)
        rp_slow = create_test_rp(half_life_factor=2.0)

        stats_fast = DriftStats(hp_fast, rp_fast)
        stats_slow = DriftStats(hp_slow, rp_slow)

        # Initialize both
        stats_fast.update(0.8, 100.0)
        stats_slow.update(0.8, 100.0)

        # Update both with lower value after time passes
        stats_fast.update(0.2, 200.0)
        stats_slow.update(0.2, 200.0)

        # Fast should adapt more quickly
        assert abs(stats_fast.mean - 0.2) < abs(stats_slow.mean - 0.2)


class TestDriftStatsEdgeCases:
    """Tests for edge cases."""

    def test_extreme_values(self: Self) -> None:
        """Test with extreme drift values."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        stats.update(0.0, 100.0)
        stats.update(1.0, 110.0)
        stats.update(0.5, 120.0)

        assert 0.0 <= stats.mean <= 1.0

    def test_long_time_gaps(self: Self) -> None:
        """Test with large time gaps between updates."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        stats.update(0.5, 100.0)
        # Very large time gap
        stats.update(0.8, 10000.0)

        # After long decay, new value should dominate
        assert abs(stats.mean - 0.8) < 0.1


class TestDriftStatsSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_structure(self: Self) -> None:
        """Test that to_dict returns correct structure."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        stats.update(0.5, 100.0)
        stats.update(0.3, 110.0)

        data = stats.to_dict()

        assert "mean" in data
        assert "var" in data
        assert "last_ts" in data

    def test_from_dict_reconstruction(self: Self) -> None:
        """Test reconstruction from dictionary."""
        hp = create_test_hp()
        rp = create_test_rp()
        data = {
            "mean": 0.42,
            "var": 0.05,
            "last_ts": 200.0,
        }

        stats = DriftStats.from_dict(data, hp, rp)

        assert stats.mean == 0.42
        assert stats.var == 0.05

    def test_round_trip_serialization(self: Self) -> None:
        """Test that serialization and deserialization preserves state."""
        hp = create_test_hp()
        rp = create_test_rp()
        original = DriftStats(hp, rp)

        original.update(0.3, 100.0)
        original.update(0.5, 120.0)
        original.update(0.4, 140.0)

        data = original.to_dict()
        restored = DriftStats.from_dict(data, hp, rp)

        assert abs(restored.mean - original.mean) < 1e-9
        assert abs(restored.var - original.var) < 1e-9
        assert abs(restored.std - original.std) < 1e-9

    def test_serialization_with_no_updates(self: Self) -> None:
        """Test serialization before any updates."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = DriftStats(hp, rp)

        data = stats.to_dict()
        restored = DriftStats.from_dict(data, hp, rp)

        assert restored.mean == 0.0
        assert restored.var == 0.0

    def test_serialization_after_first_update(self: Self) -> None:
        """Test serialization after single update."""
        hp = create_test_hp()
        rp = create_test_rp()
        original = DriftStats(hp, rp)
        original.update(0.75, 100.0)

        data = original.to_dict()
        restored = DriftStats.from_dict(data, hp, rp)

        assert restored.mean == 0.75
        assert restored.var == 0.0

    def test_serialization_preserves_statistics(self: Self) -> None:
        """Test that serialization preserves all statistics accurately."""
        hp = create_test_hp()
        rp = create_test_rp()
        original = DriftStats(hp, rp)

        # Add varying values to create non-trivial statistics
        values = [0.1, 0.5, 0.2, 0.8, 0.3, 0.6]
        for i, v in enumerate(values):
            original.update(v, 100.0 + i * 10.0)

        data = original.to_dict()
        restored = DriftStats.from_dict(data, hp, rp)

        # All statistics should match
        assert abs(restored.mean - original.mean) < 1e-12
        assert abs(restored.var - original.var) < 1e-12
        assert abs(restored.std - original.std) < 1e-12

    def test_continued_updates_after_deserialization(self: Self) -> None:
        """Test that deserialized stats can be updated."""
        hp = create_test_hp()
        rp = create_test_rp()
        original = DriftStats(hp, rp)

        original.update(0.5, 100.0)
        original.update(0.6, 110.0)

        data = original.to_dict()
        restored = DriftStats.from_dict(data, hp, rp)

        # Continue updating the restored instance
        restored.update(0.7, 120.0)

        assert restored.mean > 0.0
        assert restored.var >= 0.0

    def test_deserialization_with_different_hyper_parameters(self: Self) -> None:
        """Test deserialization with different hyper-parameters."""
        hp1 = create_test_hp()
        hp2 = create_test_hp()
        rp1 = create_test_rp(half_life_factor=1.0)
        rp2 = create_test_rp(half_life_factor=2.0)

        stats = DriftStats(hp1, rp1)
        stats.update(0.5, 100.0)
        stats.update(0.6, 110.0)

        data = stats.to_dict()

        # Restore with different hyper-parameters
        restored = DriftStats.from_dict(data, hp2, rp2)

        # Statistical values should be preserved
        assert abs(restored.mean - stats.mean) < 1e-9
        assert abs(restored.var - stats.var) < 1e-9

        # But future updates will use new hyper-parameters
        # (different decay rates)
