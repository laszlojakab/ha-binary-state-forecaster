"""
Unit tests for DriftStats.

Comprehensive tests for exponentially weighted statistics tracking used
for adaptive drift thresholds.
"""

import math
from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats import (
    DriftStats,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_hyper_parameters import (
    DriftStatsHyperParameters,
)


def create_test_hp(half_life_factor: float = 1.0) -> DriftStatsHyperParameters:
    """Create test hyper-parameters."""
    base_hp = HyperParameters(
        half_life=50.0,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
    return DriftStatsHyperParameters(
        hyper_parameters=drift_hp,
        half_life_factor=half_life_factor,
    )


class TestDriftStatsInitialization:
    """Tests for DriftStats initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating DriftStats with default configuration."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        assert stats.mean == 0.0
        assert stats.var == 0.0
        assert stats.std >= 0.0

    def test_initial_values(self: Self) -> None:
        """Test initial values are zero."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        assert stats.mean == 0.0
        assert stats.var == 0.0


class TestDriftStatsUpdate:
    """Tests for DriftStats.update method."""

    def test_first_update(self: Self) -> None:
        """Test first update sets mean to value."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)

        assert abs(stats.mean - 0.5) < 1e-9
        assert stats.var == 0.0

    def test_multiple_updates_same_value(self: Self) -> None:
        """Test multiple updates with same value stabilize mean."""
        hp = create_test_hp()
        stats = DriftStats(hp)

        for i in range(10):
            stats.update(0.3, 100.0 + i * 10.0)

        assert abs(stats.mean - 0.3) < 0.01
        assert stats.var < 0.01

    def test_update_with_zero_time_delta(self: Self) -> None:
        """Test that zero time delta is ignored."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)
        mean1 = stats.mean

        stats.update(0.9, 100.0)
        mean2 = stats.mean

        assert mean1 == mean2

    def test_update_with_negative_time_delta(self: Self) -> None:
        """Test that negative time delta is ignored."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)
        mean1 = stats.mean

        stats.update(0.9, 50.0)
        mean2 = stats.mean

        assert mean1 == mean2

    def test_varying_updates_compute_variance(self: Self) -> None:
        """Test that varying values produce non-zero variance."""
        hp = create_test_hp()
        stats = DriftStats(hp)

        values = [0.1, 0.5, 0.2, 0.8, 0.3]
        for i, v in enumerate(values):
            stats.update(v, 100.0 + i * 10.0)

        assert stats.var > 0


class TestDriftStatsMean:
    """Tests for mean property."""

    def test_mean_after_updates(self: Self) -> None:
        """Test mean computation after updates."""
        hp = create_test_hp()
        stats = DriftStats(hp)

        stats.update(0.4, 100.0)
        stats.update(0.6, 110.0)

        # Mean should be between 0.4 and 0.6
        assert 0.4 <= stats.mean <= 0.6

    def test_mean_multiple_calls(self: Self) -> None:
        """Test that mean returns consistent value."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)

        mean1 = stats.mean
        mean2 = stats.mean

        assert mean1 == mean2


class TestDriftStatsVariance:
    """Tests for variance and std properties."""

    def test_variance_after_first_update(self: Self) -> None:
        """Test variance is zero after first update."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)

        assert stats.var == 0.0

    def test_std_computation(self: Self) -> None:
        """Test that std is sqrt of variance."""
        hp = create_test_hp()
        stats = DriftStats(hp)

        stats.update(0.5, 100.0)
        stats.update(0.3, 110.0)
        stats.update(0.7, 120.0)

        expected_std = math.sqrt(stats.var)
        assert abs(stats.std - expected_std) < 1e-9

    def test_std_has_floor(self: Self) -> None:
        """Test that std has a minimum floor value."""
        hp = create_test_hp()
        stats = DriftStats(hp)
        stats.update(0.5, 100.0)

        # Even with zero variance, std should be >= sqrt(1e-12)
        assert stats.std >= math.sqrt(1e-12) * 0.99


class TestDriftStatsDecay:
    """Tests for exponential decay behavior."""

    def test_decay_with_different_half_lives(self: Self) -> None:
        """Test that different half-lives affect decay rate."""
        hp_fast = create_test_hp(half_life_factor=0.5)
        hp_slow = create_test_hp(half_life_factor=2.0)

        stats_fast = DriftStats(hp_fast)
        stats_slow = DriftStats(hp_slow)

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
        stats = DriftStats(hp)

        stats.update(0.0, 100.0)
        stats.update(1.0, 110.0)
        stats.update(0.5, 120.0)

        assert 0.0 <= stats.mean <= 1.0

    def test_long_time_gaps(self: Self) -> None:
        """Test with large time gaps between updates."""
        hp = create_test_hp()
        stats = DriftStats(hp)

        stats.update(0.5, 100.0)
        # Very large time gap
        stats.update(0.8, 10000.0)

        # After long decay, new value should dominate
        assert abs(stats.mean - 0.8) < 0.1
