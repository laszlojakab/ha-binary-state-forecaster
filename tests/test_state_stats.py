"""
Unit tests for StateStats.

Comprehensive tests for the StateStats class, covering initialization,
support tracking, decay application, and edge cases.
"""

from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.statistics.state_stats import (
    StateStats,
)


class TestStateStatsInitialization:
    """Tests for StateStats initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating StateStats with default initialization."""
        stats = StateStats()
        assert stats.support == 0.0

    def test_initial_support_is_zero(self: Self) -> None:
        """Test that newly created StateStats has zero support."""
        stats = StateStats()
        assert stats.support == 0.0

    def test_initial_not_active(self: Self) -> None:
        """Test that newly created StateStats is not active with any threshold."""
        stats = StateStats()
        assert not stats.is_active(0.1)
        assert not stats.is_active(1.0)
        assert not stats.is_active(100.0)


class TestStateStatsUpdate:
    """Tests for StateStats.update method."""

    def test_update_single_weight(self: Self) -> None:
        """Test updating with a single weight."""
        stats = StateStats()
        stats.update(5.0)
        assert stats.support == 5.0

    def test_update_multiple_times(self: Self) -> None:
        """Test multiple updates accumulate."""
        stats = StateStats()
        stats.update(2.0)
        stats.update(3.0)
        stats.update(5.0)
        assert stats.support == 10.0

    def test_update_zero_weight(self: Self) -> None:
        """Test updating with zero weight."""
        stats = StateStats()
        stats.update(0.0)
        assert stats.support == 0.0

    def test_update_fractional_weight(self: Self) -> None:
        """Test updating with fractional weights."""
        stats = StateStats()
        stats.update(0.5)
        stats.update(0.3)
        assert abs(stats.support - 0.8) < 1e-9

    def test_update_negative_weight_raises_error(self: Self) -> None:
        """Test that negative weights raise ValueError."""
        stats = StateStats()
        stats.update(10.0)
        # Negative weights are not allowed
        with pytest.raises(ValueError, match="weight must be non negative"):
            stats.update(-3.0)

    def test_update_very_large_weight(self: Self) -> None:
        """Test updating with very large weights."""
        stats = StateStats()
        stats.update(1e10)
        assert stats.support == 1e10

    def test_update_very_small_weight(self: Self) -> None:
        """Test updating with very small positive weights."""
        stats = StateStats()
        stats.update(1e-10)
        assert abs(stats.support - 1e-10) < 1e-15


class TestStateStatsSupport:
    """Tests for StateStats.support method."""

    def test_support_returns_accumulated_weight(self: Self) -> None:
        """Test support returns exact accumulated weight."""
        stats = StateStats()
        stats.update(2.5)
        stats.update(3.5)
        assert abs(stats.support - 6.0) < 1e-9

    def test_support_multiple_calls_same_result(self: Self) -> None:
        """Test support returns the same value on repeated calls."""
        stats = StateStats()
        stats.update(5.0)
        result1 = stats.support
        result2 = stats.support
        assert result1 == result2


class TestStateStatsApplyDecay:
    """Tests for StateStats.apply_decay method."""

    def test_decay_with_factor_0_5(self: Self) -> None:
        """Test decay with factor 0.5."""
        stats = StateStats()
        stats.update(10.0)
        stats.apply_decay(0.5)
        assert stats.support == 5.0

    def test_decay_with_factor_1_0(self: Self) -> None:
        """Test decay with factor 1.0 (no decay)."""
        stats = StateStats()
        stats.update(10.0)
        stats.apply_decay(1.0)
        assert stats.support == 10.0

    def test_decay_with_factor_0_raises_error(self: Self) -> None:
        """Test decay with factor 0 raises ValueError."""
        stats = StateStats()
        stats.update(10.0)
        # Factor must be in (0, 1], not including 0
        with pytest.raises(ValueError, match="decay factor must be in"):
            stats.apply_decay(0.0)

    def test_decay_multiple_times(self: Self) -> None:
        """Test multiple decay operations compound."""
        stats = StateStats()
        stats.update(100.0)
        stats.apply_decay(0.9)  # 90.0
        stats.apply_decay(0.9)  # 81.0
        assert abs(stats.support - 81.0) < 1e-9

    def test_decay_after_update(self: Self) -> None:
        """Test decay applies to accumulated support."""
        stats = StateStats()
        stats.update(10.0)
        stats.apply_decay(0.8)
        stats.update(5.0)
        # 10.0 * 0.8 + 5.0 = 8.0 + 5.0 = 13.0
        assert abs(stats.support - 13.0) < 1e-9

    def test_decay_very_small_factor(self: Self) -> None:
        """Test decay with very small factor."""
        stats = StateStats()
        stats.update(1000.0)
        stats.apply_decay(0.001)
        assert abs(stats.support - 1.0) < 1e-6

    def test_decay_fractional_factor(self: Self) -> None:
        """Test decay with fractional factor between 0 and 1."""
        stats = StateStats()
        stats.update(7.0)
        stats.apply_decay(0.3)
        assert abs(stats.support - 2.1) < 1e-9


class TestStateStatsIsActive:
    """Tests for StateStats.is_active method."""

    def test_is_active_threshold_zero(self: Self) -> None:
        """Test is_active with zero threshold."""
        stats = StateStats()
        stats.update(1.0)
        assert stats.is_active(0.0)

    def test_is_active_below_threshold(self: Self) -> None:
        """Test is_active returns False when support below threshold."""
        stats = StateStats()
        stats.update(5.0)
        assert not stats.is_active(10.0)

    def test_is_active_at_threshold(self: Self) -> None:
        """Test is_active returns True when support equals threshold."""
        stats = StateStats()
        stats.update(10.0)
        assert stats.is_active(10.0)

    def test_is_active_above_threshold(self: Self) -> None:
        """Test is_active returns True when support above threshold."""
        stats = StateStats()
        stats.update(15.0)
        assert stats.is_active(10.0)

    def test_is_active_empty_stats(self: Self) -> None:
        """Test is_active with empty stats and positive threshold."""
        stats = StateStats()
        assert not stats.is_active(1.0)

    def test_is_active_negative_threshold(self: Self) -> None:
        """Test is_active with negative threshold."""
        stats = StateStats()
        stats.update(1.0)
        assert stats.is_active(-10.0)

    def test_is_active_high_threshold(self: Self) -> None:
        """Test is_active with very high threshold."""
        stats = StateStats()
        stats.update(100.0)
        assert stats.is_active(100.0)
        assert not stats.is_active(100.1)

    def test_is_active_after_decay(self: Self) -> None:
        """Test is_active adjusts after decay."""
        stats = StateStats()
        stats.update(10.0)
        assert stats.is_active(10.0)
        stats.apply_decay(0.5)
        assert stats.is_active(5.0)
        assert not stats.is_active(5.1)


class TestStateStatsConsistency:
    """Tests for consistency and invariants."""

    def test_support_non_negative_after_decay(self: Self) -> None:
        """Test support never becomes negative after decay."""
        stats = StateStats()
        stats.update(100.0)
        for _ in range(100):
            stats.apply_decay(0.99)
        assert stats.support >= 0.0

    def test_support_decreases_with_decay(self: Self) -> None:
        """Test support strictly decreases with decay < 1."""
        stats = StateStats()
        stats.update(100.0)
        prev = stats.support
        for _ in range(10):
            stats.apply_decay(0.9)
            curr = stats.support
            assert curr < prev
            prev = curr

    def test_support_constant_with_decay_1(self: Self) -> None:
        """Test support unchanged with decay factor 1."""
        stats = StateStats()
        stats.update(50.0)
        for _ in range(10):
            stats.apply_decay(1.0)
        assert stats.support == 50.0

    def test_update_then_decay_order_matters(self: Self) -> None:
        """Test that order of update and decay matters."""
        stats1 = StateStats()
        stats1.update(100.0)
        stats1.apply_decay(0.5)
        stats1.update(50.0)

        stats2 = StateStats()
        stats2.update(50.0)
        stats2.apply_decay(0.5)
        stats2.update(50.0)

        # Different order → different results
        assert stats1.support != stats2.support


class TestStateStatsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_accumulated_support(self: Self) -> None:
        """Test with very large accumulated support values."""
        stats = StateStats()
        for _ in range(1000):
            stats.update(1e6)
        assert stats.support == 1e9

    def test_alternating_update_and_decay(self: Self) -> None:
        """Test alternating updates and decays."""
        stats = StateStats()
        for _ in range(10):
            stats.update(10.0)
            stats.apply_decay(0.9)
        # After 10 iterations: roughly converges
        assert 50.0 < stats.support < 100.0

    def test_support_precision_with_many_small_updates(self: Self) -> None:
        """Test precision with many small updates."""
        stats = StateStats()
        for _ in range(1000):
            stats.update(0.001)
        assert abs(stats.support - 1.0) < 1e-6

    def test_decay_to_near_zero(self: Self) -> None:
        """Test approaching zero through repeated decay."""
        stats = StateStats()
        stats.update(1.0)
        for _ in range(100):
            stats.apply_decay(0.9)
        # Should be very small but positive
        assert stats.support >= 0.0
        assert stats.support < 1e-4

    def test_is_active_boundary_precision(self: Self) -> None:
        """Test is_active at boundary with floating point."""
        stats = StateStats()
        # Use value that may have floating point issues
        stats.update(0.1 + 0.2)  # = 0.30000...
        threshold = 0.3
        result = stats.is_active(threshold)
        # Should be True since 0.3 >= 0.3
        assert result is True


class TestStateStatsMultipleSequences:
    """Tests with multiple sequences of operations."""

    def test_sequence_1_update_decay_update(self: Self) -> None:
        """Test sequence: update, decay, update."""
        stats = StateStats()
        stats.update(10.0)
        stats.apply_decay(0.5)  # Now 5.0
        stats.update(5.0)  # Now 10.0
        assert stats.support == 10.0

    def test_sequence_2_multiple_decays(self: Self) -> None:
        """Test multiple decays."""
        stats = StateStats()
        stats.update(100.0)
        for decay_factor in [0.9, 0.8, 0.7, 0.6]:
            stats.apply_decay(decay_factor)
        # 100 * 0.9 * 0.8 * 0.7 * 0.6 = 30.24
        assert abs(stats.support - 30.24) < 0.01

    def test_sequence_3_decay_heavy_then_recover(self: Self) -> None:
        """Test heavy decay followed by recovery through updates."""
        stats = StateStats()
        stats.update(100.0)
        for _ in range(20):
            stats.apply_decay(0.5)
        # Very small now (roughly 100 * 0.5^20 ≈ 0)
        assert stats.support < 0.01

        # Recover with updates
        stats.update(50.0)
        assert stats.support > 49.0


class TestStateStatsSerialization:
    """
    Tests for StateStats JSON-serializable conversion helpers.

    Covers `to_dict` and `from_dict` to ensure state can be round-tripped.
    """

    def test_to_dict_returns_expected_structure(self: Self) -> None:
        """Ensure `to_dict` returns a mapping with the `support` value."""
        stats = StateStats()
        stats.update(2.75)
        data = stats.to_dict()
        assert isinstance(data, dict)
        assert data.get("support") == 2.75

    def test_from_dict_restores_support(self: Self) -> None:
        """Ensure `from_dict` reconstructs the `StateStats` support value."""
        data = {"support": 3.14}
        stats = StateStats.from_dict(data)
        assert abs(stats.support - 3.14) < 1e-12
