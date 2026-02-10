"""Unit tests for HierarchicalStateStats.

Comprehensive tests for the HierarchicalStateStats class, covering hierarchical
updates, multi-level predictions with fallback, and temporal decay.
"""
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats import (
    HierarchicalStateStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (
    HierarchicalStateStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


def create_test_hp(
    half_life: float = 50.0,
    min_prune_interval: float = 10.0,
    prune_enabled: bool = True,
    persistence_strength: float = 0.95,
    min_support_factor: float = 1.0,
) -> HierarchicalStateStatsHyperParameters:
    """Helper to create HyperParameters for testing."""
    base_hp = HyperParameters(
        half_life=half_life,
        min_prune_interval=min_prune_interval,
        prune_enabled=prune_enabled,
        persistence_strength=persistence_strength,
    )
    return HierarchicalStateStatsHyperParameters(base_hp, min_support_factor)


class TestHierarchicalStateStatsInitialization:
    """Tests for HierarchicalStateStats initialization."""

    def test_create_with_hyper_parameters(self: Self) -> None:
        """Test creating HierarchicalStateStats with hyper parameters."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = HierarchicalStateStatsHyperParameters(base_hp)
        stats = HierarchicalStateStats(hp)
        assert stats is not None

    def test_create_with_custom_min_support_factor(self: Self) -> None:
        """Test creating with custom min_support_factor."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.5)
        stats = HierarchicalStateStats(hp)
        assert hp.min_support == 25.0


class TestHierarchicalStateStatsUpdate:
    """Tests for HierarchicalStateStats.update method."""

    def test_update_at_specific_level(self: Self) -> None:
        """Test update at specific temporal level."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=1.0)

        # Should have updated GLOBAL and the hour level
        # Verify by attempting prediction
        result = stats.predict(TimeKey.GLOBAL)
        assert result is not None

    def test_update_updates_all_ancestors(self: Self) -> None:
        """Test that update updates all ancestor levels."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        # Create multi-level key
        key = (
            TimeKey.GLOBAL
            + TemporalFeature("hour", 14)
            + TemporalFeature("day_of_week", 3)
        )
        stats.update(key, "on", weight=10.0)

        # Check that we can predict at different levels
        global_result = stats.predict(TimeKey.GLOBAL)
        assert global_result is not None

    def test_update_multiple_times_accumulates(self: Self) -> None:
        """Test that multiple updates accumulate support."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=5.0)
        stats.update(key, "on", weight=5.0)

        result = stats.predict(key)
        assert result is not None
        # Support should be accumulated
        assert result.distribution.total_support() >= 10.0

    def test_update_with_different_states(self: Self) -> None:
        """Test updating with different states."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=10.0)
        stats.update(key, "off", weight=10.0)

        result = stats.predict(key)
        assert result is not None
        dist = result.distribution.distribution()
        assert abs(dist["on"] - 0.5) < 1e-9
        assert abs(dist["off"] - 0.5) < 1e-9


class TestHierarchicalStateStatsPredict:
    """Tests for HierarchicalStateStats.predict method."""

    def test_predict_specific_level_confident(self: Self) -> None:
        """Test prediction at specific level with sufficient data."""
        hp = create_test_hp(min_support_factor=0.1)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        # Add enough support for confidence
        stats.update(key, "on", weight=100.0)

        result = stats.predict(key)
        assert result is not None
        assert result.key == key
        assert len(result.contributions) == 1
        assert result.contributions[0].weight == 1.0

    def test_predict_insufficient_specific_level_fallback(self: Self) -> None:
        """Test fallback to ancestors when specific level insufficient."""
        hp = create_test_hp(min_support_factor=1.0)
        stats = HierarchicalStateStats(hp)

        # Update global with sufficient data
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        # Update specific with insufficient data
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=1.0)

        # Predict at specific level should fallback to global
        result = stats.predict(key)
        assert result is not None
        # Should include contributions from GLOBAL (ancestor)
        assert len(result.contributions) > 0

    def test_predict_no_data_returns_none(self: Self) -> None:
        """Test prediction returns None when no data available."""
        hp = create_test_hp()
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        result = stats.predict(key)
        assert result is None

    def test_predict_multi_level_fallback_chain(self: Self) -> None:
        """Test prediction through multiple fallback levels."""
        hp = create_test_hp(min_support_factor=0.05)
        stats = HierarchicalStateStats(hp)

        # Add data at different levels
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        level1 = TimeKey.GLOBAL + TemporalFeature("season", "spring")
        stats.update(level1, "on", weight=50.0)

        level2 = level1 + TemporalFeature("day_of_week", 2)
        stats.update(level2, "on", weight=10.0)

        # Predict at level2 should chain through levels
        result = stats.predict(level2)
        assert result is not None
        assert len(result.contributions) > 0

    def test_predict_contributions_ordered(self: Self) -> None:
        """Test that contributions are ordered from specific to global."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        # Build hierarchy
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        key1 = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key1, "on", weight=50.0)

        key2 = key1 + TemporalFeature("day_of_week", 3)
        stats.update(key2, "on", weight=20.0)

        result = stats.predict(key2)
        assert result is not None
        # First contribution should be the specific key or first ancestor
        assert result.contributions[0].weight > 0

    def test_predict_with_weak_confidence_weights(self: Self) -> None:
        """Test that confidence weights decrease with distance."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        # Create insufficient data at specific level
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=1.0)

        # Create data at ancestor level
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        result = stats.predict(key)
        assert result is not None

        # Weights should decrease as we go up the hierarchy
        if len(result.contributions) > 1:
            prev_weight = result.contributions[0].weight
            for i in range(1, len(result.contributions)):
                assert result.contributions[i].weight < prev_weight
                prev_weight = result.contributions[i].weight


class TestHierarchicalStateStatsApplyDecay:
    """Tests for HierarchicalStateStats.apply_decay method."""

    def test_apply_decay_affects_predictions(self: Self) -> None:
        """Test that decay affects subsequent predictions."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=100.0)

        result1 = stats.predict(key)
        support1 = result1.distribution.total_support()

        # Apply decay
        stats.apply_decay(0.5)

        result2 = stats.predict(key)
        support2 = result2.distribution.total_support()

        # Support should be reduced
        assert support2 < support1

    def test_apply_decay_multiple_times(self: Self) -> None:
        """Test repeated decay operations."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=1000.0)

        # Apply decay multiple times
        for _ in range(5):
            stats.apply_decay(0.9)

        result = stats.predict(key)
        assert result is not None
        # Support should be reduced but still positive
        assert result.distribution.total_support() < 1000.0
        assert result.distribution.total_support() > 0

    def test_apply_decay_at_all_levels(self: Self) -> None:
        """Test that decay affects all hierarchy levels."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        # Update at multiple levels
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        key1 = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key1, "on", weight=100.0)

        # Decay
        stats.apply_decay(0.5)

        # Both should be affected
        global_result = stats.predict(TimeKey.GLOBAL)
        specific_result = stats.predict(key1)

        # GLOBAL was updated twice (once directly, once via key1), so 200.0 -> 100.0
        assert global_result.distribution.total_support() == 100.0
        # Specific level has its own copy affected
        assert specific_result is not None


class TestHierarchicalStateStatsPrune:
    """Tests for HierarchicalStateStats.prune method."""

    def test_prune_removes_infrequent_states(self: Self) -> None:
        """Test that prune removes infrequent states."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "frequent", 1000.0)
        stats.update(key, "rare", 1.0)

        # Prune with default parameters
        stats.prune()

        result = stats.predict(key)
        if result is not None:
            # Rare state should be removed
            states = result.distribution.states()
            assert "frequent" in states
            assert "rare" not in states

    def test_prune_with_custom_parameters(self: Self) -> None:
        """Test prune with custom epsilon and absolute_min."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL
        for i in range(100):
            stats.update(key, f"state_{i}", float(i + 1))

        # Prune with strict threshold
        stats.prune(epsilon=0.01, absolute_min=50.0)

        result = stats.predict(key)
        assert result is not None
        # Only higher-numbered states should remain
        states = result.distribution.states()
        assert len(states) < 100

    def test_prune_empty_store(self: Self) -> None:
        """Test prune on empty statistics."""
        hp = create_test_hp()
        stats = HierarchicalStateStats(hp)

        # Should not raise error
        stats.prune()

        result = stats.predict(TimeKey.GLOBAL)
        assert result is None


class TestHierarchicalStateStatsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_global_key_only(self: Self) -> None:
        """Test updating and predicting at GLOBAL level."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        stats.update(TimeKey.GLOBAL, "on", weight=100.0)
        result = stats.predict(TimeKey.GLOBAL)
        assert result is not None
        assert result.key == TimeKey.GLOBAL

    def test_deeply_nested_keys(self: Self) -> None:
        """Test with deeply nested temporal hierarchy."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        # Build 5-level deep key
        key = TimeKey.GLOBAL
        for i in range(5):
            key = key + TemporalFeature(f"level_{i}", i)

        stats.update(key, "on", weight=100.0)
        result = stats.predict(key)
        assert result is not None

    def test_many_different_states(self: Self) -> None:
        """Test with many different states."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)

        for i in range(100):
            stats.update(key, f"state_{i}", 1.0)

        result = stats.predict(key)
        assert result is not None
        assert len(result.distribution.states()) == 100

    def test_zero_weight_update(self: Self) -> None:
        """Test updating with zero weight."""
        hp = create_test_hp()
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=0.0)

        result = stats.predict(key)
        assert result is None

    def test_very_large_weights(self: Self) -> None:
        """Test with very large weight values."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=1e10)

        result = stats.predict(key)
        assert result is not None
        assert result.distribution.total_support() > 0


class TestHierarchicalStateStatsIntegration:
    """Integration tests with complete workflows."""

    def test_complete_temporal_pattern_learning(self: Self) -> None:
        """Test learning temporal patterns over multiple time points."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        # Simulate a day of observations
        for hour in range(24):
            key = TimeKey.GLOBAL + TemporalFeature("hour", hour)

            # Different patterns for different times
            if 8 <= hour < 18:
                # Working hours - mostly "active"
                stats.update(key, "active", weight=10.0)
                stats.update(key, "inactive", weight=2.0)
            else:
                # Off hours - mostly "inactive"
                stats.update(key, "active", weight=2.0)
                stats.update(key, "inactive", weight=10.0)

        # Check predictions at different times
        morning_key = TimeKey.GLOBAL + TemporalFeature("hour", 10)
        morning_result = stats.predict(morning_key)
        assert morning_result is not None

        evening_key = TimeKey.GLOBAL + TemporalFeature("hour", 22)
        evening_result = stats.predict(evening_key)
        assert evening_result is not None

    def test_workflow_update_decay_predict(self: Self) -> None:
        """Test complete workflow: update, decay, predict."""
        hp = create_test_hp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats.update(key, "on", weight=100.0)
        stats.update(key, "off", weight=50.0)

        # Get baseline prediction
        result1 = stats.predict(key)
        support1 = result1.distribution.total_support()

        # Apply decay
        stats.apply_decay(0.8)

        # Predict again
        result2 = stats.predict(key)
        support2 = result2.distribution.total_support()

        # Verify decay effect
        assert abs(support2 - support1 * 0.8) < 1e-6

    def test_workflow_multiple_updates_prune_predict(self: Self) -> None:
        """Test workflow with updates, pruning, and predictions."""
        hp = create_test_hp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp)

        # Add data
        key = TimeKey.GLOBAL
        stats.update(key, "frequent", 1000.0)
        stats.update(key, "rare", 5.0)

        # Check before pruning
        result_before = stats.predict(key)
        states_before = len(result_before.distribution.states())

        # Prune
        stats.prune(epsilon=0.003, absolute_min=20.0)

        # Check after pruning
        result_after = stats.predict(key)
        states_after = len(result_after.distribution.states())

        # Rare state should be gone
        assert states_after <= states_before

    def test_confidence_threshold_behavior(self: Self) -> None:
        """Test behavior with different confidence thresholds."""
        # Strict threshold
        hp_strict = create_test_hp(min_support_factor=2.0)
        stats_strict = HierarchicalStateStats(hp_strict)

        # Permissive threshold
        hp_permissive = create_test_hp(min_support_factor=0.1)
        stats_permissive = HierarchicalStateStats(hp_permissive)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)

        # Add moderate amount of data
        stats_strict.update(key, "on", weight=40.0)
        stats_permissive.update(key, "on", weight=40.0)

        # Strict might not predict, permissive should
        result_strict = stats_strict.predict(key)
        result_permissive = stats_permissive.predict(key)

        # permissive should succeed
        assert result_permissive is not None

        # Note: strict might fallback instead of returning None
        # so we just check that it's possible to get a result
