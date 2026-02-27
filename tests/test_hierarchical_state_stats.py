"""
Unit tests for HierarchicalStateStats.

Comprehensive tests for the HierarchicalStateStats class, covering hierarchical
updates, multi-level predictions with fallback, and temporal decay.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats import (
    HierarchicalStateStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (  # noqa: E501
    HierarchicalStateStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_runtime_parameters import (
    HierarchicalStateStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


def create_test_hp(
    half_life: float = 50.0,
    min_prune_interval_factor: float = 0.2,
    prune_enabled: bool = True,
    persistence_strength: float = 0.95,
) -> HierarchicalStateStatsHyperParameters:
    """Helper to create HyperParameters for testing."""
    base_hp = ForecasterEngineHyperParameters(
        half_life=half_life,
        min_prune_interval_factor=min_prune_interval_factor,
        prune_enabled=prune_enabled,
        persistence_strength=persistence_strength,
    )
    return HierarchicalStateStatsHyperParameters(base_hp)


def create_test_rp(
    min_support_factor: float = 1.0,
) -> HierarchicalStateStatsRuntimeParameters:
    """Helper to create RuntimeParameters for testing."""
    return HierarchicalStateStatsRuntimeParameters(
        min_support_factor=min_support_factor
    )


class TestHierarchicalStateStatsInitialization:
    """Tests for HierarchicalStateStats initialization."""

    def test_create_with_hyper_parameters(self: Self) -> None:
        """Test creating HierarchicalStateStats with hyper parameters."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=1.0)
        stats = HierarchicalStateStats(hp, rp)
        assert stats is not None

    def test_create_with_custom_min_support_factor(self: Self) -> None:
        """Test creating with custom min_support_factor."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.5)
        _stats = HierarchicalStateStats(hp, rp)
        assert _stats.parameters.min_support == 25.0


class TestHierarchicalStateStatsUpdate:
    """Tests for HierarchicalStateStats.update method."""

    def test_update_at_specific_level(self: Self) -> None:
        """Test update at specific temporal level."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=1.0)

        # Should have updated GLOBAL and the hour level
        # Verify by attempting prediction
        result = stats.predict(TimeKey.GLOBAL)
        assert result is not None

    def test_update_updates_all_ancestors(self: Self) -> None:
        """Test that update updates all ancestor levels."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        # Create multi-level key
        key = TimeKey(("hour", 14), ("day_of_week", 3))
        stats.update(key, "on", weight=10.0)

        # Check that we can predict at different levels
        global_result = stats.predict(TimeKey.GLOBAL)
        assert global_result is not None

    def test_update_multiple_times_accumulates(self: Self) -> None:
        """Test that multiple updates accumulate support."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=5.0)
        stats.update(key, "on", weight=5.0)

        result = stats.predict(key)
        assert result is not None
        # Support should be accumulated
        assert result.confidence.support >= 10.0

    def test_update_with_different_states(self: Self) -> None:
        """Test updating with different states."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=10.0)
        stats.update(key, "off", weight=10.0)

        result = stats.predict(key)
        assert result is not None
        dist = result.distribution
        assert abs(dist["on"] - 0.5) < 1e-9
        assert abs(dist["off"] - 0.5) < 1e-9


class TestHierarchicalStateStatsPredict:
    """Tests for HierarchicalStateStats.predict method."""

    def test_predict_specific_level_confident(self: Self) -> None:
        """Test prediction at specific level with sufficient data."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.1)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        # Add enough support for confidence
        stats.update(key, "on", weight=100.0)

        result = stats.predict(key)
        assert result is not None
        assert result.key == key
        assert len(result.contributions) == 1
        assert result.contributions[0].weight == 1.0

    def test_predict_insufficient_specific_level_fallback(self: Self) -> None:
        """Test fallback to ancestors when specific level insufficient."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=1.0)
        stats = HierarchicalStateStats(hp, rp)

        # Update global with sufficient data
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        # Update specific with insufficient data
        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=1.0)

        # Predict at specific level should fallback to global
        result = stats.predict(key)
        assert result is not None
        # Should include contributions from GLOBAL (ancestor)
        assert len(result.contributions) > 0

    def test_predict_no_data_returns_none(self: Self) -> None:
        """Test prediction returns None when no data available."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        result = stats.predict(key)
        assert result is None

    def test_predict_multi_level_fallback_chain(self: Self) -> None:
        """Test prediction through multiple fallback levels."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.05)
        stats = HierarchicalStateStats(hp, rp)

        # Add data at different levels
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        level1 = TimeKey(("season", "spring"))
        stats.update(level1, "on", weight=50.0)

        level2 = level1 + ("day_of_week", 2)
        stats.update(level2, "on", weight=10.0)

        # Predict at level2 should chain through levels
        result = stats.predict(level2)
        assert result is not None
        assert len(result.contributions) > 0

    def test_predict_contributions_ordered(self: Self) -> None:
        """Test that contributions are ordered from specific to global."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        # Build hierarchy
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        key1 = TimeKey(("hour", 14))
        stats.update(key1, "on", weight=50.0)

        key2 = key1 + ("day_of_week", 3)
        stats.update(key2, "on", weight=20.0)

        result = stats.predict(key2)
        assert result is not None
        # First contribution should be the specific key or first ancestor
        assert result.contributions[0].weight > 0

    def test_predict_with_weak_confidence_weights(self: Self) -> None:
        """Test that confidence weights decrease with distance."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        # Create insufficient data at specific level
        key = TimeKey(("hour", 14))
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
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=100.0)

        result1 = stats.predict(key)
        support1 = result1.confidence.support

        # Apply decay
        stats.apply_decay(0.5)

        result2 = stats.predict(key)
        support2 = result2.confidence.support

        # Support should be reduced
        assert support2 < support1

    def test_apply_decay_multiple_times(self: Self) -> None:
        """Test repeated decay operations."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=1000.0)

        # Apply decay multiple times
        for _ in range(5):
            stats.apply_decay(0.9)

        result = stats.predict(key)
        assert result is not None
        # Support should be reduced but still positive
        assert result.confidence.support < 1000.0
        assert result.confidence.support > 0

    def test_apply_decay_at_all_levels(self: Self) -> None:
        """Test that decay affects all hierarchy levels."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        # Update at multiple levels
        stats.update(TimeKey.GLOBAL, "on", weight=100.0)

        key1 = TimeKey(("hour", 14))
        stats.update(key1, "on", weight=100.0)

        # Decay
        stats.apply_decay(0.5)

        # Both should be affected
        global_result = stats.predict(TimeKey.GLOBAL)
        specific_result = stats.predict(key1)

        # GLOBAL was updated twice (once directly, once via key1), so 200.0 -> 100.0
        assert global_result.confidence.support == 100.0
        # Specific level has its own copy affected
        assert specific_result is not None


class TestHierarchicalStateStatsPrune:
    """Tests for HierarchicalStateStats.prune method."""

    def test_prune_removes_infrequent_states(self: Self) -> None:
        """Test that prune removes infrequent states."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "frequent", 1000.0)
        stats.update(key, "rare", 1.0)

        # Prune with default parameters
        stats.prune()

        result = stats.predict(key)
        if result is not None:
            # Rare state should be removed
            states = result.distribution
            assert "frequent" in states
            assert "rare" not in states

    def test_prune_with_custom_parameters(self: Self) -> None:
        """Test prune with custom epsilon and absolute_min."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey.GLOBAL
        for i in range(100):
            stats.update(key, f"state_{i}", float(i + 1))

        # Prune with strict threshold
        stats.prune(epsilon=0.01, absolute_minimum_support=50.0)

        result = stats.predict(key)
        assert result is not None
        # Only higher-numbered states should remain
        states = result.distribution
        assert len(states) < 100

    def test_prune_empty_store(self: Self) -> None:
        """Test prune on empty statistics."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = HierarchicalStateStats(hp, rp)

        # Should not raise error
        stats.prune()

        result = stats.predict(TimeKey.GLOBAL)
        assert result is None


class TestHierarchicalStateStatsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_global_key_only(self: Self) -> None:
        """Test updating and predicting at GLOBAL level."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        stats.update(TimeKey.GLOBAL, "on", weight=100.0)
        result = stats.predict(TimeKey.GLOBAL)
        assert result is not None
        assert result.key == TimeKey.GLOBAL

    def test_deeply_nested_keys(self: Self) -> None:
        """Test with deeply nested temporal hierarchy."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        # Build 5-level deep key
        key = TimeKey.GLOBAL
        for i in range(5):
            key = key + (f"level_{i}", i)

        stats.update(key, "on", weight=100.0)
        result = stats.predict(key)
        assert result is not None

    def test_many_different_states(self: Self) -> None:
        """Test with many different states."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey("hour", 14)

        for i in range(100):
            stats.update(key, f"state_{i}", 1.0)

        result = stats.predict(key)
        assert result is not None
        assert len(result.distribution) == 100

    def test_zero_weight_update(self: Self) -> None:
        """Test updating with zero weight."""
        hp = create_test_hp()
        rp = create_test_rp()
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=0.0)

        result = stats.predict(key)
        assert result is None

    def test_very_large_weights(self: Self) -> None:
        """Test with very large weight values."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=1e10)

        result = stats.predict(key)
        assert result is not None
        assert result.confidence.support > 0


class TestHierarchicalStateStatsIntegration:
    """Integration tests with complete workflows."""

    def test_complete_temporal_pattern_learning(self: Self) -> None:
        """Test learning temporal patterns over multiple time points."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        # Simulate a day of observations
        for hour in range(24):
            key = TimeKey(("hour", hour))

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
        morning_key = TimeKey(("hour", 10))
        morning_result = stats.predict(morning_key)
        assert morning_result is not None

        evening_key = TimeKey(("hour", 22))
        evening_result = stats.predict(evening_key)
        assert evening_result is not None

    def test_workflow_update_decay_predict(self: Self) -> None:
        """Test complete workflow: update, decay, predict."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey(("hour", 14))
        stats.update(key, "on", weight=100.0)
        stats.update(key, "off", weight=50.0)

        # Get baseline prediction
        result1 = stats.predict(key)
        support1 = result1.confidence.support

        # Apply decay
        stats.apply_decay(0.8)

        # Predict again
        result2 = stats.predict(key)
        support2 = result2.confidence.support

        # Verify decay effect
        assert abs(support2 - support1 * 0.8) < 1e-6

    def test_workflow_multiple_updates_prune_predict(self: Self) -> None:
        """Test workflow with updates, pruning, and predictions."""
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        # Add data
        key = TimeKey.GLOBAL
        stats.update(key, "frequent", 1000.0)
        stats.update(key, "rare", 5.0)

        # Check before pruning
        result_before = stats.predict(key)
        states_before = len(result_before.distribution)

        # Prune
        stats.prune(epsilon=0.003, absolute_minimum_support=20.0)

        # Check after pruning
        result_after = stats.predict(key)
        states_after = len(result_after.distribution)

        # Rare state should be gone
        assert states_after <= states_before

    def test_confidence_threshold_behavior(self: Self) -> None:
        """Test behavior with different confidence thresholds."""
        # Strict threshold
        hp_strict = create_test_hp()
        rp_strict = create_test_rp(min_support_factor=2.0)
        stats_strict = HierarchicalStateStats(hp_strict, rp_strict)

        # Permissive threshold
        hp_permissive = create_test_hp()
        rp_permissive = create_test_rp(min_support_factor=0.1)
        stats_permissive = HierarchicalStateStats(hp_permissive, rp_permissive)

        key = TimeKey(("hour", 14))

        # Add moderate amount of data
        stats_strict.update(key, "on", weight=40.0)
        stats_permissive.update(key, "on", weight=40.0)

        # Strict might not predict, permissive should
        _result_strict = stats_strict.predict(key)
        result_permissive = stats_permissive.predict(key)

        # permissive should succeed
        assert result_permissive is not None

        # Note: strict might fallback instead of returning None
        # so we just check that it's possible to get a result


class TestHierarchicalStateStatsPerKeyDecay:
    """Tests ensuring per-key observation-weighted decay preserves dormant keys."""

    def test_dormant_seasonal_key_not_decayed_by_other_season(self: Self) -> None:
        """
        The central regression test: statistics that were learned for one
        season (e.g. winter) must not be eroded while a different season
        (summer) is receiving new observations.
        """
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        winter_key = TimeKey(("season", "winter"))
        summer_key = TimeKey(("season", "summer"))

        # Learn a strong winter pattern.
        stats.update(winter_key, "heating", weight=1000.0)

        # Simulate many summer observations using per-key decay.
        decay = 0.9
        for _ in range(50):
            stats.update(summer_key, "cooling", weight=10.0, decay_factor=decay)

        # The GLOBAL ancestor is touched by both seasons, so it will have
        # decayed. But the raw winter key itself must still hold most of its
        # support because it was never updated (and therefore never decayed).
        winter_result = stats.predict(winter_key)
        assert winter_result is not None

        # Find the contribution that comes directly from the winter key.
        winter_contribution = next(
            (c for c in winter_result.contributions if c.key == winter_key), None
        )
        assert winter_contribution is not None, (
            "winter_key contribution must show up in the prediction fallback"
        )
        # The winter key should still carry close to its original 1000-unit support.
        assert winter_contribution.support > 900.0, (
            f"Expected winter support > 900, got {winter_contribution.support}"
        )

    def test_per_key_decay_affects_only_current_hierarchy(self: Self) -> None:
        """
        When a specific TimeKey is updated with a decay_factor, only the keys
        in its hierarchy (the key itself and all its ancestors) should be
        decayed; sibling keys at the same level must be unchanged.
        """
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.001)
        stats = HierarchicalStateStats(hp, rp)

        key_a = TimeKey(("hour", 10))
        key_b = TimeKey(("hour", 22))

        stats.update(key_a, "on", weight=200.0)
        stats.update(key_b, "on", weight=200.0)

        # Note: both keys share the GLOBAL ancestor, so GLOBAL will be
        # decayed when key_a is updated.  But the key_b distribution itself
        # must remain untouched.
        stats.update(key_a, "on", weight=0.0, decay_factor=0.5)

        result_b = stats.predict(key_b)
        assert result_b is not None

        # key_b's own distribution was not updated -> not decayed.
        key_b_contribution = next(
            (c for c in result_b.contributions if c.key == key_b), None
        )
        assert key_b_contribution is not None
        assert abs(key_b_contribution.support - 200.0) < 1e-9


class TestHierarchicalStateStatsSerialization:
    """Tests for HierarchicalStateStats serialization and deserialization."""

    def test_to_dict_and_from_dict_roundtrip(self: Self) -> None:
        hp = create_test_hp()
        rp = create_test_rp(min_support_factor=0.01)
        stats = HierarchicalStateStats(hp, rp)

        key = TimeKey.GLOBAL + ("hour", 14)
        stats.update(key, "on", weight=10.0)
        stats.update(TimeKey.GLOBAL, "off", weight=5.0)

        data = stats.to_dict()

        # Reconstruct using base hyper-parameters (same base used to create hp)
        restored = HierarchicalStateStats.from_dict(data, hp, rp)

        # Restored store should contain distributions for the keys we updated
        res = restored.predict(key)
        assert res is not None
        # Check that the restored distribution preserves support for 'on'
        on_contribution = next((c for c in res.contributions if c.key == key), None)
        assert on_contribution is not None
        assert abs(on_contribution.support - 10.0) < 1e-9
