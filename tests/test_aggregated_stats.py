"""Unit tests for AggregatedStats.

Comprehensive tests for the AggregatedStats class, covering TimeKey association,
distribution conversion, and inherited functionality.
"""
from typing import Self

from custom_components.discrete_state_forecaster.model.statistics.aggregated_stats import (
    AggregatedStats,
)
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestAggregatedStatsInitialization:
    """Tests for AggregatedStats initialization."""

    def test_create_with_global_key(self: Self) -> None:
        """Test creating AggregatedStats with GLOBAL key."""
        stats = AggregatedStats(TimeKey.GLOBAL)
        assert stats.key == TimeKey.GLOBAL
        assert stats.is_empty()

    def test_create_with_single_feature_key(self: Self) -> None:
        """Test creating AggregatedStats with single-feature key."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        assert stats.key == key
        assert stats.is_empty()

    def test_create_with_multi_feature_key(self: Self) -> None:
        """Test creating AggregatedStats with multi-feature key."""
        key = (
            TimeKey.GLOBAL
            + TemporalFeature("hour", 14)
            + TemporalFeature("day_of_week", 3)
        )
        stats = AggregatedStats(key)
        assert stats.key == key
        assert len(stats.key) == 2

    def test_key_is_immutable(self: Self) -> None:
        """Test that key attribute is immutable."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        # Check that Final[TimeKey] is properly typed
        assert stats.key == key
        # Verify key doesn't change
        original_key = stats.key
        # Operations on stats shouldn't change the key
        stats.update("on", 5.0)
        assert stats.key == original_key

    def test_initial_support_is_zero(self: Self) -> None:
        """Test initial total support is zero."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        assert stats.total_support() == 0.0


class TestAggregatedStatsInheritedMethods:
    """Tests for inherited DistributionStats methods."""

    def test_update_works(self: Self) -> None:
        """Test that inherited update method works."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        stats.update("on", 5.0)
        assert stats.support("on") == 5.0

    def test_distribution_works(self: Self) -> None:
        """Test that inherited distribution method works."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        stats.update("on", 2.0)
        stats.update("off", 1.0)
        dist = stats.distribution()
        assert abs(dist["on"] - 2/3) < 1e-9
        assert abs(dist["off"] - 1/3) < 1e-9

    def test_entropy_works(self: Self) -> None:
        """Test that inherited entropy method works."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        stats.update("on", 1.0)
        stats.update("off", 1.0)
        entropy = stats.entropy()
        assert entropy > 0.0

    def test_apply_decay_works(self: Self) -> None:
        """Test that inherited apply_decay method works."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        stats.update("on", 10.0)
        stats.apply_decay(0.5)
        assert stats.support("on") == 5.0

    def test_prune_works(self: Self) -> None:
        """Test that inherited prune method works."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        stats.update("a", 5.0)
        stats.update("b", 15.0)
        stats.prune(10.0)
        assert stats.states() == {"b"}


class TestAggregatedStatsFromDistribution:
    """Tests for AggregatedStats.from_distribution class method."""

    def test_from_distribution_single_state(self: Self) -> None:
        """Test conversion from single-state distribution."""
        dist = DistributionStats()
        dist.update("on", 10.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert agg.key == key
        assert agg.support("on") == 10.0

    def test_from_distribution_multiple_states(self: Self) -> None:
        """Test conversion from multi-state distribution."""
        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("off", 1.0)
        dist.update("unknown", 1.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        # Total support should be preserved
        assert agg.total_support() == 4.0
        # Probabilities should be preserved
        dist_orig = dist.distribution()
        dist_conv = agg.distribution()
        for state in ["on", "off", "unknown"]:
            assert abs(dist_orig[state] - dist_conv[state]) < 1e-9

    def test_from_distribution_empty(self: Self) -> None:
        """Test conversion from empty distribution."""
        dist = DistributionStats()
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert agg.key == key
        assert agg.is_empty()

    def test_from_distribution_preserves_distribution(self: Self) -> None:
        """Test that conversion preserves the probability distribution."""
        dist = DistributionStats()
        dist.update("a", 10.0)
        dist.update("b", 20.0)
        dist.update("c", 30.0)

        key = TimeKey.from_tuple((("hour", 10),))
        agg = AggregatedStats.from_distribution(dist, key)

        # Check that probabilities (not absolute support) are preserved
        original_dist = dist.distribution()
        converted_dist = agg.distribution()

        for state in ["a", "b", "c"]:
            assert abs(original_dist[state] - converted_dist[state]) < 1e-9

    def test_from_distribution_with_complex_key(self: Self) -> None:
        """Test conversion with complex multi-level key."""
        dist = DistributionStats()
        dist.update("on", 100.0)
        dist.update("off", 50.0)

        key = (
            TimeKey.GLOBAL
            + TemporalFeature("season", "spring")
            + TemporalFeature("day_of_week", 2)
            + TemporalFeature("hour", 14)
        )
        agg = AggregatedStats.from_distribution(dist, key)

        assert agg.key == key
        assert len(agg.key) == 3

    def test_from_distribution_returns_aggregated_stats(self: Self) -> None:
        """Test that from_distribution returns AggregatedStats instance."""
        dist = DistributionStats()
        dist.update("on", 5.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert isinstance(agg, AggregatedStats)

    def test_from_distribution_fractional_support(self: Self) -> None:
        """Test conversion with fractional support values."""
        dist = DistributionStats()
        dist.update("a", 100.0)
        dist.update("b", 50.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        # Total should be 150
        assert abs(agg.total_support() - 150.0) < 1e-6

        # Probabilities should be 2/3 and 1/3
        dist_dict = agg.distribution()
        assert abs(dist_dict["a"] - 2/3) < 1e-9
        assert abs(dist_dict["b"] - 1/3) < 1e-9


class TestAggregatedStatsKeyAssociation:
    """Tests for key association and uniqueness."""

    def test_different_keys_create_different_instances(self: Self) -> None:
        """Test that different keys create distinct instances."""
        key1 = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        key2 = TimeKey.GLOBAL + TemporalFeature("hour", 15)

        stats1 = AggregatedStats(key1)
        stats2 = AggregatedStats(key2)

        assert stats1.key != stats2.key
        assert stats1.key == key1
        assert stats2.key == key2

    def test_same_key_creates_equal_stats(self: Self) -> None:
        """Test that same key creates equal key objects."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats1 = AggregatedStats(key)
        stats2 = AggregatedStats(key)

        assert stats1.key == stats2.key

    def test_global_key_aggregated_stats(self: Self) -> None:
        """Test AggregatedStats with TimeKey.GLOBAL."""
        stats = AggregatedStats(TimeKey.GLOBAL)
        assert stats.key.is_root
        assert len(stats.key) == 0


class TestAggregatedStatsComparison:
    """Tests for comparing aggregated stats."""

    def test_aggregated_stats_with_different_keys_have_same_support_count(self: Self) -> None:
        """Test that different keys can store same states."""
        key1 = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        key2 = TimeKey.GLOBAL + TemporalFeature("hour", 15)

        dist = DistributionStats()
        dist.update("on", 10.0)
        dist.update("off", 5.0)

        agg1 = AggregatedStats.from_distribution(dist, key1)
        agg2 = AggregatedStats.from_distribution(dist, key2)

        # Same distribution but different keys
        assert agg1.key != agg2.key
        assert agg1.total_support() == agg2.total_support()
        assert agg1.states() == agg2.states()


class TestAggregatedStatsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_aggregated_stats_with_zero_total_support(self: Self) -> None:
        """Test AggregatedStats created from zero-support distribution."""
        dist = DistributionStats()
        dist.update("a", 0.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert agg.total_support() == 0.0
        assert agg.distribution() == {}

    def test_aggregated_stats_large_support_values(self: Self) -> None:
        """Test with very large support values."""
        dist = DistributionStats()
        dist.update("a", 1e10)
        dist.update("b", 1e10)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert agg.total_support() == 2e10

    def test_aggregated_stats_many_states(self: Self) -> None:
        """Test with many different states."""
        dist = DistributionStats()
        for i in range(100):
            dist.update(f"state_{i}", 1.0)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        assert len(agg.states()) == 100
        assert agg.total_support() == 100.0

    def test_aggregated_stats_preserves_order(self: Self) -> None:
        """Test that from_distribution preserves state information correctly."""
        dist = DistributionStats()
        states_and_weights = [
            ("on", 100.0),
            ("off", 50.0),
            ("unknown", 25.0),
            ("error", 10.0),
        ]

        for state, weight in states_and_weights:
            dist.update(state, weight)

        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        agg = AggregatedStats.from_distribution(dist, key)

        # Check all states are present
        for state, weight in states_and_weights:
            assert state in agg.states()
            # Support should be preserved
            assert agg.support(state) == weight


class TestAggregatedStatsInheritanceChain:
    """Tests for proper inheritance from DistributionStats."""

    def test_is_instance_of_distribution_stats(self: Self) -> None:
        """Test that AggregatedStats is instance of DistributionStats."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)
        assert isinstance(stats, DistributionStats)

    def test_has_all_parent_methods(self: Self) -> None:
        """Test that all parent methods are accessible."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)

        # Check key parent methods exist
        assert hasattr(stats, "update")
        assert hasattr(stats, "total_support")
        assert hasattr(stats, "support")
        assert hasattr(stats, "distribution")
        assert hasattr(stats, "is_confident")
        assert hasattr(stats, "active_states")
        assert hasattr(stats, "entropy")
        assert hasattr(stats, "max_probability")
        assert hasattr(stats, "apply_decay")
        assert hasattr(stats, "states")
        assert hasattr(stats, "is_empty")
        assert hasattr(stats, "prune")
        assert hasattr(stats, "prune_adaptive")

    def test_parent_method_functionality(self: Self) -> None:
        """Test that parent methods work correctly through inheritance."""
        key = TimeKey.GLOBAL + TemporalFeature("hour", 14)
        stats = AggregatedStats(key)

        # Use inherited methods
        stats.update("on", 10.0)
        stats.update("off", 10.0)

        # All these should work as in parent class
        assert stats.total_support() == 20.0
        assert stats.is_confident(15.0)
        assert len(stats.active_states(5.0)) == 2
        assert stats.entropy() > 0.0
        assert stats.max_probability() == 0.5
