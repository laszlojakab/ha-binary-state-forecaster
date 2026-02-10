"""Unit tests for KeyedDistributionStore.

Comprehensive tests for the KeyedDistributionStore class, covering key-based
distribution management, aggregation, and bulk operations.
"""
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.statistics.keyed_distribution_store import (
    KeyedDistributionStore,
)


class TestKeyedDistributionStoreInitialization:
    """Tests for KeyedDistributionStore initialization."""

    def test_create_empty_store(self: Self) -> None:
        """Test creating an empty store."""
        store = KeyedDistributionStore()
        assert store.get_distribution("any_key") is None

    def test_initial_state(self: Self) -> None:
        """Test that initial store is empty."""
        store = KeyedDistributionStore()
        # Store should be empty initially
        assert store.get_distribution("key1") is None


class TestKeyedDistributionStoreUpdate:
    """Tests for KeyedDistributionStore.update method."""

    def test_update_single_key_single_state(self: Self) -> None:
        """Test updating single state in single key."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 5.0)
        dist = store.get_distribution("key1")
        assert dist is not None
        assert dist.support("on") == 5.0

    def test_update_single_key_multiple_states(self: Self) -> None:
        """Test updating multiple states in same key."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 2.0)
        store.update("key1", "off", 3.0)
        dist = store.get_distribution("key1")
        assert dist.total_support() == 5.0

    def test_update_multiple_keys(self: Self) -> None:
        """Test updating multiple different keys."""
        store = KeyedDistributionStore()
        store.update("morning", "on", 10.0)
        store.update("evening", "on", 5.0)

        morning = store.get_distribution("morning")
        evening = store.get_distribution("evening")

        assert morning.total_support() == 10.0
        assert evening.total_support() == 5.0

    def test_update_default_weight(self: Self) -> None:
        """Test update with default weight 1.0."""
        store = KeyedDistributionStore()
        store.update("key1", "state1")
        dist = store.get_distribution("key1")
        assert dist.support("state1") == 1.0

    def test_update_accumulates_in_key(self: Self) -> None:
        """Test that updates to same key accumulate."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 2.0)
        store.update("key1", "on", 3.0)
        dist = store.get_distribution("key1")
        assert dist.support("on") == 5.0


class TestKeyedDistributionStoreGetDistribution:
    """Tests for KeyedDistributionStore.get_distribution method."""

    def test_get_existing_distribution(self: Self) -> None:
        """Test getting an existing distribution."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 5.0)
        dist = store.get_distribution("key1")
        assert dist is not None
        assert isinstance(dist, DistributionStats)

    def test_get_nonexistent_distribution(self: Self) -> None:
        """Test getting nonexistent key returns None."""
        store = KeyedDistributionStore()
        assert store.get_distribution("nonexistent") is None

    def test_get_returns_same_object(self: Self) -> None:
        """Test that get returns the actual stored object."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 5.0)
        dist1 = store.get_distribution("key1")
        dist2 = store.get_distribution("key1")
        assert dist1 is dist2


class TestKeyedDistributionStoreApplyDecay:
    """Tests for KeyedDistributionStore.apply_decay method."""

    def test_apply_decay_single_key(self: Self) -> None:
        """Test decay applied to single key."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 10.0)
        store.apply_decay(0.5)
        dist = store.get_distribution("key1")
        assert dist.support("on") == 5.0

    def test_apply_decay_multiple_keys(self: Self) -> None:
        """Test decay applied to all keys."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 10.0)
        store.update("key2", "on", 20.0)
        store.apply_decay(0.5)

        dist1 = store.get_distribution("key1")
        dist2 = store.get_distribution("key2")

        assert dist1.support("on") == 5.0
        assert dist2.support("on") == 10.0

    def test_apply_decay_empty_store(self: Self) -> None:
        """Test decay on empty store does nothing."""
        store = KeyedDistributionStore()
        store.apply_decay(0.5)  # Should not raise error
        assert store.get_distribution("any_key") is None

    def test_apply_decay_preserves_distribution_shape(self: Self) -> None:
        """Test that decay preserves probability distribution shape."""
        store = KeyedDistributionStore()
        store.update("key1", "a", 10.0)
        store.update("key1", "b", 10.0)

        dist_before = store.get_distribution("key1").distribution()
        store.apply_decay(0.7)
        dist_after = store.get_distribution("key1").distribution()

        # Probabilities should remain same
        for state in ["a", "b"]:
            assert abs(dist_before[state] - dist_after[state]) < 1e-9


class TestKeyedDistributionStorePrune:
    """Tests for KeyedDistributionStore.prune method."""

    def test_prune_removes_low_support_states(self: Self) -> None:
        """Test prune removes states below threshold."""
        store = KeyedDistributionStore()
        store.update("key1", "a", 5.0)
        store.update("key1", "b", 15.0)
        store.prune(epsilon=0.003, absolute_min=10.0)

        dist = store.get_distribution("key1")
        # 'a' with 5.0 should be removed (< 10.0)
        assert "a" not in dist.states()
        assert "b" in dist.states()

    def test_prune_removes_empty_distributions(self: Self) -> None:
        """Test prune removes keys with empty distributions."""
        store = KeyedDistributionStore()
        store.update("key1", "a", 5.0)
        store.update("key2", "b", 1000.0)

        # After pruning, key1 should be removed (all states < 20)
        store.prune(epsilon=0.003, absolute_min=20.0)

        assert store.get_distribution("key1") is None
        assert store.get_distribution("key2") is not None

    def test_prune_empty_store(self: Self) -> None:
        """Test prune on empty store."""
        store = KeyedDistributionStore()
        store.prune()
        assert store.get_distribution("any_key") is None

    def test_prune_with_custom_epsilon(self: Self) -> None:
        """Test prune with custom epsilon."""
        store = KeyedDistributionStore()
        store.update("key1", "a", 100.0)
        store.update("key1", "b", 1.0)  # 0.1% of total

        # With epsilon=0.001, threshold = max(101 * 0.001, 20) = 20
        store.prune(epsilon=0.001, absolute_min=20.0)

        dist = store.get_distribution("key1")
        assert "a" in dist.states()
        assert "b" not in dist.states()

    def test_prune_with_high_epsilon(self: Self) -> None:
        """Test prune with high epsilon value."""
        store = KeyedDistributionStore()
        store.update("key1", "a", 1000.0)
        store.update("key1", "b", 100.0)

        # With epsilon=0.1, threshold = max(1100 * 0.1, 20) = 110
        store.prune(epsilon=0.1, absolute_min=20.0)

        dist = store.get_distribution("key1")
        assert "a" in dist.states()
        assert "b" not in dist.states()


class TestKeyedDistributionStoreAggregate:
    """Tests for KeyedDistributionStore.aggregate method."""

    def test_aggregate_single_key_confident(self: Self) -> None:
        """Test aggregating single confident key."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 50.0)
        store.update("key1", "off", 30.0)

        result = store.aggregate(["key1"], min_support=50.0)
        assert result is not None
        agg, key = result
        assert key == "key1"
        assert agg.total_support() == 80.0

    def test_aggregate_multiple_keys_until_confident(self: Self) -> None:
        """Test aggregating multiple keys until threshold reached."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 10.0)
        store.update("key2", "on", 20.0)
        store.update("key3", "on", 30.0)

        # Should stop at key2 since 10+20=30 >= 25
        result = store.aggregate(["key1", "key2", "key3"], min_support=25.0)
        assert result is not None
        agg, key = result
        assert key == "key2"
        assert abs(agg.total_support() - 30.0) < 1e-6

    def test_aggregate_nonexistent_key_skipped(self: Self) -> None:
        """Test that nonexistent keys are skipped."""
        store = KeyedDistributionStore()
        store.update("key2", "on", 50.0)

        result = store.aggregate(["key1", "key2"], min_support=40.0)
        assert result is not None
        agg, key = result
        assert key == "key2"
        assert agg.total_support() == 50.0

    def test_aggregate_empty_list(self: Self) -> None:
        """Test aggregating empty list of keys."""
        store = KeyedDistributionStore()
        result = store.aggregate([], min_support=10.0)
        assert result is None

    def test_aggregate_never_reaches_threshold(self: Self) -> None:
        """Test aggregation when never reaching threshold."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 10.0)
        store.update("key2", "on", 15.0)

        result = store.aggregate(["key1", "key2"], min_support=100.0)
        # Should still return result with both keys and last key
        assert result is not None
        agg, key = result
        assert key is None
        assert agg.total_support() == 25.0

    def test_aggregate_preserves_probabilities(self: Self) -> None:
        """Test that aggregation combines distributions correctly."""
        store = KeyedDistributionStore()
        # key1: on=60%, off=40%
        store.update("key1", "on", 6.0)
        store.update("key1", "off", 4.0)
        # key2: on=50%, off=50%
        store.update("key2", "on", 5.0)
        store.update("key2", "off", 5.0)

        result = store.aggregate(["key1", "key2"], min_support=5.0)
        assert result is not None
        agg, key = result

        # Verify aggregation accumulated data
        assert agg.total_support() > 0
        dist = agg.distribution()
        # Verify both states are present
        assert "on" in dist
        assert "off" in dist
        # Verify probabilities sum to 1
        prob_sum = sum(dist.values())
        assert abs(prob_sum - 1.0) < 1e-6

    def test_aggregate_empty_store(self: Self) -> None:
        """Test aggregation on empty store."""
        store = KeyedDistributionStore()
        result = store.aggregate(["key1"], min_support=10.0)
        assert result is None


class TestKeyedDistributionStoreEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_string_and_numeric_keys(self: Self) -> None:
        """Test using both string and numeric keys."""
        store = KeyedDistributionStore()
        store.update("string_key", "on", 10.0)
        store.update(42, "on", 20.0)

        assert store.get_distribution("string_key") is not None
        assert store.get_distribution(42) is not None
        assert store.get_distribution(42).total_support() == 20.0

    def test_tuple_keys(self: Self) -> None:
        """Test using tuple keys."""
        store = KeyedDistributionStore()
        key = ("hour", 14)
        store.update(key, "on", 5.0)

        dist = store.get_distribution(key)
        assert dist is not None
        assert dist.support("on") == 5.0

    def test_very_large_number_of_keys(self: Self) -> None:
        """Test with many different keys."""
        store = KeyedDistributionStore()
        for i in range(1000):
            store.update(f"key_{i}", "state", 1.0)

        # Check a few keys
        assert store.get_distribution("key_0") is not None
        assert store.get_distribution("key_500") is not None
        assert store.get_distribution("key_999") is not None

    def test_zero_weight_updates(self: Self) -> None:
        """Test updating with zero weight."""
        store = KeyedDistributionStore()
        store.update("key1", "on", 0.0)
        dist = store.get_distribution("key1")
        assert dist.support("on") == 0.0

    def test_many_states_per_key(self: Self) -> None:
        """Test key with many different states."""
        store = KeyedDistributionStore()
        for i in range(100):
            store.update("key1", f"state_{i}", 1.0)

        dist = store.get_distribution("key1")
        assert len(dist.states()) == 100


class TestKeyedDistributionStoreIntegration:
    """Integration tests with multiple operations."""

    def test_workflow_update_decay_aggregate(self: Self) -> None:
        """Test complete workflow: update, decay, aggregate."""
        store = KeyedDistributionStore()
        store.update("morning", "on", 50.0)
        store.update("morning", "off", 30.0)
        store.update("evening", "on", 20.0)
        store.update("evening", "off", 10.0)

        # Apply decay
        store.apply_decay(0.8)

        # Aggregate
        result = store.aggregate(["morning", "evening"], min_support=30.0)
        assert result is not None
        agg, key = result
        assert key == "morning"

    def test_workflow_update_prune_aggregate(self: Self) -> None:
        """Test workflow: update, prune, aggregate."""
        store = KeyedDistributionStore()
        store.update("key1", "frequent", 1000.0)
        store.update("key1", "rare", 1.0)
        store.update("key2", "state", 100.0)

        # Prune with absolute_min=20
        store.prune(epsilon=0.003, absolute_min=20.0)

        # key1's "rare" should be removed
        key1_dist = store.get_distribution("key1")
        assert key1_dist is not None
        assert "rare" not in key1_dist.states()
        assert "frequent" in key1_dist.states()

    def test_many_updates_then_aggregate(self: Self) -> None:
        """Test many updates followed by aggregation."""
        store = KeyedDistributionStore()

        # Create temporal pattern
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                key = f"{hour:02d}:{minute:02d}"
                # Higher probability in day hours
                if 8 <= hour < 18:
                    store.update(key, "active", 2.0)
                    store.update(key, "inactive", 1.0)
                else:
                    store.update(key, "active", 0.5)
                    store.update(key, "inactive", 2.0)

        # Aggregate several time slots
        time_keys = [f"{h:02d}:00" for h in range(8, 18)]
        result = store.aggregate(time_keys, min_support=10.0)
        assert result is not None

    def test_sequential_decay_on_same_key(self: Self) -> None:
        """Test repeated decay operations."""
        store = KeyedDistributionStore()
        store.update("key1", "state", 1000.0)

        # Apply decay multiple times
        for _ in range(10):
            store.apply_decay(0.9)

        dist = store.get_distribution("key1")
        # Should still have some support
        assert dist.support("state") > 0
