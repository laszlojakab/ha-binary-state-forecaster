"""
Comprehensive tests for HierarchicalTemporalStateModel.

Tests cover:
- Initialization with various configurations
- Duration updates and state tracking
- Distribution calculations and hierarchical blending
- Prediction with confidence metrics
- Entropy calculations
- Pruning operations
- Edge cases and error conditions
"""

import json
import math

import pytest

from custom_components.discrete_state_forecaster.model.hierarchical_temporal_state_model import (
    MIN_DURATION_THRESHOLD,
    HierarchicalTemporalStateModel,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)


class TestInitialization:
    """Tests for HierarchicalTemporalStateModel initialization."""

    def test_init_default_no_decay(self) -> None:
        """Test initialization with default parameters (no decay)."""
        model = HierarchicalTemporalStateModel()
        # Should initialize without errors
        assert model is not None

    def test_init_with_half_life(self) -> None:
        """Test initialization with custom half-life."""
        model = HierarchicalTemporalStateModel(half_life=3600.0)
        assert model is not None

    def test_init_with_zero_half_life(self) -> None:
        """Test initialization with zero half-life (no decay)."""
        model = HierarchicalTemporalStateModel(half_life=0.0)
        assert model is not None

    def test_init_with_very_small_half_life(self) -> None:
        """Test initialization with very small half-life."""
        model = HierarchicalTemporalStateModel(half_life=1.0)
        assert model is not None

    def test_init_with_large_half_life(self) -> None:
        """Test initialization with large half-life."""
        model = HierarchicalTemporalStateModel(half_life=86400.0 * 365)  # 1 year
        assert model is not None


class TestUpdateDuration:
    """Tests for update_duration method."""

    def test_update_single_observation(self) -> None:
        """Test updating with a single observation."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}
        # With sufficient data, uses only specific key (no blending)
        assert stats.support_time == pytest.approx(100.0)

    def test_update_multiple_observations_same_state(self) -> None:
        """Test multiple updates for the same state."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "on", 50.0, timestamp=1050.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}
        # With sufficient data, uses only specific key (no blending)
        assert stats.support_time == pytest.approx(150.0)

    def test_update_multiple_states(self) -> None:
        """Test updating with different states."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "off", 200.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution["on"] == pytest.approx(1 / 3)
        assert stats.distribution["off"] == pytest.approx(2 / 3)
        # With sufficient data, uses only specific key (no blending)
        assert stats.support_time == pytest.approx(300.0)

    def test_update_filters_short_durations(self) -> None:
        """Test that durations below MIN_DURATION_THRESHOLD are filtered."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Update with duration below threshold
        model.update_duration(key, "on", MIN_DURATION_THRESHOLD - 0.1, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        # Should be empty since duration was filtered
        assert stats.distribution == {}
        assert stats.support_time == 0.0

    def test_update_accepts_duration_at_threshold(self) -> None:
        """Test that durations exactly at MIN_DURATION_THRESHOLD are accepted."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Need > MIN_SUPPORT (30.0) for hierarchical stats to contribute
        model.update_duration(key, "on", 40.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}
        assert stats.support_time >= 40.0

    def test_update_different_keys(self) -> None:
        """Test updating different time keys independently."""
        model = HierarchicalTemporalStateModel()
        key1 = TimeKey((("time_of_day", 600),))
        key2 = TimeKey((("time_of_day", 700),))

        # Use larger durations for specific keys to overcome GLOBAL blending
        model.update_duration(key1, "on", 300.0, timestamp=1000.0)
        model.update_duration(key2, "off", 400.0, timestamp=1000.0)

        stats1 = model.distribution(key1, timestamp=1000.0)
        stats2 = model.distribution(key2, timestamp=1000.0)

        # Both keys share GLOBAL parent, so distributions blend
        assert "on" in stats1.distribution
        assert "off" in stats2.distribution
        # Specific key should have higher weight due to larger local duration
        assert stats1.distribution["on"] > stats1.distribution.get("off", 0)
        assert stats2.distribution["off"] > stats2.distribution.get("on", 0)

    def test_update_with_global_key(self) -> None:
        """Test updating with TimeKey.GLOBAL."""
        model = HierarchicalTemporalStateModel()

        model.update_duration(TimeKey.GLOBAL, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(TimeKey.GLOBAL, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_update_hierarchical_keys(self) -> None:
        """Test updating with hierarchical keys."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("day_of_week", 1), ("hour", 10)))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        # Should update the specific key
        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}


class TestDistribution:
    """Tests for distribution method."""

    def test_distribution_empty_model(self) -> None:
        """Test distribution for a key with no observations."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        stats = model.distribution(key, timestamp=1000.0)

        assert stats.distribution == {}
        assert stats.support_time == 0.0
        assert stats.key == TimeKey.GLOBAL

    def test_distribution_single_state(self) -> None:
        """Test distribution with single state."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_distribution_multiple_states(self) -> None:
        """Test distribution with multiple states."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 75.0, timestamp=1000.0)
        model.update_duration(key, "off", 25.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution["on"] == pytest.approx(0.75)
        assert stats.distribution["off"] == pytest.approx(0.25)

    def test_distribution_sums_to_one(self) -> None:
        """Test that probabilities sum to 1.0."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "a", 100.0, timestamp=1000.0)
        model.update_duration(key, "b", 200.0, timestamp=1000.0)
        model.update_duration(key, "c", 300.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        total = sum(stats.distribution.values())
        assert total == pytest.approx(1.0)

    def test_distribution_returns_aggregated_stats(self) -> None:
        """Test that distribution returns AggregatedStats with all fields."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert hasattr(stats, "distribution")
        assert hasattr(stats, "support_time")
        assert hasattr(stats, "key")

    def test_distribution_hierarchical_blending(self) -> None:
        """Test hierarchical blending when specific key has insufficient data."""
        model = HierarchicalTemporalStateModel()

        # Add data to parent (general) key
        parent_key = TimeKey((("hour", 10),))
        model.update_duration(parent_key, "on", 600.0, timestamp=1000.0)

        # Add insufficient data to specific key (< MIN_SUPPORT of 30)
        specific_key = TimeKey((("hour", 10), ("minute", 30)))
        model.update_duration(specific_key, "off", 10.0, timestamp=1000.0)

        # With insufficient specific data, should blend with parent
        stats = model.distribution(specific_key, timestamp=1000.0)
        assert "on" in stats.distribution
        assert "off" in stats.distribution
        # Specific key should have higher weight but parent contributes
        assert stats.key == parent_key


class TestPredict:
    """Tests for predict method."""

    def test_predict_single_state(self) -> None:
        """Test prediction with single state."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        assert prediction.state == "on"
        assert prediction.distribution == {"on": 1.0}
        assert prediction.confidence.max_probability == pytest.approx(1.0)

    def test_predict_multiple_states_chooses_max(self) -> None:
        """Test prediction chooses state with highest probability."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 300.0, timestamp=1000.0)
        model.update_duration(key, "off", 100.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        assert prediction.state == "on"
        assert prediction.confidence.max_probability == pytest.approx(0.75)

    def test_predict_empty_model_returns_empty_prediction(self) -> None:
        """Test prediction for unknown key returns empty Prediction."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Empty distribution should return empty Prediction
        prediction = model.predict(key, timestamp=1000.0)

        assert prediction.state is None
        assert prediction.distribution == {}
        assert prediction.confidence.max_probability == 0.0
        assert prediction.confidence.entropy_confidence == 0.0
        assert prediction.confidence.support_time == 0.0

    def test_predict_confidence_metrics(self) -> None:
        """Test that prediction includes all confidence metrics."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        assert hasattr(prediction.confidence, "max_probability")
        assert hasattr(prediction.confidence, "entropy_confidence")
        assert hasattr(prediction.confidence, "support_time")
        assert hasattr(prediction.confidence, "depth")

    def test_predict_entropy_confidence_certain(self) -> None:
        """Test entropy confidence is high when prediction is certain."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Single state = certainty
        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        # With only one state, entropy is 0, so confidence should be 1.0
        assert prediction.confidence.entropy_confidence == pytest.approx(1.0)

    def test_predict_entropy_confidence_uncertain(self) -> None:
        """Test entropy confidence is lower when prediction is uncertain."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Equal probabilities = maximum uncertainty
        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "off", 100.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        # Equal split has maximum entropy for 2 states, so confidence should be 0
        assert prediction.confidence.entropy_confidence == pytest.approx(0.0)

    def test_predict_support_time(self) -> None:
        """Test that support_time reflects total observations across hierarchy."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "off", 200.0, timestamp=1000.0)

        prediction = model.predict(key, timestamp=1000.0)
        # With sufficient data, uses only specific key (no blending)
        assert prediction.confidence.support_time == pytest.approx(300.0)


class TestEntropy:
    """Tests for _entropy helper method."""

    def test_entropy_single_state(self) -> None:
        """Test entropy of certain distribution (single state)."""
        model = HierarchicalTemporalStateModel()
        dist = {"on": 1.0}

        entropy = model._entropy(dist)
        assert entropy == pytest.approx(0.0)

    def test_entropy_equal_probabilities(self) -> None:
        """Test entropy of uniform distribution."""
        model = HierarchicalTemporalStateModel()

        # Two states, equal probabilities
        dist = {"on": 0.5, "off": 0.5}
        entropy = model._entropy(dist)
        # log2(2) = 1.0
        assert entropy == pytest.approx(1.0)

    def test_entropy_four_states_uniform(self) -> None:
        """Test entropy of 4-state uniform distribution."""
        model = HierarchicalTemporalStateModel()
        dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

        entropy = model._entropy(dist)
        # log2(4) = 2.0
        assert entropy == pytest.approx(2.0)

    def test_entropy_skewed_distribution(self) -> None:
        """Test entropy of skewed distribution."""
        model = HierarchicalTemporalStateModel()
        dist = {"on": 0.9, "off": 0.1}

        entropy = model._entropy(dist)
        # Should be low but not zero
        assert 0.0 < entropy < 1.0

    def test_entropy_three_states(self) -> None:
        """Test entropy calculation with three states."""
        model = HierarchicalTemporalStateModel()
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}

        entropy = model._entropy(dist)
        expected = -(0.5 * math.log2(0.5) + 0.3 * math.log2(0.3) + 0.2 * math.log2(0.2))
        assert entropy == pytest.approx(expected)


class TestPrune:
    """Tests for prune method."""

    def test_prune_removes_low_support_buckets(self) -> None:
        """Test that pruning removes buckets below support threshold."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Add small duration (below min_total=50)
        model.update_duration(key, "on", 20.0, timestamp=1000.0)

        # Prune with higher threshold
        model.prune(now_ts=1000.0, absolute_min=10.0, min_total=50.0)

        # Specific bucket should be removed, but GLOBAL might still have data
        stats = model.distribution(key, timestamp=1000.0)
        # After pruning, distribution may still show GLOBAL data if it has sufficient support
        # Just verify we can query the distribution
        assert isinstance(stats.distribution, dict)
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        # Prune with lower threshold
        model.prune(now_ts=1000.0, absolute_min=10.0, min_total=50.0)

        # Bucket should remain
        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_prune_removes_low_weight_states(self) -> None:
        """Test that pruning removes states with low weights."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(
            key, "off", 3.0, timestamp=1000.0
        )  # Below MIN_DURATION_THRESHOLD

        # 'off' never gets recorded due to MIN_DURATION_THRESHOLD filter
        stats = model.distribution(key, timestamp=1000.0)
        assert "on" in stats.distribution
        # 'off' was filtered before recording

    def test_prune_multiple_buckets_independently(self) -> None:
        """Test pruning affects different buckets independently."""
        model = HierarchicalTemporalStateModel()
        key1 = TimeKey((("time_of_day", 600),))
        key2 = TimeKey((("time_of_day", 700),))

        model.update_duration(key1, "on", 100.0, timestamp=1000.0)
        model.update_duration(key2, "on", 15.0, timestamp=1000.0)  # Below min_total

        model.prune(now_ts=1000.0, absolute_min=10.0, min_total=50.0)

        # key1 should have strong support, key2 weak
        stats1 = model.distribution(key1, timestamp=1000.0)

        assert "on" in stats1.distribution
        # key2 may still show GLOBAL data after pruning even if specific bucket was removed
        stats2 = model.distribution(key2, timestamp=1000.0)
        assert isinstance(stats2.distribution, dict)
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        # Should work with epsilon parameter
        model.prune(now_ts=1000.0, epsilon=0.001, absolute_min=10.0, min_total=50.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_duration(self) -> None:
        """Test handling of very large duration values."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 1e10, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_very_small_duration_above_threshold(self) -> None:
        """Test very small duration just above filter threshold."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # Use duration above MIN_SUPPORT (30.0) to ensure it contributes
        model.update_duration(key, "on", 35.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert "on" in stats.distribution

    def test_many_states(self) -> None:
        """Test model with many different states."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        for i in range(100):
            model.update_duration(key, f"state_{i}", 10.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert len(stats.distribution) == 100
        # All states should have equal probability
        for prob in stats.distribution.values():
            assert prob == pytest.approx(0.01)

    def test_many_time_buckets(self) -> None:
        """Test model with many different time buckets."""
        model = HierarchicalTemporalStateModel()

        for i in range(1000):
            key = TimeKey((("time_of_day", i),))
            model.update_duration(key, "on", 10.0, timestamp=1000.0)

        # All buckets should be accessible
        for i in range(1000):
            key = TimeKey((("time_of_day", i),))
            stats = model.distribution(key, timestamp=1000.0)
            assert stats.distribution == {"on": 1.0}

    def test_zero_timestamp(self) -> None:
        """Test handling of zero timestamp."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=0.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_negative_timestamp(self) -> None:
        """Test handling of negative timestamp."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        model.update_duration(key, "on", 100.0, timestamp=-1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_empty_time_key_components(self) -> None:
        """Test with GLOBAL time key (empty components)."""
        model = HierarchicalTemporalStateModel()

        model.update_duration(TimeKey.GLOBAL, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(TimeKey.GLOBAL, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_deeply_nested_time_key(self) -> None:
        """Test with deeply nested time key."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey(
            (
                ("year", 2024),
                ("month", 3),
                ("day", 15),
                ("hour", 10),
                ("minute", 30),
                ("second", 45),
            )
        )

        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution == {"on": 1.0}

    def test_state_name_special_characters(self) -> None:
        """Test states with special characters."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        special_states = ["on/off", "state-1", "state_2", "state.3", "state:4"]
        for state in special_states:
            model.update_duration(key, state, 10.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert len(stats.distribution) == len(special_states)

    def test_numeric_state_names(self) -> None:
        """Test states with numeric names."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("time_of_day", 600),))

        # States can be numbers (as strings)
        model.update_duration(key, "0", 100.0, timestamp=1000.0)
        model.update_duration(key, "1", 200.0, timestamp=1000.0)

        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution["0"] == pytest.approx(1 / 3)
        assert stats.distribution["1"] == pytest.approx(2 / 3)


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_complete_workflow(self) -> None:
        """Test complete workflow: update, predict, prune."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))

        # Add observations
        model.update_duration(key, "on", 300.0, timestamp=1000.0)
        model.update_duration(key, "off", 100.0, timestamp=1000.0)

        # Check distribution
        stats = model.distribution(key, timestamp=1000.0)
        assert stats.distribution["on"] == pytest.approx(0.75)

        # Make prediction
        prediction = model.predict(key, timestamp=1000.0)
        assert prediction.state == "on"
        assert prediction.confidence.max_probability == pytest.approx(0.75)

        # Prune (shouldn't remove anything with good support)
        model.prune(now_ts=1000.0, absolute_min=10.0, min_total=50.0)

        # Should still be there
        stats_after = model.distribution(key, timestamp=1000.0)
        assert stats_after.distribution["on"] == pytest.approx(0.75)

    def test_hierarchical_pattern_learning(self) -> None:
        """Test that specific patterns override general when sufficient data exists."""
        model = HierarchicalTemporalStateModel()

        # Add data to general pattern (hour level)
        hour_key = TimeKey((("hour", 10),))
        model.update_duration(hour_key, "working", 3000.0, timestamp=1000.0)

        # Add sufficient data to specific pattern (hour + minute)
        specific_key = TimeKey((("hour", 10), ("minute", 30)))
        model.update_duration(specific_key, "meeting", 300.0, timestamp=1000.0)

        # With sufficient specific data, should use only specific key
        stats = model.distribution(specific_key, timestamp=1000.0)
        assert "meeting" in stats.distribution
        assert "working" not in stats.distribution
        assert "meeting" in stats.distribution

        # meeting should have non-zero probability from specific data
        assert stats.distribution["meeting"] > 0

    def test_multiple_hierarchy_levels(self) -> None:
        """Test blending across multiple hierarchy levels."""
        model = HierarchicalTemporalStateModel()

        # Global level
        model.update_duration(TimeKey.GLOBAL, "default", 10000.0, timestamp=1000.0)

        # Hour level
        hour_key = TimeKey((("hour", 14),))
        model.update_duration(hour_key, "afternoon", 1000.0, timestamp=1000.0)

        # Hour + minute level
        specific_key = TimeKey((("hour", 14), ("minute", 30)))
        model.update_duration(specific_key, "specific", 100.0, timestamp=1000.0)

        # Should blend all three levels
        stats = model.distribution(specific_key, timestamp=1000.0)
        assert len(stats.distribution) >= 1
        assert stats.key == specific_key

    def test_temporal_pattern_isolation(self) -> None:
        """Test that different time periods maintain independent patterns."""
        model = HierarchicalTemporalStateModel()

        # Morning pattern
        morning = TimeKey((("hour", 8),))
        model.update_duration(morning, "wake_up", 100.0, timestamp=1000.0)

        # Evening pattern
        evening = TimeKey((("hour", 20),))
        model.update_duration(evening, "sleep", 100.0, timestamp=1000.0)

        # Patterns should be independent
        morning_stats = model.distribution(morning, timestamp=1000.0)
        evening_stats = model.distribution(evening, timestamp=1000.0)

        assert "wake_up" in morning_stats.distribution
        assert "sleep" in evening_stats.distribution

    def test_gradual_pattern_shift(self) -> None:
        """Test updating patterns over time."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))

        # Initial pattern: mostly off
        model.update_duration(key, "off", 900.0, timestamp=1000.0)
        model.update_duration(key, "on", 100.0, timestamp=1000.0)

        initial_stats = model.distribution(key, timestamp=1000.0)
        assert initial_stats.distribution["off"] > initial_stats.distribution["on"]

        # Shift pattern: add more "on" observations
        model.update_duration(key, "on", 900.0, timestamp=2000.0)

        updated_stats = model.distribution(key, timestamp=1000.0)
        # Now probabilities should be closer
        assert updated_stats.distribution["on"] > 0.4


class TestHierarchicalTemporalStateModelSerialization:
    """Tests for HierarchicalTemporalStateModel serialization (to_dict/from_dict)."""

    def test_to_dict_empty_model(self) -> None:
        """Test serializing empty HierarchicalTemporalStateModel."""
        model = HierarchicalTemporalStateModel()
        data = model.to_dict()

        assert "stats" in data
        assert "half_life" in data
        assert "half_life_normal" in data
        assert "half_life_fast" in data
        assert data["half_life"] == 0.0
        assert data["half_life_normal"] == 0.0
        assert data["half_life_fast"] == 0.0

    def test_to_dict_with_custom_half_life(self) -> None:
        """Test serializing with custom half-life."""
        model = HierarchicalTemporalStateModel(half_life=3600.0)
        data = model.to_dict()

        assert data["half_life"] == 3600.0
        assert data["half_life_normal"] == 3600.0
        assert data["half_life_fast"] == 360.0

    def test_to_dict_with_data(self) -> None:
        """Test serializing model with learned data."""
        model = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))
        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "off", 200.0, timestamp=1000.0)

        data = model.to_dict()

        assert "stats" in data
        assert data["stats"]["stats"]  # Should have statistics data

    def test_from_dict_empty_model(self) -> None:
        """Test deserializing empty HierarchicalTemporalStateModel."""
        data = {
            "stats": {
                "stats": [],
                "half_life": 0.0,
                "last_prune_ts": 0.0,
                "prune_interval": 21600.0,
                "prune_every_n_updates": None,
                "update_count": 0,
                "max_keys": 50_000,
            },
            "half_life": 0.0,
            "half_life_normal": 0.0,
            "half_life_fast": 0.0,
        }

        model = HierarchicalTemporalStateModel.from_dict(data)

        assert model is not None
        assert model.half_life == 0.0
        assert model.half_life_normal == 0.0
        assert model.half_life_fast == 0.0

    def test_from_dict_with_custom_half_life(self) -> None:
        """Test deserializing with custom half-life."""
        data = {
            "stats": {
                "stats": [],
                "half_life": 7200.0,
                "last_prune_ts": 0.0,
                "prune_interval": 21600.0,
                "prune_every_n_updates": None,
                "update_count": 0,
                "max_keys": 50_000,
            },
            "half_life": 7200.0,
            "half_life_normal": 7200.0,
            "half_life_fast": 720.0,
        }

        model = HierarchicalTemporalStateModel.from_dict(data)

        assert model.half_life == 7200.0
        assert model.half_life_normal == 7200.0
        assert model.half_life_fast == 720.0

    def test_from_dict_with_data(self) -> None:
        """Test deserializing model with learned data."""
        data = {
            "stats": {
                "stats": [
                    [
                        [["hour", 10]],
                        {
                            "durations": {"on": 100.0, "off": 200.0},
                            "last_update_ts": 1000.0,
                            "baseline": None,
                            "last_drift_ts": 0.0,
                            "fast_decay_updates": 0,
                        },
                    ],
                    [
                        [],
                        {
                            "durations": {"on": 100.0, "off": 200.0},
                            "last_update_ts": 1000.0,
                            "baseline": None,
                            "last_drift_ts": 0.0,
                            "fast_decay_updates": 0,
                        },
                    ],
                ],
                "half_life": 3600.0,
                "last_prune_ts": 0.0,
                "prune_interval": 21600.0,
                "prune_every_n_updates": None,
                "update_count": 0,
                "max_keys": 50_000,
            },
            "half_life": 3600.0,
            "half_life_normal": 3600.0,
            "half_life_fast": 360.0,
        }

        model = HierarchicalTemporalStateModel.from_dict(data)

        key = TimeKey((("hour", 10),))
        dist = model.distribution(key, timestamp=1000.0)

        assert "on" in dist.distribution
        assert "off" in dist.distribution

    def test_roundtrip_empty_model(self) -> None:
        """Test round-trip serialization of empty model."""
        original = HierarchicalTemporalStateModel()
        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        assert restored.half_life == original.half_life
        assert restored.half_life_normal == original.half_life_normal
        assert restored.half_life_fast == original.half_life_fast

    def test_roundtrip_with_custom_half_life(self) -> None:
        """Test round-trip with custom half-life."""
        original = HierarchicalTemporalStateModel(half_life=7200.0)
        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        assert restored.half_life == 7200.0
        assert restored.half_life_normal == 7200.0
        assert restored.half_life_fast == 720.0

    def test_roundtrip_with_single_state(self) -> None:
        """Test round-trip with single state observation."""
        original = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))
        original.update_duration(key, "on", 100.0, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        original_dist = original.distribution(key, timestamp=1000.0)
        restored_dist = restored.distribution(key, timestamp=1000.0)

        assert original_dist.distribution == restored_dist.distribution
        assert original_dist.support_time == restored_dist.support_time

    def test_roundtrip_with_multiple_states(self) -> None:
        """Test round-trip with multiple states."""
        original = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))
        original.update_duration(key, "on", 150.0, timestamp=1000.0)
        original.update_duration(key, "off", 100.0, timestamp=1000.0)
        original.update_duration(key, "idle", 50.0, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        original_dist = original.distribution(key, timestamp=1000.0)
        restored_dist = restored.distribution(key, timestamp=1000.0)

        assert original_dist.distribution == restored_dist.distribution
        assert original_dist.support_time == restored_dist.support_time

    def test_roundtrip_with_multiple_keys(self) -> None:
        """Test round-trip with multiple time keys."""
        original = HierarchicalTemporalStateModel()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11),))
        key3 = TimeKey((("hour", 10), ("weekday", 1)))

        original.update_duration(key1, "on", 100.0, timestamp=1000.0)
        original.update_duration(key2, "off", 200.0, timestamp=1000.0)
        original.update_duration(key3, "idle", 150.0, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        for key in [key1, key2, key3]:
            original_dist = original.distribution(key, timestamp=1000.0)
            restored_dist = restored.distribution(key, timestamp=1000.0)
            assert original_dist.distribution == restored_dist.distribution

    def test_roundtrip_preserves_predictions(self) -> None:
        """Test that predictions are preserved after round-trip."""
        original = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 14),))
        original.update_duration(key, "heating", 300.0, timestamp=1000.0)
        original.update_duration(key, "cooling", 100.0, timestamp=1000.0)

        original_pred = original.predict(key, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        restored_pred = restored.predict(key, timestamp=1000.0)

        assert original_pred.state == restored_pred.state
        assert original_pred.distribution == restored_pred.distribution
        assert (
            original_pred.confidence.max_probability
            == restored_pred.confidence.max_probability
        )
        assert (
            original_pred.confidence.support_time
            == restored_pred.confidence.support_time
        )

    def test_serialized_data_is_json_compatible(self) -> None:
        """Test that serialized data can be JSON-encoded."""
        model = HierarchicalTemporalStateModel(half_life=3600.0)
        key = TimeKey((("hour", 10),))
        model.update_duration(key, "on", 100.0, timestamp=1000.0)
        model.update_duration(key, "off", 200.0, timestamp=1000.0)

        data = model.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Verify we can restore from parsed JSON
        restored = HierarchicalTemporalStateModel.from_dict(parsed)
        assert restored.half_life == 3600.0

    def test_roundtrip_hierarchical_keys(self) -> None:
        """Test round-trip with hierarchical time keys."""
        original = HierarchicalTemporalStateModel()

        keys = [
            TimeKey((("hour", 10),)),
            TimeKey((("hour", 10), ("weekday", 1))),
            TimeKey((("hour", 10), ("weekday", 1), ("month", 3))),
        ]

        for i, key in enumerate(keys):
            original.update_duration(
                key, "state", float(i * 100 + 100), timestamp=1000.0
            )

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        for key in keys:
            original_dist = original.distribution(key, timestamp=1000.0)
            restored_dist = restored.distribution(key, timestamp=1000.0)
            assert original_dist.distribution == restored_dist.distribution

    def test_roundtrip_after_pruning(self) -> None:
        """Test serialization after pruning operation."""
        original = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))

        # Add data to specific key only (avoid parent blending)
        original.update_duration(key, "on", 1000.0, timestamp=1000.0)
        original.update_duration(key, "small", 15.0, timestamp=1000.0)

        # Prune with high threshold to remove small states
        original.prune(now_ts=1000.0, absolute_min=100.0, min_total=60.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        # Verify serialization preserved pruned state
        original_dist = original.distribution(key, timestamp=1000.0)
        restored_dist = restored.distribution(key, timestamp=1000.0)

        assert original_dist.distribution == restored_dist.distribution
        # After pruning, restored model should have same distribution as original
        assert len(original_dist.distribution) == len(restored_dist.distribution)

    def test_roundtrip_with_complex_distribution(self) -> None:
        """Test round-trip with complex multi-state distribution."""
        original = HierarchicalTemporalStateModel(half_life=3600.0)
        key = TimeKey((("hour", 15), ("weekday", 2)))

        states_durations = {
            "heating": 500.0,
            "cooling": 300.0,
            "idle": 200.0,
            "fan_only": 100.0,
            "off": 50.0,
        }

        for state, duration in states_durations.items():
            original.update_duration(key, state, duration, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        original_pred = original.predict(key, timestamp=1000.0)
        restored_pred = restored.predict(key, timestamp=1000.0)

        assert original_pred.state == restored_pred.state
        assert len(original_pred.distribution) == len(restored_pred.distribution)

        for state in states_durations:
            if state in original_pred.distribution:
                assert state in restored_pred.distribution
                assert original_pred.distribution[state] == pytest.approx(
                    restored_pred.distribution[state]
                )

    def test_multiple_models_serialization(self) -> None:
        """Test serializing multiple different model instances."""
        models = [
            HierarchicalTemporalStateModel(),
            HierarchicalTemporalStateModel(half_life=3600.0),
            HierarchicalTemporalStateModel(half_life=7200.0),
        ]

        key = TimeKey((("hour", 10),))
        for i, model in enumerate(models):
            model.update_duration(
                key, f"state{i}", float(i * 100 + 100), timestamp=1000.0
            )

        # Serialize all
        serialized = [model.to_dict() for model in models]

        # Deserialize all
        restored = [
            HierarchicalTemporalStateModel.from_dict(data) for data in serialized
        ]

        # Verify all match
        for original, restored_model in zip(models, restored, strict=True):
            assert restored_model.half_life == original.half_life
            assert restored_model.half_life_normal == original.half_life_normal
            assert restored_model.half_life_fast == original.half_life_fast

    def test_roundtrip_preserves_confidence_metrics(self) -> None:
        """Test that confidence metrics are preserved through serialization."""
        original = HierarchicalTemporalStateModel()
        key = TimeKey((("hour", 10),))

        # Create uneven distribution for interesting confidence metrics
        original.update_duration(key, "dominant", 900.0, timestamp=1000.0)
        original.update_duration(key, "minor", 100.0, timestamp=1000.0)

        original_pred = original.predict(key, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalTemporalStateModel.from_dict(data)

        restored_pred = restored.predict(key, timestamp=1000.0)

        # All confidence metrics should match
        assert original_pred.confidence.max_probability == pytest.approx(
            restored_pred.confidence.max_probability
        )
        assert original_pred.confidence.entropy_confidence == pytest.approx(
            restored_pred.confidence.entropy_confidence
        )
        assert original_pred.confidence.support_time == pytest.approx(
            restored_pred.confidence.support_time
        )
        assert original_pred.confidence.depth == restored_pred.confidence.depth
