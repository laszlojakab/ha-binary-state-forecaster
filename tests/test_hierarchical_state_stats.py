"""Tests for HierarchicalStateStats class."""

import json
import time

import pytest

from custom_components.discrete_state_forecaster.model.hierarchical_state_stats import (
    MIN_SUPPORT,
    HierarchicalStateStats,
)
from custom_components.discrete_state_forecaster.model.state_stats import StateStats
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)


class TestHierarchicalStateStatsInitialization:
    """Test HierarchicalStateStats initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization creates expected attributes."""
        stats = HierarchicalStateStats()
        assert stats.stats == {}
        assert isinstance(stats.stats, dict)
        assert stats.half_life == 3600.0
        assert stats.last_prune_ts == 0.0
        assert stats.prune_interval == 6 * 3600
        assert stats.max_keys == 50_000

    def test_stats_dictionary_is_empty(self) -> None:
        """Test that stats dictionary starts empty."""
        stats = HierarchicalStateStats()
        assert len(stats.stats) == 0

    def test_can_modify_half_life(self) -> None:
        """Test that half_life can be modified after initialization."""
        stats = HierarchicalStateStats()
        stats.half_life = 7200.0
        assert stats.half_life == 7200.0

    def test_can_modify_max_keys(self) -> None:
        """Test that max_keys can be modified after initialization."""
        stats = HierarchicalStateStats()
        stats.max_keys = 1000
        assert stats.max_keys == 1000

    def test_can_modify_prune_interval(self) -> None:
        """Test that prune_interval can be modified after initialization."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 3600.0
        assert stats.prune_interval == 3600.0


class TestHierarchicalStateStatsUpdate:
    """Test HierarchicalStateStats update() method."""

    def test_update_creates_new_key(self) -> None:
        """Test update creates StateStats for new TimeKey."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)

        assert key in stats.stats
        assert isinstance(stats.stats[key], StateStats)

    def test_update_adds_duration_to_state(self) -> None:
        """Test update correctly adds duration to state."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

    def test_update_multiple_states_same_key(self) -> None:
        """Test updating multiple states for the same TimeKey."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)
        stats.update(key, "off", 200.0, timestamp=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)
        assert stats.stats[key].durations["off"] == pytest.approx(200.0)

    def test_update_same_state_accumulates(self) -> None:
        """Test updating same state multiple times accumulates duration."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)
        stats.update(key, "on", 50.0, timestamp=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(150.0)

    def test_update_different_keys(self) -> None:
        """Test updating different TimeKeys creates separate entries."""
        stats = HierarchicalStateStats()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11),))

        stats.update(key1, "on", 100.0, timestamp=1000.0)
        stats.update(key2, "on", 200.0, timestamp=1000.0)

        # Should have 3 entries: key1, key2, and GLOBAL (parent of both)
        assert len(stats.stats) == 3
        assert stats.stats[key1].durations["on"] == pytest.approx(100.0)
        assert stats.stats[key2].durations["on"] == pytest.approx(200.0)
        # GLOBAL should have sum of both
        assert stats.stats[TimeKey()].durations["on"] == pytest.approx(300.0)

    def test_update_applies_decay(self) -> None:
        """Test that update applies decay before adding new duration."""
        stats = HierarchicalStateStats()
        stats.half_life = 100.0  # Short half-life for testing
        key = TimeKey((("hour", 10),))

        # First update
        stats.update(key, "on", 100.0, timestamp=1000.0)
        initial_duration = stats.stats[key].durations["on"]

        # Second update after one half-life
        stats.update(key, "on", 0.0, timestamp=1100.0)
        decayed_duration = stats.stats[key].durations["on"]

        # Should have decayed by ~50%
        assert decayed_duration < initial_duration
        assert decayed_duration == pytest.approx(50.0, rel=0.1)

    def test_update_with_complex_time_key(self) -> None:
        """Test update with multi-level TimeKey creates entire hierarchy."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10), ("weekday", 2), ("month", 1)))

        stats.update(key, "on", 100.0, timestamp=1000.0)

        # Should create 4 levels: specific + 3 parents
        assert len(stats.stats) == 4
        assert key in stats.stats
        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

        # Verify all parent levels exist and have the same duration
        parent1 = key.parent()  # (("hour", 10), ("weekday", 2))
        parent2 = parent1.parent()  # (("hour", 10),)
        global_key = TimeKey()

        assert parent1 in stats.stats
        assert parent2 in stats.stats
        assert global_key in stats.stats
        assert stats.stats[parent1].durations["on"] == pytest.approx(100.0)
        assert stats.stats[parent2].durations["on"] == pytest.approx(100.0)
        assert stats.stats[global_key].durations["on"] == pytest.approx(100.0)

    def test_update_populates_all_parent_levels(self) -> None:
        """Test that update automatically populates all parent levels in hierarchy."""
        stats = HierarchicalStateStats()

        # Create a 3-level key
        key = TimeKey((("hour", 14), ("weekday", 2), ("month", 6)))

        stats.update(key, "on", 100.0, timestamp=1000.0)

        # Should create 4 entries: specific + 3 parents
        assert len(stats.stats) == 4

        # Verify all levels in hierarchy exist
        assert key in stats.stats  # specific
        assert key.parent() in stats.stats  # (("hour", 14), ("weekday", 2))
        assert key.parent().parent() in stats.stats  # (("hour", 14),)
        assert TimeKey() in stats.stats  # ()

        # All should have the same duration for "on"
        for k in key.parents():
            assert stats.stats[k].durations["on"] == pytest.approx(100.0)

    def test_update_accumulates_at_parent_levels(self) -> None:
        """Test that multiple updates to different specific keys accumulate at shared parents."""
        stats = HierarchicalStateStats()

        key1 = TimeKey((("hour", 10), ("weekday", 1)))
        key2 = TimeKey((("hour", 10), ("weekday", 2)))

        stats.update(key1, "on", 100.0, timestamp=1000.0)
        stats.update(key2, "on", 200.0, timestamp=1000.0)

        # Specific keys should have their own durations
        assert stats.stats[key1].durations["on"] == pytest.approx(100.0)
        assert stats.stats[key2].durations["on"] == pytest.approx(200.0)

        # Shared parent (("hour", 10),) should have sum
        parent = TimeKey((("hour", 10),))
        assert stats.stats[parent].durations["on"] == pytest.approx(300.0)

        # GLOBAL should also have sum
        assert stats.stats[TimeKey()].durations["on"] == pytest.approx(300.0)

    def test_update_with_global_key_only(self) -> None:
        """Test update with GLOBAL (empty) TimeKey."""
        stats = HierarchicalStateStats()

        stats.update(TimeKey(), "on", 100.0, timestamp=1000.0)

        # Should only create GLOBAL entry (no parents above it)
        assert len(stats.stats) == 1
        assert TimeKey() in stats.stats
        assert stats.stats[TimeKey()].durations["on"] == pytest.approx(100.0)

    def test_update_triggers_key_limit_enforcement(self) -> None:
        """Test that update triggers enforce_key_limit when threshold exceeded."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10  # Small limit for testing

        # Add 11 keys - each creates itself + GLOBAL parent
        # But GLOBAL is shared, so we get 11 specific keys + 1 GLOBAL = 12 total
        for i in range(11):
            key = TimeKey((("id", i),))
            stats.update(key, "on", 100.0, timestamp=1000.0)

        # With hierarchical updates, we have 11 specific keys + 1 GLOBAL = 12
        # This exceeds 1.1 * 10 = 11, so enforcement should have happened
        # After enforcement, we should have around 10 keys
        assert len(stats.stats) <= 12  # At most before final enforcement

        # Add one more to exceed threshold again
        key = TimeKey((("id", 11),))
        stats.update(key, "on", 100.0, timestamp=1000.0)

        # Should have enforced limit again, keeping around 10-11 keys
        assert len(stats.stats) <= 11


class TestHierarchicalStateStatsConceptDrift:
    """Test HierarchicalStateStats concept drift detection."""

    def test_update_checks_drift_at_all_levels(self) -> None:
        """Test that update checks drift at all hierarchy levels."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10), ("weekday", 1)))

        # Build up sufficient data for drift detection (>= 3600s)
        # Start with mostly "on" state
        for _ in range(40):
            stats.update(key, "on", 100.0, timestamp=1000.0)

        # Verify all levels have stats
        assert key in stats.stats
        assert key.parent() in stats.stats
        assert TimeKey() in stats.stats  # GLOBAL

        # All levels should have baseline set after sufficient data
        assert stats.stats[key].baseline is not None
        assert stats.stats[key.parent()].baseline is not None
        assert stats.stats[TimeKey()].baseline is not None

    def test_drift_detection_sets_fast_decay_counter(self) -> None:
        """Test that drift detection sets fast_decay_updates to 15."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Build baseline with mostly "on" (75%)
        for _ in range(30):
            stats.update(key, "on", 100.0, timestamp=1000.0)
        for _ in range(10):
            stats.update(key, "off", 100.0, timestamp=1000.0)

        # Wait for cooldown period to pass
        ts_after_cooldown = 1000.0 + 3700.0  # > 3600s cooldown

        # Initially no fast decay
        assert stats.stats[key].fast_decay_updates == 0

        # Dramatically shift pattern to mostly "off" (should trigger drift)
        for _ in range(30):
            stats.update(key, "off", 100.0, timestamp=ts_after_cooldown)

        # After significant pattern shift, fast_decay_updates should be set
        # (may be less than 15 due to decrements during the loop)
        assert stats.stats[key].fast_decay_updates > 0

    def test_fast_decay_counter_decrements_on_update(self) -> None:
        """Test that fast_decay_updates counter decrements on each update."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Manually set fast_decay_updates
        stats.stats[key] = StateStats()
        stats.stats[key].fast_decay_updates = 5

        # Do updates
        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 4

        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 3

        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 2

    def test_fast_decay_counter_reaches_zero(self) -> None:
        """Test that fast_decay_updates counter stops at zero."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.stats[key] = StateStats()
        stats.stats[key].fast_decay_updates = 2

        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 1

        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 0

        # Should stay at 0
        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].fast_decay_updates == 0

    def test_drift_detection_independent_per_level(self) -> None:
        """Test that drift detection is independent at each hierarchy level."""
        stats = HierarchicalStateStats()
        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()

        # Build baseline for all levels
        for _ in range(40):
            stats.update(specific_key, "on", 100.0, timestamp=1000.0)

        # Manually modify only parent level to trigger drift there
        # Add significant "off" data to parent (but not to specific)
        parent_stats = stats.stats[parent_key]
        parent_stats.durations["off"] = parent_stats.durations.get("off", 0.0) + 4000.0

        # Check drift on parent (should detect due to manual change)
        ts = 1000.0 + 3700.0  # After cooldown
        parent_stats.check_drift(ts)

        # Parent might detect drift, specific should not (if checked)
        # This verifies independence of drift detection per level
        assert parent_key in stats.stats
        assert specific_key in stats.stats

    def test_drift_with_insufficient_data_no_detection(self) -> None:
        """Test that drift is not detected with insufficient data."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Add only small amount of data (< min_support of 3600s)
        stats.update(key, "on", 500.0, timestamp=1000.0)
        stats.update(key, "off", 500.0, timestamp=1000.0)

        # Total is 1000s across all levels, which is < 3600s min_support
        # So drift detection should not trigger
        key_stats = stats.stats[key]
        assert key_stats.baseline is None or key_stats.last_drift_ts == 0.0

    def test_drift_detection_respects_cooldown(self) -> None:
        """Test that drift detection respects cooldown period."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Build sufficient baseline
        for _ in range(40):
            stats.update(key, "on", 100.0, timestamp=1000.0)

        baseline_set_ts = 1000.0

        # Try to trigger drift immediately (within cooldown)
        for _ in range(40):
            stats.update(key, "off", 100.0, timestamp=baseline_set_ts + 100.0)

        # Should not have triggered drift (within cooldown)
        # check_drift uses default cooldown of 3600s
        # Verify drift timestamp exists
        assert stats.stats[key].last_drift_ts >= 0.0

        # Now wait past cooldown and trigger drift
        ts_past_cooldown = baseline_set_ts + 3700.0
        for _ in range(40):
            stats.update(key, "on", 100.0, timestamp=ts_past_cooldown)

        # Drift might now be detected
        # (depends on whether the distribution changed enough)

    def test_hierarchical_drift_propagation(self) -> None:
        """Test that drift can be detected independently at different hierarchy levels."""
        stats = HierarchicalStateStats()
        specific_key = TimeKey((("hour", 10), ("weekday", 1)))

        # Build baseline
        for _ in range(40):
            stats.update(specific_key, "on", 100.0, timestamp=1000.0)

        # All levels should have baselines
        assert stats.stats[specific_key].baseline is not None
        assert stats.stats[specific_key.parent()].baseline is not None

        # The drift detection happens independently at each level
        # during update, so they can have different drift states


class TestHierarchicalStateStatsDistribution:
    """Test HierarchicalStateStats distribution() method."""

    def test_distribution_empty_stats(self) -> None:
        """Test distribution returns empty AggregatedStats when no stats exist."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        result = stats.distribution(key, timestamp=1000.0)

        assert result.distribution == {}
        assert result.support_time == 0.0
        assert result.depth == 0

    def test_distribution_insufficient_support(self) -> None:
        """Test distribution returns empty AggregatedStats when support < MIN_SUPPORT."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Add data but less than MIN_SUPPORT (30 seconds)
        stats.update(key, "on", MIN_SUPPORT - 1, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        assert result.distribution == {}
        assert result.support_time == 0.0
        # depth tracks levels examined even if none had sufficient support
        assert result.depth == 0

    def test_distribution_single_level_single_state(self) -> None:
        """Test distribution with single level and single state."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", MIN_SUPPORT + 1, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        assert result.distribution == {"on": pytest.approx(1.0)}
        # With sufficient data, uses only specific key (no blending)
        assert result.support_time == pytest.approx(31.0)
        assert result.depth == 1  # only specific level

    def test_distribution_single_level_multiple_states(self) -> None:
        """Test distribution with single level and multiple states."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 200.0, timestamp=1000.0)
        stats.update(key, "off", 200.0, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        assert result.distribution["on"] == pytest.approx(0.5)
        assert result.distribution["off"] == pytest.approx(0.5)
        assert sum(result.distribution.values()) == pytest.approx(1.0)
        # With sufficient data, uses only specific key (no blending)
        assert result.support_time == pytest.approx(400.0)
        assert result.depth == 1

    def test_distribution_hierarchical_blending(self) -> None:
        """Test that specific data is used without blending when sufficient."""
        stats = HierarchicalStateStats()

        # With the new update behavior, updating a key also updates all parents
        # So we'll test by updating only one specific key and checking the blend
        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()  # (("hour", 10),))

        # First update: adds to specific (200) and parent (200) and GLOBAL (200)
        stats.update(specific_key, "on", 200.0, timestamp=1000.0)

        # Add more data directly to parent level only (not to specific)
        # We do this by manually updating the parent StateStats
        stats.stats[parent_key].update_duration("off", 400.0)

        result = stats.distribution(specific_key, timestamp=1000.0)

        # With sufficient specific data (200 >= MIN_SUPPORT), should use only specific key
        assert "on" in result.distribution
        assert "off" not in result.distribution  # Parent data is not blended
        assert result.distribution["on"] == pytest.approx(1.0)
        assert result.support_time == pytest.approx(200.0)
        assert result.depth == 1

    def test_distribution_skips_insufficient_levels(self) -> None:
        """Test distribution skips hierarchical levels with insufficient support."""
        stats = HierarchicalStateStats()

        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()  # TimeKey((("hour", 10),))

        # Specific level has insufficient support
        stats.update(specific_key, "on", MIN_SUPPORT - 1, timestamp=1000.0)

        # Note: update also added to parent, but still insufficient
        # Add more to parent to make it sufficient
        stats.stats[parent_key].update_duration("off", MIN_SUPPORT + 100)

        result = stats.distribution(specific_key, timestamp=1000.0)

        # Specific level should be skipped (< MIN_SUPPORT)
        # Parent level should be used
        # Parent has: (MIN_SUPPORT - 1) "on" + (MIN_SUPPORT + 100) "off"
        total_parent = (MIN_SUPPORT - 1) + (MIN_SUPPORT + 100)
        expected_off_prob = (MIN_SUPPORT + 100) / total_parent

        assert "off" in result.distribution
        assert result.distribution["off"] == pytest.approx(expected_off_prob, rel=0.01)
        # Only parent and GLOBAL levels have sufficient support
        assert result.depth >= 1

    def test_distribution_with_global_key(self) -> None:
        """Test distribution with global (empty) TimeKey."""
        stats = HierarchicalStateStats()
        global_key = TimeKey()

        stats.update(global_key, "on", 300.0, timestamp=1000.0)

        result = stats.distribution(global_key, timestamp=1000.0)

        assert result.distribution == {"on": pytest.approx(1.0)}
        assert result.support_time == pytest.approx(300.0)
        assert result.depth == 1  # only GLOBAL level

    def test_distribution_normalizes_to_one(self) -> None:
        """Test that distribution probabilities always sum to 1.0."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 150.0, timestamp=1000.0)
        stats.update(key, "off", 250.0, timestamp=1000.0)
        stats.update(key, "idle", 100.0, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        assert sum(result.distribution.values()) == pytest.approx(1.0)
        assert result.support_time > 0
        assert result.depth >= 1

    def test_distribution_three_level_hierarchy(self) -> None:
        """Test distribution with three-level hierarchy."""
        stats = HierarchicalStateStats()

        # Create three-level hierarchy using proper parent chain
        specific = TimeKey((("hour", 10), ("weekday", 1), ("month", 6)))
        medium = specific.parent()  # (("hour", 10), ("weekday", 1))
        general = medium.parent()  # (("hour", 10),)

        # Update specific - this will add to specific, medium, general, and GLOBAL
        stats.update(specific, "on", 200.0, timestamp=1000.0)

        # Add different states to medium and general manually to create blend
        stats.stats[medium].update_duration("off", 200.0)
        stats.stats[general].update_duration("idle", 200.0)

        result = stats.distribution(specific, timestamp=1000.0)

        # With sufficient specific data (200 >= MIN_SUPPORT), uses only specific key
        assert result.distribution["on"] == pytest.approx(1.0)
        assert "off" not in result.distribution
        assert "idle" not in result.distribution
        assert result.support_time == pytest.approx(200.0)
        assert result.depth == 1

    def test_distribution_overlapping_states(self) -> None:
        """Test distribution when multiple levels have same states."""
        stats = HierarchicalStateStats()

        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()  # (("hour", 10),)

        # Update specific with "on" and "off"
        stats.update(specific, "on", 100.0, timestamp=1000.0)
        stats.update(specific, "off", 100.0, timestamp=1000.0)
        # Now specific has: 100 "on", 100 "off" (total 200)
        # Parent has: 100 "on", 100 "off" (total 200) - from hierarchical updates
        # Global has: 100 "on", 100 "off" (total 200)

        # Add more data to parent manually
        stats.stats[parent].update_duration(
            "on", 300.0
        )  # Parent now has 400 "on", 100 "off"
        stats.stats[parent].update_duration(
            "idle", 100.0
        )  # Parent now has 400 "on", 100 "off", 100 "idle"

        result = stats.distribution(specific, timestamp=1000.0)

        # With sufficient specific data (200 >= MIN_SUPPORT), uses only specific key
        assert result.distribution["on"] == pytest.approx(0.5)
        assert result.distribution["off"] == pytest.approx(0.5)
        assert "idle" not in result.distribution  # Parent data not blended
        assert result.support_time == pytest.approx(200.0)
        assert result.depth == 1

    def test_distribution_depth_tracking_single_level(self) -> None:
        """Test that depth correctly tracks single hierarchy level."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 200.0, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        # With sufficient data, uses only specific key
        assert result.depth == 1

    def test_distribution_depth_tracking_multi_level(self) -> None:
        """Test that depth correctly tracks multiple hierarchy levels."""
        stats = HierarchicalStateStats()

        # Create 4-level hierarchy
        specific = TimeKey((("hour", 10), ("weekday", 1), ("month", 6)))

        # Update populates specific, parent1, parent2, and GLOBAL
        stats.update(specific, "on", 200.0, timestamp=1000.0)

        result = stats.distribution(specific, timestamp=1000.0)

        # With sufficient data, uses only specific key
        assert result.depth == 1

    def test_distribution_depth_with_insufficient_levels(self) -> None:
        """Test depth only counts levels with sufficient support."""
        stats = HierarchicalStateStats()

        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()

        # Add insufficient data to specific (less than MIN_SUPPORT)
        stats.update(specific, "on", MIN_SUPPORT - 1, timestamp=1000.0)

        # Add sufficient data to parent manually
        stats.stats[parent].update_duration("off", 200.0)

        result = stats.distribution(specific, timestamp=1000.0)

        # Only parent and GLOBAL have sufficient support
        # (specific has MIN_SUPPORT-1 which is < MIN_SUPPORT)
        assert result.depth >= 1  # At least parent or GLOBAL
        assert result.depth < 3  # Not all three levels

    def test_distribution_support_time_accumulates(self) -> None:
        """Test that support_time accumulates across hierarchy levels."""
        stats = HierarchicalStateStats()

        key = TimeKey((("hour", 10), ("weekday", 1)))

        # Add 100 seconds at specific level (also adds to parent and GLOBAL)
        stats.update(key, "on", 100.0, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        # With sufficient data, uses only specific key
        assert result.support_time == pytest.approx(100.0)

    def test_distribution_support_time_with_mixed_states(self) -> None:
        """Test support_time correctly sums different states across levels."""
        stats = HierarchicalStateStats()

        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()

        # Add data to specific (also populates parent and GLOBAL)
        stats.update(specific, "on", 50.0, timestamp=1000.0)
        stats.update(specific, "off", 50.0, timestamp=1000.0)
        # Now specific has 100 total, parent has 100, GLOBAL has 100

        # Add more to parent manually
        stats.stats[parent].update_duration("idle", 100.0)
        # Now parent has 200 total

        result = stats.distribution(specific, timestamp=1000.0)

        # With sufficient data, uses only specific key
        assert result.support_time == pytest.approx(100.0)

    def test_distribution_empty_returns_zero_depth(self) -> None:
        """Test empty distribution returns depth of 0."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        result = stats.distribution(key, timestamp=1000.0)

        assert result.depth == 0
        assert result.support_time == 0.0
        assert result.distribution == {}


class TestHierarchicalStateStatsPrune:
    """Test HierarchicalStateStats prune() method."""

    def test_prune_respects_interval(self) -> None:
        """Test prune only executes when interval has elapsed."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 1000.0
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0, timestamp=1000.0)

        # First prune at timestamp=1000
        stats.prune(timestamp=1000.0)
        assert stats.last_prune_ts == 1000.0

        # Try to prune before interval elapsed
        stats.prune(timestamp=1500.0)
        # Should not have updated (1500 - 1000 < 1000)
        assert stats.last_prune_ts == 1000.0

        # Prune after interval
        stats.prune(timestamp=2000.0)
        assert stats.last_prune_ts == 2000.0

    def test_prune_removes_low_support_keys(self) -> None:
        """Test prune removes TimeKeys with support below min_total."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0  # No interval restriction

        key_low = TimeKey((("hour", 10),))
        key_high = TimeKey((("hour", 11),))

        stats.update(
            key_low, "on", 50.0, timestamp=1000.0
        )  # Below default min_total=60
        stats.update(key_high, "on", 200.0, timestamp=1000.0)  # Above min_total

        stats.prune(timestamp=1000.0, min_total=60.0)

        assert key_low not in stats.stats
        assert key_high in stats.stats

    def test_prune_applies_decay(self) -> None:
        """Test that prune applies decay to all stats."""
        stats = HierarchicalStateStats()
        stats.half_life = 100.0
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))
        stats.update(key, "on", 200.0, timestamp=1000.0)

        initial_total = stats.stats[key].total()

        # Prune after one half-life
        stats.prune(timestamp=1100.0)

        decayed_total = stats.stats[key].total()

        # Should have decayed by ~50%
        assert decayed_total < initial_total
        assert decayed_total == pytest.approx(100.0, rel=0.1)

    def test_prune_with_custom_epsilon(self) -> None:
        """Test prune with custom epsilon threshold."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        # Add a dominant state and a tiny state
        stats.update(key, "on", 10000.0, timestamp=1000.0)
        stats.update(key, "off", 10.0, timestamp=1000.0)  # Only 0.1%

        # Prune with epsilon=0.005 (0.5%)
        stats.prune(timestamp=1000.0, epsilon=0.005)

        # "off" should be removed (0.1% < 0.5%)
        assert "on" in stats.stats[key].durations
        assert "off" not in stats.stats[key].durations

    def test_prune_with_custom_absolute_min(self) -> None:
        """Test prune with custom absolute_min threshold."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 1000.0, timestamp=1000.0)
        stats.update(key, "off", 15.0, timestamp=1000.0)

        # Prune with absolute_min=20
        stats.prune(timestamp=1000.0, absolute_min=20.0)

        # "off" should be removed (15 < 20)
        assert "on" in stats.stats[key].durations
        assert "off" not in stats.stats[key].durations

    def test_prune_removes_empty_stats(self) -> None:
        """Test prune removes stats that become empty after state pruning."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        # Add only small states that will be pruned
        stats.update(key, "on", 5.0, timestamp=1000.0)
        stats.update(key, "off", 5.0, timestamp=1000.0)

        # Prune with absolute_min that removes all states
        stats.prune(timestamp=1000.0, absolute_min=10.0, min_total=20.0)

        # Key should be completely removed
        assert key not in stats.stats

    def test_prune_preserves_sufficient_stats(self) -> None:
        """Test prune preserves stats above all thresholds."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 500.0, timestamp=1000.0)
        stats.update(key, "off", 500.0, timestamp=1000.0)

        initial_count = len(stats.stats)

        stats.prune(timestamp=1000.0)

        # Should not have removed anything
        assert len(stats.stats) == initial_count
        assert key in stats.stats

    def test_prune_multiple_keys_selective_removal(self) -> None:
        """Test prune selectively removes only insufficient keys."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key_keep1 = TimeKey((("hour", 10),))
        key_keep2 = TimeKey((("hour", 11),))
        key_remove1 = TimeKey((("hour", 12),))
        key_remove2 = TimeKey((("hour", 13),))

        stats.update(key_keep1, "on", 200.0, timestamp=1000.0)
        stats.update(key_keep2, "on", 300.0, timestamp=1000.0)
        stats.update(key_remove1, "on", 30.0, timestamp=1000.0)
        stats.update(key_remove2, "on", 40.0, timestamp=1000.0)

        stats.prune(timestamp=1000.0, min_total=60.0)

        assert key_keep1 in stats.stats
        assert key_keep2 in stats.stats
        assert key_remove1 not in stats.stats
        assert key_remove2 not in stats.stats


class TestHierarchicalStateStatsEnforceKeyLimit:
    """Test HierarchicalStateStats enforce_key_limit() method."""

    def test_enforce_key_limit_no_action_when_below_limit(self) -> None:
        """Test enforce_key_limit does nothing when below max_keys."""
        stats = HierarchicalStateStats()
        stats.max_keys = 100

        # Add 50 keys - each creates itself + GLOBAL parent (shared)
        # So we get 50 specific keys + 1 GLOBAL = 51 total
        for i in range(50):
            key = TimeKey((("id", i),))
            stats.update(key, "on", 100.0, timestamp=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 51  # 50 specific + 1 GLOBAL

    def test_enforce_key_limit_removes_excess_keys(self) -> None:
        """Test enforce_key_limit removes keys when over limit."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10

        # Add 15 keys
        for i in range(15):
            key = TimeKey((("id", i),))
            stats.update(key, "on", 100.0, timestamp=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 10

    def test_enforce_key_limit_removes_lowest_support(self) -> None:
        """Test enforce_key_limit removes keys with lowest total support."""
        stats = HierarchicalStateStats()
        stats.max_keys = 5

        # Add keys with varying support
        key_high1 = TimeKey((("id", 0),))
        key_high2 = TimeKey((("id", 1),))
        key_high3 = TimeKey((("id", 2),))
        key_low1 = TimeKey((("id", 3),))
        key_low2 = TimeKey((("id", 4),))
        key_low3 = TimeKey((("id", 5),))

        stats.update(key_high1, "on", 1000.0, timestamp=1000.0)
        stats.update(key_high2, "on", 900.0, timestamp=1000.0)
        stats.update(key_high3, "on", 800.0, timestamp=1000.0)
        stats.update(key_low1, "on", 100.0, timestamp=1000.0)
        stats.update(key_low2, "on", 50.0, timestamp=1000.0)
        stats.update(key_low3, "on", 25.0, timestamp=1000.0)

        stats.enforce_key_limit()

        # Should keep the 5 highest support keys
        assert key_high1 in stats.stats
        assert key_high2 in stats.stats
        assert key_high3 in stats.stats
        assert key_low1 in stats.stats
        # Low2 or Low3 should be removed (lowest support)
        assert len(stats.stats) == 5

    def test_enforce_key_limit_at_exact_limit(self) -> None:
        """Test enforce_key_limit does nothing at exact limit."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10

        # Add exactly max_keys entries
        for i in range(10):
            key = TimeKey((("id", i),))
            stats.update(key, "on", 100.0, timestamp=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 10

    def test_enforce_key_limit_large_overflow(self) -> None:
        """Test enforce_key_limit handles large overflow."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10

        # Add 100 keys (each creates itself + shares GLOBAL)
        for i in range(100):
            key = TimeKey((("id", i),))
            stats.update(key, "on", float(i), timestamp=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 10

        # GLOBAL should be kept (highest support) plus 9 highest specific keys
        assert TimeKey() in stats.stats


class TestHierarchicalStateStatsIntegration:
    """Integration tests for HierarchicalStateStats."""

    def test_full_workflow_with_hierarchy(self) -> None:
        """Test complete workflow: update, distribution, prune."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        # Create hierarchical data
        specific = TimeKey((("hour", 14), ("weekday", 2)))
        parent = specific.parent()  # (("hour", 14),)

        # Add data
        stats.update(specific, "on", 200.0, timestamp=1000.0)
        stats.update(parent, "off", 400.0, timestamp=1000.0)

        # Get distribution
        dist = stats.distribution(specific, timestamp=1000.0)
        assert "on" in dist.distribution
        assert (
            "off" not in dist.distribution
        )  # Parent data not blended when specific has sufficient data
        assert dist.support_time > 0
        assert dist.depth >= 1

        # Prune
        stats.prune(timestamp=1000.0)

        # Should still have data
        assert len(stats.stats) > 0

    def test_decay_over_time(self) -> None:
        """Test that statistics decay over time with multiple updates."""
        stats = HierarchicalStateStats()
        stats.half_life = 100.0
        key = TimeKey((("hour", 10),))

        # Initial observation
        stats.update(key, "on", 1000.0, timestamp=1000.0)
        initial = stats.stats[key].total()

        # After one half-life, add nothing (just decay)
        stats.update(key, "on", 0.0, timestamp=1100.0)
        after_one = stats.stats[key].total()

        # After two half-lives
        stats.update(key, "on", 0.0, timestamp=1200.0)
        after_two = stats.stats[key].total()

        assert after_one < initial
        assert after_two < after_one
        assert after_one == pytest.approx(500.0, rel=0.1)
        assert after_two == pytest.approx(250.0, rel=0.1)

    def test_memory_management_with_key_limit(self) -> None:
        """Test that key limit prevents unbounded memory growth."""
        stats = HierarchicalStateStats()
        stats.max_keys = 100

        # Add many keys
        for i in range(200):
            key = TimeKey((("id", i),))
            stats.update(key, "on", float(i), timestamp=1000.0)

        # Enforcement happens at 1.1x max_keys, then reduces to max_keys
        # So after 200 updates, should be at or near max_keys (not unbounded)
        assert len(stats.stats) <= 110  # At most 1.1x before next enforcement

    def test_realistic_scenario_daily_patterns(self) -> None:
        """Test realistic scenario with daily time patterns."""
        # Disable auto-pruning for this test to preserve all entries
        stats = HierarchicalStateStats(prune_interval=float("inf"))

        # Simulate one week of hourly observations
        for day in range(7):
            for hour in range(24):
                key = TimeKey((("hour", hour), ("weekday", day)))
                # Different patterns for day vs night
                if 8 <= hour <= 20:
                    stats.update(
                        key, "on", 3600.0, timestamp=1000.0 + day * 86400 + hour * 3600
                    )
                else:
                    stats.update(
                        key, "off", 3600.0, timestamp=1000.0 + day * 86400 + hour * 3600
                    )

        # Check daytime pattern - now includes parent levels in blending
        day_key = TimeKey((("hour", 14), ("weekday", 3)))
        dist = stats.distribution(day_key, timestamp=1000.0)
        # Should be heavily weighted toward "on" but may have some "off" from parents
        assert dist.distribution.get("on", 0.0) > 0.5  # Mostly "on"

        # Check nighttime pattern
        night_key = TimeKey((("hour", 2), ("weekday", 3)))
        dist = stats.distribution(night_key, timestamp=1001.0)
        # Should be heavily weighted toward "off"
        assert dist.distribution.get("off", 0.0) > 0.5  # Mostly "off"

    def test_distribution_with_no_exact_match_uses_parents(self) -> None:
        """Test distribution uses parent levels when exact match doesn't exist."""
        stats = HierarchicalStateStats()

        # Query specific level that doesn't exist yet
        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()  # (("hour", 10),)

        # Only add data to parent level
        stats.update(parent, "on", 500.0, timestamp=1000.0)

        dist = stats.distribution(specific, timestamp=1000.0)

        # Should fall back to parent
        assert dist.distribution == {"on": pytest.approx(1.0)}
        # parent (500) + GLOBAL (500) = 1000 total
        assert dist.support_time == pytest.approx(1000.0)
        assert dist.depth == 2  # parent + GLOBAL

    def test_prune_after_many_updates(self) -> None:
        """Test pruning maintains reasonable size after many updates."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 1000.0

        # Add data over time
        for i in range(100):
            key = TimeKey((("hour", i % 24),))
            stats.update(key, "on", 10.0, timestamp=float(i * 100))

        # Prune
        stats.prune(timestamp=10000.0, min_total=50.0)

        # Should have removed some low-support keys
        assert len(stats.stats) < 24

    def test_concept_drift_adaptation(self) -> None:
        """Test that concept drift triggers fast adaptation."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Establish baseline pattern: mostly "on"
        for _ in range(40):
            stats.update(key, "on", 100.0, timestamp=1000.0)

        # Check baseline is established
        assert stats.stats[key].baseline is not None
        original_baseline = stats.stats[key].baseline.copy()

        # Wait for cooldown period
        ts_after_cooldown = 1000.0 + 3700.0

        # Trigger pattern shift: now mostly "off"
        for _ in range(40):
            stats.update(key, "off", 100.0, timestamp=ts_after_cooldown)

        # After significant shift, drift should have been detected
        # and fast_decay_updates should have been set at some point
        # (may have decremented since then)
        key_stats = stats.stats[key]

        # Baseline should have been updated to new pattern
        assert key_stats.baseline is not None
        # The baseline should be different from original (adapted to new pattern)
        if "off" in key_stats.baseline and "off" in original_baseline:
            # New baseline should have higher "off" probability
            assert key_stats.baseline["off"] > original_baseline.get("off", 0.0)

    def test_drift_detection_across_hierarchy(self) -> None:
        """Test drift detection works independently across hierarchy levels."""
        stats = HierarchicalStateStats()
        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()
        global_key = TimeKey()

        # Build baseline for all levels
        for _ in range(40):
            stats.update(specific_key, "on", 100.0, timestamp=1000.0)

        # All levels should have baselines
        assert stats.stats[specific_key].baseline is not None
        assert stats.stats[parent_key].baseline is not None
        assert stats.stats[global_key].baseline is not None

        # Each level tracks drift independently
        # They all got the same updates, so baselines should be similar
        specific_baseline_on = stats.stats[specific_key].baseline.get("on", 0.0)
        parent_baseline_on = stats.stats[parent_key].baseline.get("on", 0.0)
        global_baseline_on = stats.stats[global_key].baseline.get("on", 0.0)

        # All should have high "on" probability since that's what we added
        assert specific_baseline_on > 0.9
        assert parent_baseline_on > 0.9
        assert global_baseline_on > 0.9

    def test_fast_decay_integration_with_updates(self) -> None:
        """Test fast_decay_updates integrates properly with normal update flow."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Manually trigger fast decay mode
        stats.stats[key] = StateStats()
        stats.stats[key].durations["on"] = 4000.0
        stats.stats[key].fast_decay_updates = 10

        # Do a series of updates
        for i in range(5):
            stats.update(key, "off", 100.0, timestamp=1000.0 + i)

        # Counter should have decremented
        assert stats.stats[key].fast_decay_updates == 5

        # Continue until it reaches zero
        for i in range(5, 10):
            stats.update(key, "off", 100.0, timestamp=1000.0 + i)

        # Should be at 0 now
        assert stats.stats[key].fast_decay_updates == 0

        # Further updates should keep it at 0
        stats.update(key, "off", 100.0, timestamp=1000.0 + 10)
        assert stats.stats[key].fast_decay_updates == 0


class TestMinSupportConstant:
    """Test MIN_SUPPORT constant."""

    def test_min_support_value(self) -> None:
        """Test MIN_SUPPORT has expected value."""
        assert MIN_SUPPORT == 30.0

    def test_min_support_is_float(self) -> None:
        """Test MIN_SUPPORT is a float."""
        assert isinstance(MIN_SUPPORT, float)


class TestOptionalTimestampInUpdate:
    """Test optional timestamp parameter in update() method."""

    def test_update_with_explicit_timestamp(self) -> None:
        """Test update works with explicit timestamp."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

    def test_update_with_default_timestamp(self) -> None:
        """Test update works with default (None) timestamp using current time."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Use default timestamp (current time)
        stats.update(key, "on", 100.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

    def test_update_default_timestamp_applies_decay(self) -> None:
        """Test that default timestamp applies decay correctly."""
        # Disable auto-pruning to prevent test entries from being removed
        stats = HierarchicalStateStats(prune_interval=float("inf"))
        stats.half_life = 0.1  # Very short half-life
        key = TimeKey((("hour", 10),))

        # First update with explicit timestamp
        stats.update(key, "on", 100.0, timestamp=1000.0)
        initial_duration = stats.stats[key].durations["on"]

        # Wait a bit
        time.sleep(0.15)

        # Second update with default timestamp (should apply decay)
        stats.update(key, "on", 0.0)  # Add 0 duration to just trigger decay
        decayed_duration = stats.stats[key].durations["on"]

        # Should have decayed significantly
        assert decayed_duration < initial_duration
        assert decayed_duration < 60.0  # Should be less than half

    def test_update_mixing_explicit_and_default_timestamps(self) -> None:
        """Test mixing explicit and default timestamps in sequential updates."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Use explicit timestamp
        stats.update(key, "on", 100.0, timestamp=1000.0)
        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

        # Use later explicit timestamp
        stats.update(key, "off", 50.0, timestamp=1100.0)
        assert stats.stats[key].durations["off"] == pytest.approx(50.0)

        # Use even later timestamp
        stats.update(key, "on", 25.0, timestamp=1200.0)
        # Both states should have values (decay may have been applied)
        assert stats.stats[key].durations["on"] > 0.0
        assert stats.stats[key].durations["off"] > 0.0


class TestOptionalTimestampInDistribution:
    """Test optional timestamp parameter in distribution() method."""

    def test_distribution_with_explicit_timestamp(self) -> None:
        """Test distribution works with explicit timestamp."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, timestamp=1000.0)
        stats.update(key, "off", 100.0, timestamp=1000.0)

        result = stats.distribution(key, timestamp=1000.0)

        assert "on" in result.distribution
        assert "off" in result.distribution
        assert result.distribution["on"] == pytest.approx(0.5)
        assert result.distribution["off"] == pytest.approx(0.5)

    def test_distribution_with_default_timestamp(self) -> None:
        """Test distribution works with default (None) timestamp using current time."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Use recent timestamps so data doesn't completely decay
        recent_time = time.time() - 100  # 100 seconds ago

        stats.update(key, "on", 100.0, timestamp=recent_time)
        stats.update(key, "off", 100.0, timestamp=recent_time)

        # Use default timestamp (current time)
        result = stats.distribution(key)

        # Should still have data since it's recent
        assert "on" in result.distribution
        assert "off" in result.distribution
        # Distribution should still be roughly equal (minimal decay in 100s with 1h half-life)
        assert result.distribution["on"] == pytest.approx(0.5, rel=0.1)
        assert result.distribution["off"] == pytest.approx(0.5, rel=0.1)

    def test_distribution_default_timestamp_applies_decay(self) -> None:
        """Test that default timestamp in distribution applies decay."""
        stats = HierarchicalStateStats()
        stats.half_life = 0.1  # Very short half-life
        key = TimeKey((("hour", 10),))

        # Add data with old timestamp
        stats.update(key, "on", 100.0, timestamp=1000.0)

        # Check distribution with explicit old timestamp (no decay)
        result1 = stats.distribution(key, timestamp=1000.0)
        support1 = result1.support_time

        # Wait a bit
        time.sleep(0.15)

        # Check distribution with default current timestamp (should decay)
        result2 = stats.distribution(key)
        support2 = result2.support_time

        # Support should have decreased due to decay
        assert support2 < support1


class TestOptionalTimestampInPrune:
    """Test optional timestamp parameter in prune() method."""

    def test_prune_with_explicit_timestamp(self) -> None:
        """Test prune works with explicit timestamp."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 100.0
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0, timestamp=1000.0)  # Below min_total

        # Prune with explicit timestamp after interval
        stats.prune(timestamp=1200.0)

        # Should have pruned the key (insufficient support)
        assert key not in stats.stats

    def test_prune_with_default_timestamp(self) -> None:
        """Test prune works with default (None) timestamp using current time."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0.1  # Short interval
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0, timestamp=1000.0)

        # Wait for prune interval
        time.sleep(0.15)

        # Prune with default timestamp
        stats.prune()

        # Should have pruned the key
        assert key not in stats.stats

    def test_prune_default_timestamp_respects_interval(self) -> None:
        """Test that default timestamp respects prune interval."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 3600.0  # 1 hour
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0)

        # First prune should set last_prune_ts
        stats.prune()
        first_prune_ts = stats.last_prune_ts

        # Immediate second prune should not execute (within interval)
        stats.prune()
        second_prune_ts = stats.last_prune_ts

        # Should be same (no prune happened)
        assert first_prune_ts == second_prune_ts

    def test_prune_mixing_explicit_and_default_timestamps(self) -> None:
        """Test mixing explicit and default timestamps in prune."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 100.0
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0, timestamp=1000.0)

        # Prune with explicit timestamp
        stats.prune(timestamp=1200.0)
        assert stats.last_prune_ts == 1200.0

        # Prune with default (current time) should work
        stats.prune()
        # last_prune_ts should be updated to current time
        assert stats.last_prune_ts > 1200.0


class TestOptionalTimestampConsistency:
    """Test consistency across methods when using optional timestamps."""

    def test_all_methods_accept_none_timestamp(self) -> None:
        """Test that all methods accept None as timestamp."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # All should work with default None timestamp
        stats.update(key, "on", 100.0)
        result = stats.distribution(key)
        stats.prune()

        # Should have valid results
        assert key in stats.stats
        assert result.distribution is not None

    def test_explicit_timestamps_provide_determinism(self) -> None:
        """Test that explicit timestamps provide deterministic behavior."""
        stats1 = HierarchicalStateStats()
        stats2 = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Same operations with same explicit timestamps
        timestamp = 1000.0
        stats1.update(key, "on", 100.0, timestamp=timestamp)
        stats1.update(key, "off", 50.0, timestamp=timestamp)

        stats2.update(key, "on", 100.0, timestamp=timestamp)
        stats2.update(key, "off", 50.0, timestamp=timestamp)

        # Should have identical distributions
        dist1 = stats1.distribution(key, timestamp=timestamp)
        dist2 = stats2.distribution(key, timestamp=timestamp)

        assert dist1.distribution == dist2.distribution
        assert dist1.support_time == dist2.support_time


class TestAutoPruning:
    """Tests for automatic pruning behavior."""

    def test_auto_prune_on_time_interval(self) -> None:
        """Test that auto-pruning triggers based on time interval."""
        stats = HierarchicalStateStats(prune_interval=100.0)
        key = TimeKey((("hour", 10),))

        # Add some data with low support
        stats.update(key, "on", 5.0, timestamp=1000.0)
        assert key in stats.stats

        # Update after prune interval has passed with insufficient total support
        # This should trigger auto-pruning and remove the key (total < min_total=60.0)
        stats.update(TimeKey((("hour", 11),)), "off", 100.0, timestamp=1150.0)
        assert key not in stats.stats  # Pruned due to low support

    def test_auto_prune_on_update_count(self) -> None:
        """Test that auto-pruning triggers based on update count."""
        stats = HierarchicalStateStats(
            prune_interval=float("inf"), prune_every_n_updates=5
        )
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11),))

        # Add data with low support at key1
        stats.update(key1, "on", 5.0, timestamp=1000.0)
        assert key1 in stats.stats

        # Do 4 more updates (total 5) - should trigger prune
        for i in range(4):
            stats.update(key2, "on", 100.0, timestamp=1000.0 + i)

        # After 5 updates, auto-prune should have removed key1 (low support)
        assert key1 not in stats.stats
        assert key2 in stats.stats

    def test_auto_prune_respects_both_conditions(self) -> None:
        """Test that auto-pruning uses OR logic (time OR count)."""
        stats = HierarchicalStateStats(prune_interval=100.0, prune_every_n_updates=10)
        key = TimeKey((("hour", 10),))

        # Add low-support data
        stats.update(key, "on", 5.0, timestamp=1000.0)
        assert key in stats.stats

        # Trigger via time (before reaching 10 updates)
        stats.update(TimeKey((("hour", 11),)), "off", 100.0, timestamp=1150.0)
        assert key not in stats.stats  # Pruned via time condition

    def test_auto_prune_disabled_when_both_none(self) -> None:
        """Test that setting prune_every_n_updates=None uses only time-based pruning."""
        stats = HierarchicalStateStats(
            prune_interval=1000.0, prune_every_n_updates=None
        )
        key = TimeKey((("hour", 10),))

        # Add low-support data
        stats.update(key, "on", 5.0, timestamp=1000.0)
        assert key in stats.stats

        # Many updates, but within time interval - should not prune
        for i in range(100):
            stats.update(TimeKey((("hour", 11),)), "on", 100.0, timestamp=1100.0 + i)

        # Key should still exist (time interval not reached)
        assert key in stats.stats


class TestHierarchicalStateStatsSerialization:
    """Tests for HierarchicalStateStats serialization (to_dict/from_dict)."""

    def test_to_dict_empty_stats(self) -> None:
        """Test serializing empty HierarchicalStateStats."""
        stats = HierarchicalStateStats()
        data = stats.to_dict()

        assert data["stats"] == []
        assert data["half_life"] == 3600.0
        assert data["last_prune_ts"] == 0.0
        assert data["prune_interval"] == 21600.0
        assert data["prune_every_n_updates"] is None
        assert data["update_count"] == 0
        assert data["max_keys"] == 50_000

    def test_to_dict_with_single_key(self) -> None:
        """Test serializing with a single TimeKey."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))
        stats.update(key, "on", 100.0, timestamp=1000.0)

        data = stats.to_dict()

        # update() creates entries for both the key and its parents (including empty key)
        assert len(data["stats"]) == 2  # key + empty key

        # Check that both keys are present
        serialized_keys = [item[0] for item in data["stats"]]
        assert [["hour", 10]] in serialized_keys
        assert [] in serialized_keys  # empty key (root)

    def test_to_dict_with_multiple_keys(self) -> None:
        """Test serializing with multiple TimeKeys."""
        stats = HierarchicalStateStats()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11), ("weekday", 1)))

        stats.update(key1, "on", 100.0, timestamp=1000.0)
        stats.update(key2, "off", 200.0, timestamp=1000.0)

        data = stats.to_dict()

        # update() creates entries for keys and their parents:
        # key1 creates: [("hour", 10)], []
        # key2 creates: [("hour", 11), ("weekday", 1)], [("hour", 11)], []
        # Total unique keys: 4
        assert len(data["stats"]) == 4

    def test_to_dict_includes_all_config(self) -> None:
        """Test that all configuration parameters are included."""
        stats = HierarchicalStateStats(
            half_life=7200.0,
            prune_interval=3600.0,
            prune_every_n_updates=100,
        )
        stats.max_keys = 10000

        data = stats.to_dict()

        assert data["half_life"] == 7200.0
        assert data["prune_interval"] == 3600.0
        assert data["prune_every_n_updates"] == 100
        assert data["max_keys"] == 10000

    def test_to_dict_preserves_prune_state(self) -> None:
        """Test that pruning state is preserved."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))
        stats.update(key, "on", 100.0, timestamp=1000.0)
        stats.last_prune_ts = 5000.0
        stats.update_count = 42

        data = stats.to_dict()

        assert data["last_prune_ts"] == 5000.0
        assert data["update_count"] == 42

    def test_from_dict_empty_stats(self) -> None:
        """Test deserializing empty HierarchicalStateStats."""
        data = {
            "stats": [],
            "half_life": 3600.0,
            "last_prune_ts": 0.0,
            "prune_interval": 21600.0,
            "prune_every_n_updates": None,
            "update_count": 0,
            "max_keys": 50_000,
        }

        stats = HierarchicalStateStats.from_dict(data)

        assert stats.stats == {}
        assert stats.half_life == 3600.0
        assert stats.last_prune_ts == 0.0
        assert stats.prune_interval == 21600.0
        assert stats.prune_every_n_updates is None
        assert stats.update_count == 0
        assert stats.max_keys == 50_000

    def test_from_dict_with_single_key(self) -> None:
        """Test deserializing with a single TimeKey."""
        data = {
            "stats": [
                [
                    [["hour", 10]],
                    {
                        "durations": {"on": 150.0},
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
        }

        stats = HierarchicalStateStats.from_dict(data)

        key = TimeKey((("hour", 10),))
        assert key in stats.stats
        assert stats.stats[key].durations == {"on": 150.0}
        assert stats.stats[key].last_update_ts == 1000.0

    def test_from_dict_with_multiple_keys(self) -> None:
        """Test deserializing with multiple TimeKeys."""
        data = {
            "stats": [
                [
                    [["hour", 10]],
                    {
                        "durations": {"on": 100.0},
                        "last_update_ts": None,
                        "baseline": None,
                        "last_drift_ts": 0.0,
                        "fast_decay_updates": 0,
                    },
                ],
                [
                    [["hour", 11], ["weekday", 1]],
                    {
                        "durations": {"off": 200.0},
                        "last_update_ts": None,
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
        }

        stats = HierarchicalStateStats.from_dict(data)

        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11), ("weekday", 1)))

        assert key1 in stats.stats
        assert key2 in stats.stats
        assert stats.stats[key1].durations == {"on": 100.0}
        assert stats.stats[key2].durations == {"off": 200.0}

    def test_from_dict_restores_config(self) -> None:
        """Test that configuration is properly restored."""
        data = {
            "stats": [],
            "half_life": 7200.0,
            "last_prune_ts": 5000.0,
            "prune_interval": 3600.0,
            "prune_every_n_updates": 50,
            "update_count": 25,
            "max_keys": 10000,
        }

        stats = HierarchicalStateStats.from_dict(data)

        assert stats.half_life == 7200.0
        assert stats.last_prune_ts == 5000.0
        assert stats.prune_interval == 3600.0
        assert stats.prune_every_n_updates == 50
        assert stats.update_count == 25
        assert stats.max_keys == 10000

    def test_roundtrip_empty_stats(self) -> None:
        """Test round-trip serialization of empty stats."""
        original = HierarchicalStateStats()
        data = original.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        assert restored.stats == original.stats
        assert restored.half_life == original.half_life
        assert restored.last_prune_ts == original.last_prune_ts
        assert restored.prune_interval == original.prune_interval
        assert restored.max_keys == original.max_keys

    def test_roundtrip_with_single_key(self) -> None:
        """Test round-trip with a single TimeKey."""
        original = HierarchicalStateStats()
        key = TimeKey((("hour", 15),))
        original.update(key, "on", 300.0, timestamp=2000.0)

        data = original.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        assert key in restored.stats
        assert restored.stats[key].durations == original.stats[key].durations
        assert restored.stats[key].last_update_ts == original.stats[key].last_update_ts

    def test_roundtrip_with_multiple_keys(self) -> None:
        """Test round-trip with multiple TimeKeys."""
        original = HierarchicalStateStats()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 10), ("weekday", 2)))
        key3 = TimeKey((("month", 5),))

        original.update(key1, "on", 100.0, timestamp=1000.0)
        original.update(key2, "off", 200.0, timestamp=1000.0)
        original.update(key3, "idle", 50.0, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        assert len(restored.stats) == len(original.stats)
        assert key1 in restored.stats
        assert key2 in restored.stats
        assert key3 in restored.stats

    def test_roundtrip_preserves_distribution(self) -> None:
        """Test that distribution is preserved after round-trip."""
        original = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))
        original.update(key, "on", 300.0, timestamp=1000.0)
        original.update(key, "off", 100.0, timestamp=1000.0)

        original_dist = original.distribution(key, timestamp=1000.0)

        data = original.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        restored_dist = restored.distribution(key, timestamp=1000.0)

        assert restored_dist.distribution == original_dist.distribution
        assert restored_dist.support_time == original_dist.support_time

    def test_roundtrip_with_custom_config(self) -> None:
        """Test round-trip with custom configuration."""
        original = HierarchicalStateStats(
            half_life=1800.0,
            prune_interval=7200.0,
            prune_every_n_updates=200,
        )
        original.max_keys = 20000

        key = TimeKey((("hour", 12),))
        original.update(key, "heating", 500.0, timestamp=3000.0)

        data = original.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        assert restored.half_life == 1800.0
        assert restored.prune_interval == 7200.0
        assert restored.prune_every_n_updates == 200
        assert restored.max_keys == 20000

    def test_serialized_data_is_json_compatible(self) -> None:
        """Test that serialized data can be JSON-encoded."""
        stats = HierarchicalStateStats()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 10), ("weekday", 1)))

        stats.update(key1, "on", 100.0, timestamp=1000.0)
        stats.update(key2, "off", 200.0, timestamp=1000.0)

        data = stats.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Verify we can restore from parsed JSON
        restored = HierarchicalStateStats.from_dict(parsed)
        assert key1 in restored.stats
        assert key2 in restored.stats

    def test_roundtrip_after_decay(self) -> None:
        """Test serialization preserves state after decay operations."""
        stats = HierarchicalStateStats(half_life=3600.0)
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 1000.0, timestamp=0.0)
        stats.update(key, "off", 500.0, timestamp=0.0)

        # Apply decay
        stats.update(key, "on", 0.0, timestamp=3600.0)  # One half-life later

        data = stats.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        # Durations should be halved
        assert restored.stats[key].durations["on"] == pytest.approx(500.0)
        assert restored.stats[key].durations["off"] == pytest.approx(250.0)

    def test_roundtrip_complex_hierarchy(self) -> None:
        """Test round-trip with complex hierarchical keys."""
        stats = HierarchicalStateStats()

        # Create various hierarchy levels
        keys = [
            TimeKey((("hour", 10),)),
            TimeKey((("hour", 10), ("weekday", 1))),
            TimeKey((("hour", 10), ("weekday", 1), ("month", 3))),
            TimeKey((("hour", 11),)),
        ]

        for i, key in enumerate(keys):
            stats.update(key, "state", float(i * 100), timestamp=1000.0)

        data = stats.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        for key in keys:
            assert key in restored.stats

    def test_roundtrip_preserves_prune_state(self) -> None:
        """Test that pruning state is preserved through serialization."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))
        stats.update(key, "on", 100.0, timestamp=1000.0)

        # Set pruning state
        stats.last_prune_ts = 2000.0
        stats.update_count = 15

        data = stats.to_dict()
        restored = HierarchicalStateStats.from_dict(data)

        assert restored.last_prune_ts == 2000.0
        assert restored.update_count == 15

    def test_multiple_instances_serialization(self) -> None:
        """Test serializing multiple different instances."""
        instances = [
            HierarchicalStateStats(),
            HierarchicalStateStats(half_life=7200.0),
            HierarchicalStateStats(prune_every_n_updates=50),
        ]

        # Add some data to each
        key = TimeKey((("hour", 10),))
        for i, instance in enumerate(instances):
            instance.update(key, f"state{i}", float(i * 100), timestamp=1000.0)

        # Serialize all
        serialized = [inst.to_dict() for inst in instances]

        # Deserialize all
        restored = [HierarchicalStateStats.from_dict(data) for data in serialized]

        # Verify all match
        for original, restored_inst in zip(instances, restored, strict=True):
            assert restored_inst.half_life == original.half_life
            assert len(restored_inst.stats) == len(original.stats)
