"""Tests for HierarchicalStateStats class."""

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

        stats.update(key, "on", 100.0, ts=1000.0)

        assert key in stats.stats
        assert isinstance(stats.stats[key], StateStats)

    def test_update_adds_duration_to_state(self) -> None:
        """Test update correctly adds duration to state."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, ts=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)

    def test_update_multiple_states_same_key(self) -> None:
        """Test updating multiple states for the same TimeKey."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, ts=1000.0)
        stats.update(key, "off", 200.0, ts=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(100.0)
        assert stats.stats[key].durations["off"] == pytest.approx(200.0)

    def test_update_same_state_accumulates(self) -> None:
        """Test updating same state multiple times accumulates duration."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 100.0, ts=1000.0)
        stats.update(key, "on", 50.0, ts=1000.0)

        assert stats.stats[key].durations["on"] == pytest.approx(150.0)

    def test_update_different_keys(self) -> None:
        """Test updating different TimeKeys creates separate entries."""
        stats = HierarchicalStateStats()
        key1 = TimeKey((("hour", 10),))
        key2 = TimeKey((("hour", 11),))

        stats.update(key1, "on", 100.0, ts=1000.0)
        stats.update(key2, "on", 200.0, ts=1000.0)

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
        stats.update(key, "on", 100.0, ts=1000.0)
        initial_duration = stats.stats[key].durations["on"]

        # Second update after one half-life
        stats.update(key, "on", 0.0, ts=1100.0)
        decayed_duration = stats.stats[key].durations["on"]

        # Should have decayed by ~50%
        assert decayed_duration < initial_duration
        assert decayed_duration == pytest.approx(50.0, rel=0.1)

    def test_update_with_complex_time_key(self) -> None:
        """Test update with multi-level TimeKey creates entire hierarchy."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10), ("weekday", 2), ("month", 1)))

        stats.update(key, "on", 100.0, ts=1000.0)

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

        stats.update(key, "on", 100.0, ts=1000.0)

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

        stats.update(key1, "on", 100.0, ts=1000.0)
        stats.update(key2, "on", 200.0, ts=1000.0)

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

        stats.update(TimeKey(), "on", 100.0, ts=1000.0)

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
            stats.update(key, "on", 100.0, ts=1000.0)

        # With hierarchical updates, we have 11 specific keys + 1 GLOBAL = 12
        # This exceeds 1.1 * 10 = 11, so enforcement should have happened
        # After enforcement, we should have around 10 keys
        assert len(stats.stats) <= 12  # At most before final enforcement

        # Add one more to exceed threshold again
        key = TimeKey((("id", 11),))
        stats.update(key, "on", 100.0, ts=1000.0)

        # Should have enforced limit again, keeping around 10-11 keys
        assert len(stats.stats) <= 11


class TestHierarchicalStateStatsDistribution:
    """Test HierarchicalStateStats distribution() method."""

    def test_distribution_empty_stats(self) -> None:
        """Test distribution returns empty dict when no stats exist."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        result = stats.distribution(key)

        assert result == {}

    def test_distribution_insufficient_support(self) -> None:
        """Test distribution returns empty dict when support < MIN_SUPPORT."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        # Add data but less than MIN_SUPPORT (120 seconds)
        stats.update(key, "on", MIN_SUPPORT - 1, ts=1000.0)

        result = stats.distribution(key)

        assert result == {}

    def test_distribution_single_level_single_state(self) -> None:
        """Test distribution with single level and single state."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", MIN_SUPPORT + 1, ts=1000.0)

        result = stats.distribution(key)

        assert result == {"on": pytest.approx(1.0)}

    def test_distribution_single_level_multiple_states(self) -> None:
        """Test distribution with single level and multiple states."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 200.0, ts=1000.0)
        stats.update(key, "off", 200.0, ts=1000.0)

        result = stats.distribution(key)

        assert result["on"] == pytest.approx(0.5)
        assert result["off"] == pytest.approx(0.5)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_distribution_hierarchical_blending(self) -> None:
        """Test distribution blends multiple hierarchical levels."""
        stats = HierarchicalStateStats()

        # With the new update behavior, updating a key also updates all parents
        # So we'll test by updating only one specific key and checking the blend
        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()  # (("hour", 10),))

        # First update: adds to specific (200) and parent (200) and GLOBAL (200)
        stats.update(specific_key, "on", 200.0, ts=1000.0)

        # Add more data directly to parent level only (not to specific)
        # We do this by manually updating the parent StateStats
        stats.stats[parent_key].update_duration("off", 400.0)

        result = stats.distribution(specific_key)

        # Now we have 3 levels:
        # Specific: 200 "on" (total 200)
        # Parent: 200 "on" + 400 "off" (total 600)
        # Global: 200 "on" (total 200)
        # Total support: 200 + 600 + 200 = 1000
        # Weights: 0.2, 0.6, 0.2
        # "on": 1.0*0.2 + (200/600)*0.6 + 1.0*0.2 = 0.2 + 0.2 + 0.2 = 0.6
        # "off": 0*0.2 + (400/600)*0.6 + 0*0.2 = 0 + 0.4 + 0 = 0.4
        assert "on" in result
        assert "off" in result
        assert result["on"] == pytest.approx(0.6, rel=0.01)
        assert result["off"] == pytest.approx(0.4, rel=0.01)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_distribution_skips_insufficient_levels(self) -> None:
        """Test distribution skips hierarchical levels with insufficient support."""
        stats = HierarchicalStateStats()

        specific_key = TimeKey((("hour", 10), ("weekday", 1)))
        parent_key = specific_key.parent()  # TimeKey((("hour", 10),))

        # Specific level has insufficient support
        stats.update(specific_key, "on", MIN_SUPPORT - 1, ts=1000.0)

        # Note: update also added to parent, but still insufficient
        # Add more to parent to make it sufficient
        stats.stats[parent_key].update_duration("off", MIN_SUPPORT + 100)

        result = stats.distribution(specific_key)

        # Specific level should be skipped (< MIN_SUPPORT)
        # Parent level should be used
        # Parent has: (MIN_SUPPORT - 1) "on" + (MIN_SUPPORT + 100) "off"
        total_parent = (MIN_SUPPORT - 1) + (MIN_SUPPORT + 100)
        expected_off_prob = (MIN_SUPPORT + 100) / total_parent

        assert "off" in result
        assert result["off"] == pytest.approx(expected_off_prob, rel=0.01)

    def test_distribution_with_global_key(self) -> None:
        """Test distribution with global (empty) TimeKey."""
        stats = HierarchicalStateStats()
        global_key = TimeKey()

        stats.update(global_key, "on", 300.0, ts=1000.0)

        result = stats.distribution(global_key)

        assert result == {"on": pytest.approx(1.0)}

    def test_distribution_normalizes_to_one(self) -> None:
        """Test that distribution probabilities always sum to 1.0."""
        stats = HierarchicalStateStats()
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 150.0, ts=1000.0)
        stats.update(key, "off", 250.0, ts=1000.0)
        stats.update(key, "idle", 100.0, ts=1000.0)

        result = stats.distribution(key)

        assert sum(result.values()) == pytest.approx(1.0)

    def test_distribution_three_level_hierarchy(self) -> None:
        """Test distribution with three-level hierarchy."""
        stats = HierarchicalStateStats()

        # Create three-level hierarchy using proper parent chain
        specific = TimeKey((("hour", 10), ("weekday", 1), ("month", 6)))
        medium = specific.parent()  # (("hour", 10), ("weekday", 1))
        general = medium.parent()  # (("hour", 10),)

        # Update specific - this will add to specific, medium, general, and GLOBAL
        stats.update(specific, "on", 200.0, ts=1000.0)

        # Add different states to medium and general manually to create blend
        stats.stats[medium].update_duration("off", 200.0)
        stats.stats[general].update_duration("idle", 200.0)

        result = stats.distribution(specific)

        # After updates:
        # Specific: 200 "on" (total 200)
        # Medium: 200 "on" + 200 "off" (total 400)
        # General: 200 "on" + 200 "idle" (total 400)
        # Global: 200 "on" (total 200)
        # Total support: 200 + 400 + 400 + 200 = 1200
        # Weights: 200/1200, 400/1200, 400/1200, 200/1200 = 1/6, 1/3, 1/3, 1/6

        # "on": 1.0*(1/6) + 0.5*(1/3) + 0.5*(1/3) + 1.0*(1/6) = 1/6 + 1/6 + 1/6 + 1/6 = 4/6 = 2/3
        # "off": 0*(1/6) + 0.5*(1/3) + 0*(1/3) + 0*(1/6) = 0 + 1/6 + 0 + 0 = 1/6
        # "idle": 0*(1/6) + 0*(1/3) + 0.5*(1/3) + 0*(1/6) = 0 + 0 + 1/6 + 0 = 1/6

        assert result["on"] == pytest.approx(2.0 / 3.0, rel=0.01)
        assert result["off"] == pytest.approx(1.0 / 6.0, rel=0.01)
        assert result["idle"] == pytest.approx(1.0 / 6.0, rel=0.01)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_distribution_overlapping_states(self) -> None:
        """Test distribution when multiple levels have same states."""
        stats = HierarchicalStateStats()

        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()  # (("hour", 10),)

        # Update specific with "on" and "off"
        stats.update(specific, "on", 100.0, ts=1000.0)
        stats.update(specific, "off", 100.0, ts=1000.0)
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

        result = stats.distribution(specific)

        # Specific: 100 "on", 100 "off" (total 200)
        # Parent: 400 "on", 100 "off", 100 "idle" (total 600)
        # Global: 100 "on", 100 "off" (total 200)
        # Total support: 200 + 600 + 200 = 1000
        # Weights: 0.2, 0.6, 0.2
        # "on": 0.5*0.2 + (400/600)*0.6 + 0.5*0.2 = 0.1 + 0.4 + 0.1 = 0.6
        # "off": 0.5*0.2 + (100/600)*0.6 + 0.5*0.2 = 0.1 + 0.1 + 0.1 = 0.3
        # "idle": 0*0.2 + (100/600)*0.6 + 0*0.2 = 0 + 0.1 + 0 = 0.1
        assert result["on"] == pytest.approx(0.6, rel=0.01)
        assert result["off"] == pytest.approx(0.3, rel=0.01)
        assert result["idle"] == pytest.approx(0.1, rel=0.01)
        assert sum(result.values()) == pytest.approx(1.0)


class TestHierarchicalStateStatsPrune:
    """Test HierarchicalStateStats prune() method."""

    def test_prune_respects_interval(self) -> None:
        """Test prune only executes when interval has elapsed."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 1000.0
        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 10.0, ts=1000.0)

        # First prune at ts=1000
        stats.prune(now_ts=1000.0)
        assert stats.last_prune_ts == 1000.0

        # Try to prune before interval elapsed
        stats.prune(now_ts=1500.0)
        # Should not have updated (1500 - 1000 < 1000)
        assert stats.last_prune_ts == 1000.0

        # Prune after interval
        stats.prune(now_ts=2000.0)
        assert stats.last_prune_ts == 2000.0

    def test_prune_removes_low_support_keys(self) -> None:
        """Test prune removes TimeKeys with support below min_total."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0  # No interval restriction

        key_low = TimeKey((("hour", 10),))
        key_high = TimeKey((("hour", 11),))

        stats.update(key_low, "on", 50.0, ts=1000.0)  # Below default min_total=60
        stats.update(key_high, "on", 200.0, ts=1000.0)  # Above min_total

        stats.prune(now_ts=1000.0, min_total=60.0)

        assert key_low not in stats.stats
        assert key_high in stats.stats

    def test_prune_applies_decay(self) -> None:
        """Test that prune applies decay to all stats."""
        stats = HierarchicalStateStats()
        stats.half_life = 100.0
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))
        stats.update(key, "on", 200.0, ts=1000.0)

        initial_total = stats.stats[key].total()

        # Prune after one half-life
        stats.prune(now_ts=1100.0)

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
        stats.update(key, "on", 10000.0, ts=1000.0)
        stats.update(key, "off", 10.0, ts=1000.0)  # Only 0.1%

        # Prune with epsilon=0.005 (0.5%)
        stats.prune(now_ts=1000.0, epsilon=0.005)

        # "off" should be removed (0.1% < 0.5%)
        assert "on" in stats.stats[key].durations
        assert "off" not in stats.stats[key].durations

    def test_prune_with_custom_absolute_min(self) -> None:
        """Test prune with custom absolute_min threshold."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 1000.0, ts=1000.0)
        stats.update(key, "off", 15.0, ts=1000.0)

        # Prune with absolute_min=20
        stats.prune(now_ts=1000.0, absolute_min=20.0)

        # "off" should be removed (15 < 20)
        assert "on" in stats.stats[key].durations
        assert "off" not in stats.stats[key].durations

    def test_prune_removes_empty_stats(self) -> None:
        """Test prune removes stats that become empty after state pruning."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        # Add only small states that will be pruned
        stats.update(key, "on", 5.0, ts=1000.0)
        stats.update(key, "off", 5.0, ts=1000.0)

        # Prune with absolute_min that removes all states
        stats.prune(now_ts=1000.0, absolute_min=10.0, min_total=20.0)

        # Key should be completely removed
        assert key not in stats.stats

    def test_prune_preserves_sufficient_stats(self) -> None:
        """Test prune preserves stats above all thresholds."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 0

        key = TimeKey((("hour", 10),))

        stats.update(key, "on", 500.0, ts=1000.0)
        stats.update(key, "off", 500.0, ts=1000.0)

        initial_count = len(stats.stats)

        stats.prune(now_ts=1000.0)

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

        stats.update(key_keep1, "on", 200.0, ts=1000.0)
        stats.update(key_keep2, "on", 300.0, ts=1000.0)
        stats.update(key_remove1, "on", 30.0, ts=1000.0)
        stats.update(key_remove2, "on", 40.0, ts=1000.0)

        stats.prune(now_ts=1000.0, min_total=60.0)

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
            stats.update(key, "on", 100.0, ts=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 51  # 50 specific + 1 GLOBAL

    def test_enforce_key_limit_removes_excess_keys(self) -> None:
        """Test enforce_key_limit removes keys when over limit."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10

        # Add 15 keys
        for i in range(15):
            key = TimeKey((("id", i),))
            stats.update(key, "on", 100.0, ts=1000.0)

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

        stats.update(key_high1, "on", 1000.0, ts=1000.0)
        stats.update(key_high2, "on", 900.0, ts=1000.0)
        stats.update(key_high3, "on", 800.0, ts=1000.0)
        stats.update(key_low1, "on", 100.0, ts=1000.0)
        stats.update(key_low2, "on", 50.0, ts=1000.0)
        stats.update(key_low3, "on", 25.0, ts=1000.0)

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
            stats.update(key, "on", 100.0, ts=1000.0)

        stats.enforce_key_limit()

        assert len(stats.stats) == 10

    def test_enforce_key_limit_large_overflow(self) -> None:
        """Test enforce_key_limit handles large overflow."""
        stats = HierarchicalStateStats()
        stats.max_keys = 10

        # Add 100 keys (each creates itself + shares GLOBAL)
        for i in range(100):
            key = TimeKey((("id", i),))
            stats.update(key, "on", float(i), ts=1000.0)

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
        stats.update(specific, "on", 200.0, ts=1000.0)
        stats.update(parent, "off", 400.0, ts=1000.0)

        # Get distribution
        dist = stats.distribution(specific)
        assert "on" in dist
        assert "off" in dist

        # Prune
        stats.prune(now_ts=1000.0)

        # Should still have data
        assert len(stats.stats) > 0

    def test_decay_over_time(self) -> None:
        """Test that statistics decay over time with multiple updates."""
        stats = HierarchicalStateStats()
        stats.half_life = 100.0
        key = TimeKey((("hour", 10),))

        # Initial observation
        stats.update(key, "on", 1000.0, ts=1000.0)
        initial = stats.stats[key].total()

        # After one half-life, add nothing (just decay)
        stats.update(key, "on", 0.0, ts=1100.0)
        after_one = stats.stats[key].total()

        # After two half-lives
        stats.update(key, "on", 0.0, ts=1200.0)
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
            stats.update(key, "on", float(i), ts=1000.0)

        # Enforcement happens at 1.1x max_keys, then reduces to max_keys
        # So after 200 updates, should be at or near max_keys (not unbounded)
        assert len(stats.stats) <= 110  # At most 1.1x before next enforcement

    def test_realistic_scenario_daily_patterns(self) -> None:
        """Test realistic scenario with daily time patterns."""
        stats = HierarchicalStateStats()

        # Simulate one week of hourly observations
        for day in range(7):
            for hour in range(24):
                key = TimeKey((("hour", hour), ("weekday", day)))
                # Different patterns for day vs night
                if 8 <= hour <= 20:
                    stats.update(
                        key, "on", 3600.0, ts=1000.0 + day * 86400 + hour * 3600
                    )
                else:
                    stats.update(
                        key, "off", 3600.0, ts=1000.0 + day * 86400 + hour * 3600
                    )

        # Check daytime pattern - now includes parent levels in blending
        day_key = TimeKey((("hour", 14), ("weekday", 3)))
        dist = stats.distribution(day_key)
        # Should be heavily weighted toward "on" but may have some "off" from parents
        assert dist.get("on", 0.0) > 0.5  # Mostly "on"

        # Check nighttime pattern
        night_key = TimeKey((("hour", 2), ("weekday", 3)))
        dist = stats.distribution(night_key)
        # Should be heavily weighted toward "off"
        assert dist.get("off", 0.0) > 0.5  # Mostly "off"

    def test_distribution_with_no_exact_match_uses_parents(self) -> None:
        """Test distribution uses parent levels when exact match doesn't exist."""
        stats = HierarchicalStateStats()

        # Query specific level that doesn't exist yet
        specific = TimeKey((("hour", 10), ("weekday", 1)))
        parent = specific.parent()  # (("hour", 10),)

        # Only add data to parent level
        stats.update(parent, "on", 500.0, ts=1000.0)

        dist = stats.distribution(specific)

        # Should fall back to parent
        assert dist == {"on": pytest.approx(1.0)}

    def test_prune_after_many_updates(self) -> None:
        """Test pruning maintains reasonable size after many updates."""
        stats = HierarchicalStateStats()
        stats.prune_interval = 1000.0

        # Add data over time
        for i in range(100):
            key = TimeKey((("hour", i % 24),))
            stats.update(key, "on", 10.0, ts=float(i * 100))

        # Prune
        stats.prune(now_ts=10000.0, min_total=50.0)

        # Should have removed some low-support keys
        assert len(stats.stats) < 24


class TestMinSupportConstant:
    """Test MIN_SUPPORT constant."""

    def test_min_support_value(self) -> None:
        """Test MIN_SUPPORT has expected value."""
        assert MIN_SUPPORT == 30.0

    def test_min_support_is_float(self) -> None:
        """Test MIN_SUPPORT is a float."""
        assert isinstance(MIN_SUPPORT, float)
