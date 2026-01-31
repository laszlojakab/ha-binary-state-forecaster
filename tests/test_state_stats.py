"""Tests for `StateStats` class and decay behavior."""

import json
import time
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.state_stats import StateStats


class TestStateStatsInitialization:
    """Test StateStats initialization."""

    def test_default_initialization(self: Self) -> None:
        """Test default initialization creates empty durations dict."""
        stats = StateStats()
        assert stats.durations == {}
        assert isinstance(stats.durations, dict)

    def test_initialization_with_durations(self: Self) -> None:
        """Test initialization with explicit durations."""
        durations = {"on": 100.0, "off": 200.0}
        stats = StateStats(durations=durations)
        assert stats.durations == durations

    def test_durations_is_mutable(self: Self) -> None:
        """Test that durations can be modified after initialization."""
        stats = StateStats()
        stats.durations["on"] = 50.0
        assert stats.durations == {"on": 50.0}


class TestStateStatsTotal:
    """Test StateStats total() method."""

    def test_total_empty_durations(self: Self) -> None:
        """Test total with no durations returns 0."""
        stats = StateStats()
        assert stats.total() == 0.0

    def test_total_single_state(self: Self) -> None:
        """Test total with single state."""
        stats = StateStats(durations={"on": 100.0})
        assert stats.total() == 100.0

    def test_total_multiple_states(self: Self) -> None:
        """Test total with multiple states."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        assert stats.total() == 300.0

    def test_total_with_zero_duration(self: Self) -> None:
        """Test total with zero duration states."""
        stats = StateStats(durations={"on": 100.0, "off": 0.0})
        assert stats.total() == 100.0

    def test_total_all_zeros(self: Self) -> None:
        """Test total when all durations are zero."""
        stats = StateStats(durations={"on": 0.0, "off": 0.0})
        assert stats.total() == 0.0

    def test_total_with_float_precision(self: Self) -> None:
        """Test total handles floating point numbers correctly."""
        stats = StateStats(durations={"on": 100.5, "off": 200.25, "idle": 50.75})
        assert stats.total() == pytest.approx(351.5)

    def test_total_with_many_states(self: Self) -> None:
        """Test total with many states."""
        durations = {f"state_{i}": float(i) for i in range(100)}
        stats = StateStats(durations=durations)
        # Sum of 0 to 99 = 99 * 100 / 2 = 4950
        assert stats.total() == pytest.approx(4950.0)

    def test_total_with_negative_duration(self: Self) -> None:
        """Test total with negative durations (edge case)."""
        stats = StateStats(durations={"on": 100.0, "off": -50.0})
        assert stats.total() == 50.0

    def test_total_updates_with_modified_durations(self: Self) -> None:
        """Test that total updates when durations are modified."""
        stats = StateStats(durations={"on": 100.0})
        assert stats.total() == 100.0

        stats.durations["off"] = 200.0
        assert stats.total() == 300.0

    def test_total_with_string_states(self: Self) -> None:
        """Test total with string state names."""
        stats = StateStats(
            durations={"heating": 150.0, "cooling": 250.0, "idle": 100.0}
        )
        assert stats.total() == 500.0

    def test_total_with_numeric_states(self: Self) -> None:
        """Test total with numeric state identifiers."""
        stats = StateStats(durations={0: 100.0, 1: 200.0, 2: 300.0})
        assert stats.total() == 600.0

    def test_total_with_mixed_state_types(self: Self) -> None:
        """Test total with mixed hashable types as states."""
        stats = StateStats(durations={"on": 100.0, 1: 200.0, (0, 0): 300.0})
        assert stats.total() == 600.0


class TestStateStatsUpdateDuration:
    """Test StateStats update_duration() method."""

    def test_update_duration_new_state(self: Self) -> None:
        """Test updating duration for a new state."""
        stats = StateStats()
        stats.update_duration("on", 100.0)
        assert stats.durations == {"on": 100.0}

    def test_update_duration_existing_state(self: Self) -> None:
        """Test updating duration for an existing state accumulates."""
        stats = StateStats(durations={"on": 100.0})
        stats.update_duration("on", 50.0)
        assert stats.durations == {"on": 150.0}

    def test_update_duration_multiple_updates_same_state(self: Self) -> None:
        """Test multiple updates to same state accumulate correctly."""
        stats = StateStats()
        stats.update_duration("on", 100.0)
        stats.update_duration("on", 50.0)
        stats.update_duration("on", 25.0)
        assert stats.durations == {"on": 175.0}

    def test_update_duration_multiple_states(self: Self) -> None:
        """Test updating different states independently."""
        stats = StateStats()
        stats.update_duration("on", 100.0)
        stats.update_duration("off", 200.0)
        stats.update_duration("on", 50.0)
        assert stats.durations == {"on": 150.0, "off": 200.0}

    def test_update_duration_zero_value(self: Self) -> None:
        """Test updating with zero duration."""
        stats = StateStats()
        stats.update_duration("on", 0.0)
        assert stats.durations == {"on": 0.0}

    def test_update_duration_zero_to_existing(self: Self) -> None:
        """Test adding zero to existing duration has no effect on value."""
        stats = StateStats(durations={"on": 100.0})
        stats.update_duration("on", 0.0)
        assert stats.durations == {"on": 100.0}

    def test_update_duration_negative_value(self: Self) -> None:
        """Test updating with negative duration (edge case)."""
        stats = StateStats(durations={"on": 100.0})
        stats.update_duration("on", -30.0)
        assert stats.durations == {"on": 70.0}

    def test_update_duration_float_precision(self: Self) -> None:
        """Test updating with precise float values."""
        stats = StateStats()
        stats.update_duration("on", 100.25)
        stats.update_duration("on", 50.75)
        assert stats.durations["on"] == pytest.approx(151.0)

    def test_update_duration_small_values(self: Self) -> None:
        """Test updating with very small duration values."""
        stats = StateStats()
        stats.update_duration("on", 0.001)
        stats.update_duration("on", 0.002)
        assert stats.durations["on"] == pytest.approx(0.003)

    def test_update_duration_large_values(self: Self) -> None:
        """Test updating with large duration values."""
        stats = StateStats()
        stats.update_duration("on", 1e6)
        stats.update_duration("on", 2e6)
        assert stats.durations["on"] == pytest.approx(3e6)

    def test_update_duration_string_states(self: Self) -> None:
        """Test updating with string state names."""
        stats = StateStats()
        stats.update_duration("heating", 100.0)
        stats.update_duration("cooling", 200.0)
        stats.update_duration("heating", 50.0)
        assert stats.durations == {"heating": 150.0, "cooling": 200.0}

    def test_update_duration_numeric_states(self: Self) -> None:
        """Test updating with numeric state identifiers."""
        stats = StateStats()
        stats.update_duration(0, 100.0)
        stats.update_duration(1, 200.0)
        stats.update_duration(0, 50.0)
        assert stats.durations == {0: 150.0, 1: 200.0}

    def test_update_duration_tuple_states(self: Self) -> None:
        """Test updating with tuple state identifiers."""
        stats = StateStats()
        stats.update_duration(("on", "heating"), 100.0)
        stats.update_duration(("off", "cooling"), 200.0)
        stats.update_duration(("on", "heating"), 50.0)
        assert stats.durations == {("on", "heating"): 150.0, ("off", "cooling"): 200.0}

    def test_update_duration_does_not_affect_other_states(self: Self) -> None:
        """Test that updating one state doesn't affect others."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 50.0})
        stats.update_duration("on", 25.0)
        assert stats.durations == {"on": 125.0, "off": 200.0, "idle": 50.0}

    def test_update_duration_preserves_existing_states(self: Self) -> None:
        """Test that adding new state preserves existing ones."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        stats.update_duration("idle", 50.0)
        assert stats.durations == {"on": 100.0, "off": 200.0, "idle": 50.0}

    def test_update_duration_affects_total(self: Self) -> None:
        """Test that update_duration affects total()."""
        stats = StateStats()
        assert stats.total() == 0.0

        stats.update_duration("on", 100.0)
        assert stats.total() == 100.0

        stats.update_duration("off", 200.0)
        assert stats.total() == 300.0

        stats.update_duration("on", 50.0)
        assert stats.total() == 350.0

    def test_update_duration_affects_distribution(self: Self) -> None:
        """Test that update_duration affects distribution()."""
        stats = StateStats()
        stats.update_duration("on", 300.0)
        stats.update_duration("off", 100.0)

        dist = stats.distribution()
        assert dist["on"] == pytest.approx(0.75)
        assert dist["off"] == pytest.approx(0.25)

        # Update and check distribution changes
        stats.update_duration("off", 300.0)
        dist = stats.distribution()
        assert dist["on"] == pytest.approx(0.43, abs=0.01)
        assert dist["off"] == pytest.approx(0.57, abs=0.01)

    def test_update_duration_many_sequential_updates(self: Self) -> None:
        """Test many sequential updates accumulate correctly."""
        stats = StateStats()
        for _ in range(100):
            stats.update_duration("on", 1.0)
        assert stats.durations["on"] == pytest.approx(100.0)

    def test_update_duration_alternating_states(self: Self) -> None:
        """Test alternating updates between states."""
        stats = StateStats()
        for _ in range(10):
            stats.update_duration("on", 10.0)
            stats.update_duration("off", 5.0)
        assert stats.durations == {"on": 100.0, "off": 50.0}

    def test_update_duration_does_not_modify_last_update_ts(self: Self) -> None:
        """Test that update_duration doesn't change last_update_ts."""
        stats = StateStats()
        stats.last_update_ts = 1000.0

        stats.update_duration("on", 100.0)
        assert stats.last_update_ts == 1000.0

    def test_update_duration_does_not_trigger_decay(self: Self) -> None:
        """Test that update_duration doesn't apply decay."""
        stats = StateStats(durations={"on": 100.0})
        stats.last_update_ts = 0.0

        stats.update_duration("on", 50.0)
        # Should be simple addition, no decay applied
        assert stats.durations["on"] == 150.0

    def test_update_duration_with_existing_baseline(self: Self) -> None:
        """Test that update_duration doesn't affect baseline."""
        stats = StateStats(durations={"on": 100.0})
        stats.baseline = {"on": 0.75, "off": 0.25}

        stats.update_duration("on", 50.0)
        # Baseline should remain unchanged
        assert stats.baseline == {"on": 0.75, "off": 0.25}

    def test_update_duration_idempotent_structure(self: Self) -> None:
        """Test that durations dict structure is preserved."""
        stats = StateStats()
        stats.update_duration("on", 100.0)

        # Verify it's still a dict
        assert isinstance(stats.durations, dict)
        # Verify we can iterate
        assert list(stats.durations.keys()) == ["on"]


class TestStateStatsDistribution:
    """Test StateStats distribution() method."""

    def test_distribution_empty_durations(self: Self) -> None:
        """Test distribution with no durations returns empty dict."""
        stats = StateStats()
        assert stats.distribution() == {}

    def test_distribution_single_state(self: Self) -> None:
        """Test distribution with single state returns 100%."""
        stats = StateStats(durations={"on": 100.0})
        dist = stats.distribution()
        assert dist == {"on": 1.0}

    def test_distribution_two_equal_states(self: Self) -> None:
        """Test distribution with two equal duration states."""
        stats = StateStats(durations={"on": 100.0, "off": 100.0})
        dist = stats.distribution()
        assert dist == pytest.approx({"on": 0.5, "off": 0.5})

    def test_distribution_two_unequal_states(self: Self) -> None:
        """Test distribution with two unequal duration states."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        dist = stats.distribution()
        assert dist == pytest.approx({"on": 1 / 3, "off": 2 / 3})

    def test_distribution_multiple_states(self: Self) -> None:
        """Test distribution with multiple states."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 300.0})
        dist = stats.distribution()
        expected = {"on": 100.0 / 600.0, "off": 200.0 / 600.0, "idle": 300.0 / 600.0}
        assert dist == pytest.approx(expected)

    def test_distribution_sum_equals_one(self: Self) -> None:
        """Test that distribution probabilities sum to 1."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 150.0})
        dist = stats.distribution()
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_distribution_with_zero_duration_state(self: Self) -> None:
        """Test distribution with one zero duration state."""
        stats = StateStats(durations={"on": 100.0, "off": 0.0})
        dist = stats.distribution()
        assert dist == pytest.approx({"on": 1.0, "off": 0.0})

    def test_distribution_all_zeros(self: Self) -> None:
        """Test distribution when all durations are zero."""
        stats = StateStats(durations={"on": 0.0, "off": 0.0})
        assert stats.distribution() == {}

    def test_distribution_preserves_keys(self: Self) -> None:
        """Test that distribution contains all state keys."""
        durations = {"on": 100.0, "off": 200.0, "idle": 300.0}
        stats = StateStats(durations=durations)
        dist = stats.distribution()
        assert set(dist.keys()) == set(durations.keys())

    def test_distribution_with_float_precision(self: Self) -> None:
        """Test distribution handles floating point correctly."""
        stats = StateStats(durations={"on": 100.5, "off": 200.25})
        dist = stats.distribution()
        expected_on = 100.5 / 300.75
        expected_off = 200.25 / 300.75
        assert dist == pytest.approx({"on": expected_on, "off": expected_off})

    def test_distribution_all_positive_values(self: Self) -> None:
        """Test that all distribution values are non-negative."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 300.0})
        dist = stats.distribution()
        assert all(v >= 0 for v in dist.values())

    def test_distribution_values_bounded(self: Self) -> None:
        """Test that all distribution values are between 0 and 1."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 300.0})
        dist = stats.distribution()
        assert all(0 <= v <= 1 for v in dist.values())

    def test_distribution_updates_with_modified_durations(self: Self) -> None:
        """Test that distribution updates when durations are modified."""
        stats = StateStats(durations={"on": 100.0})
        dist1 = stats.distribution()
        assert dist1 == {"on": 1.0}

        stats.durations["off"] = 100.0
        dist2 = stats.distribution()
        assert dist2 == pytest.approx({"on": 0.5, "off": 0.5})

    def test_distribution_many_states(self: Self) -> None:
        """Test distribution with many states."""
        durations = {f"state_{i}": 10.0 for i in range(10)}
        stats = StateStats(durations=durations)
        dist = stats.distribution()

        # All should have equal probability
        assert len(dist) == 10
        assert all(v == pytest.approx(0.1) for v in dist.values())

    def test_distribution_very_small_durations(self: Self) -> None:
        """Test distribution with very small durations."""
        stats = StateStats(durations={"on": 0.001, "off": 0.002})
        dist = stats.distribution()
        assert dist == pytest.approx({"on": 1 / 3, "off": 2 / 3})

    def test_distribution_very_large_durations(self: Self) -> None:
        """Test distribution with very large durations."""
        stats = StateStats(durations={"on": 1e10, "off": 2e10})
        dist = stats.distribution()
        assert dist == pytest.approx({"on": 1 / 3, "off": 2 / 3})


class TestStateStatsEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_empty_stats_workflow(self: Self) -> None:
        """Test complete workflow starting from empty stats."""
        stats = StateStats()
        assert stats.total() == 0.0
        assert stats.distribution() == {}

        stats.durations["on"] = 100.0
        assert stats.total() == 100.0
        assert stats.distribution() == {"on": 1.0}

        stats.durations["off"] = 200.0
        assert stats.total() == 300.0
        assert stats.distribution() == pytest.approx({"on": 1 / 3, "off": 2 / 3})

    def test_dataclass_equality(self: Self) -> None:
        """Test that StateStats instances with same durations are equal."""
        stats1 = StateStats(durations={"on": 100.0, "off": 200.0})
        stats2 = StateStats(durations={"on": 100.0, "off": 200.0})
        assert stats1 == stats2

    def test_dataclass_inequality(self: Self) -> None:
        """Test that StateStats instances with different durations are not equal."""
        stats1 = StateStats(durations={"on": 100.0})
        stats2 = StateStats(durations={"on": 200.0})
        assert stats1 != stats2

    def test_tuple_state_keys(self: Self) -> None:
        """Test using tuples as state keys."""
        stats = StateStats(durations={(0, 0): 100.0, (1, 1): 200.0})
        assert stats.total() == 300.0
        dist = stats.distribution()
        assert dist == pytest.approx({(0, 0): 1 / 3, (1, 1): 2 / 3})

    def test_none_as_state_key(self: Self) -> None:
        """Test using None as a state key."""
        stats = StateStats(durations={None: 100.0, "on": 200.0})
        assert stats.total() == 300.0
        dist = stats.distribution()
        assert dist == pytest.approx({None: 1 / 3, "on": 2 / 3})

    def test_modifying_durations_directly(self: Self) -> None:
        """Test modifying durations dictionary directly."""
        stats = StateStats()
        stats.durations.update({"on": 100.0, "off": 200.0})
        assert stats.total() == 300.0

        del stats.durations["on"]
        assert stats.total() == 200.0

    def test_clearing_durations(self: Self) -> None:
        """Test clearing all durations."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        assert stats.total() == 300.0

        stats.durations.clear()
        assert stats.total() == 0.0
        assert stats.distribution() == {}

    def test_incremental_duration_updates(self: Self) -> None:
        """Test incrementally updating durations."""
        stats = StateStats()

        # Add duration incrementally
        stats.durations["on"] = stats.durations.get("on", 0.0) + 50.0
        assert stats.total() == 50.0

        stats.durations["on"] = stats.durations.get("on", 0.0) + 50.0
        assert stats.total() == 100.0

        stats.durations["off"] = stats.durations.get("off", 0.0) + 100.0
        assert stats.total() == 200.0


class TestApplyDecayInitialization:
    """
    Initialization behavior for `apply_decay()`.

    Verifies first invocation only sets `last_update_ts` and does not
    modify existing durations.
    """

    def test_first_call_sets_last_update_ts_without_change(self: Self) -> None:
        """First call sets timestamp without changing durations."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        assert stats.last_update_ts is None

        stats.apply_decay(timestamp=10.0, half_life=100.0)
        assert stats.last_update_ts == 10.0
        # No decay applied on first call
        assert stats.durations["on"] == 100.0
        assert stats.durations["off"] == 200.0


class TestApplyDecayBasic:
    """Basic scenarios for `apply_decay()` including no/negative elapsed."""

    def test_no_elapsed_time_no_change(self: Self) -> None:
        """No change when `current_ts == last_update_ts`."""
        stats = StateStats(durations={"on": 50.0})
        stats.last_update_ts = 10.0

        stats.apply_decay(timestamp=10.0, half_life=100.0)
        assert stats.last_update_ts == 10.0
        assert stats.durations["on"] == 50.0

    def test_negative_elapsed_no_change(self: Self) -> None:
        """No change when `current_ts < last_update_ts`."""
        stats = StateStats(durations={"on": 50.0})
        stats.last_update_ts = 10.0

        stats.apply_decay(timestamp=9.0, half_life=100.0)
        assert stats.last_update_ts == 10.0
        assert stats.durations["on"] == 50.0

    def test_zero_half_life_no_change(self: Self) -> None:
        """No change when `half_life == 0`."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        stats.last_update_ts = 0.0

        stats.apply_decay(timestamp=10.0, half_life=0.0)
        assert stats.last_update_ts == 10.0
        assert stats.durations["on"] == 100.0
        assert stats.durations["off"] == 200.0

    def test_positive_decay_scales_durations(self: Self) -> None:
        """Durations are scaled by 0.5^(elapsed/half_life) when elapsed > 0."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        stats.last_update_ts = 0.0

        # After 1 half-life, durations should be 50%
        stats.apply_decay(timestamp=10.0, half_life=10.0)
        assert stats.durations["on"] == pytest.approx(50.0)
        assert stats.durations["off"] == pytest.approx(100.0)
        assert stats.last_update_ts == 10.0


class TestApplyDecayCumulative:
    """Cumulative behavior across multiple `apply_decay()` calls."""

    def test_multiple_decay_calls_accumulate(self: Self) -> None:
        """Successive calls multiply factors for additional elapsed time."""
        stats = StateStats(durations={"a": 10.0, "b": 20.0})
        stats.last_update_ts = 0.0

        # First decay: elapsed=5, half_life=10 -> factor=0.5^0.5
        stats.apply_decay(timestamp=5.0, half_life=10.0)
        factor1 = 0.5 ** (5.0 / 10.0)
        assert stats.durations["a"] == pytest.approx(10.0 * factor1)
        assert stats.durations["b"] == pytest.approx(20.0 * factor1)

        # Second decay: elapsed=3, half_life=10 -> factor=0.5^0.3
        stats.apply_decay(timestamp=8.0, half_life=10.0)
        factor2 = 0.5 ** (3.0 / 10.0)
        # Overall factor multiplies
        assert stats.durations["a"] == pytest.approx(10.0 * factor1 * factor2)
        assert stats.durations["b"] == pytest.approx(20.0 * factor1 * factor2)
        assert stats.last_update_ts == 8.0


class TestApplyDecayEdgeCases:
    """Edge cases around empty durations and distribution stability."""

    def test_empty_durations_safe(self: Self) -> None:
        """Safe when durations are empty; only timestamp is set."""
        stats = StateStats()
        stats.apply_decay(timestamp=10.0, half_life=100.0)
        # No exception and last_update_ts set
        assert stats.last_update_ts == 10.0
        assert stats.durations == {}

    def test_distribution_after_decay_scales_consistently(self: Self) -> None:
        """Decay preserves probability ratios in the distribution."""
        stats = StateStats(durations={"x": 1.0, "y": 3.0})
        stats.last_update_ts = 0.0

        # Decay should preserve ratios
        stats.apply_decay(timestamp=10.0, half_life=100.0)
        dist = stats.distribution()
        assert dist["x"] == pytest.approx(0.25)
        assert dist["y"] == pytest.approx(0.75)


class TestApplyDecayOptionalTimestamp:
    """Test apply_decay with optional timestamp parameter."""

    def test_default_timestamp_uses_current_time(self: Self) -> None:
        """When timestamp is None, uses current system time."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})

        # Call without timestamp - should use time.time()
        stats.apply_decay(half_life=3600.0)

        # Should have set last_update_ts to something
        assert stats.last_update_ts is not None
        assert stats.last_update_ts > 0
        # Durations should be unchanged on first call
        assert stats.durations["on"] == 100.0
        assert stats.durations["off"] == 200.0

    def test_default_timestamp_on_second_call_applies_decay(self: Self) -> None:
        """Second call without timestamp applies decay based on elapsed time."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})

        # First call sets timestamp
        stats.apply_decay(half_life=1.0)
        first_ts = stats.last_update_ts

        # Wait a tiny bit
        time.sleep(0.01)

        # Second call should apply decay
        stats.apply_decay(half_life=1.0)

        # Timestamp should be updated
        assert stats.last_update_ts is not None
        assert stats.last_update_ts > first_ts

        # Durations should have decayed (very small amount due to short wait)
        assert stats.durations["on"] < 100.0
        assert stats.durations["off"] < 200.0

    def test_explicit_timestamp_overrides_default(self: Self) -> None:
        """Explicit timestamp parameter overrides default time.time()."""
        stats = StateStats(durations={"on": 100.0})
        stats.last_update_ts = 0.0

        # Use explicit timestamp
        stats.apply_decay(timestamp=10.0, half_life=10.0)

        assert stats.last_update_ts == 10.0
        assert stats.durations["on"] == pytest.approx(50.0)

    def test_mixing_explicit_and_default_timestamps(self: Self) -> None:
        """Can mix explicit and default timestamps across calls."""
        stats = StateStats(durations={"on": 100.0})

        # First call with explicit timestamp
        stats.apply_decay(timestamp=0.0, half_life=100.0)
        assert stats.last_update_ts == 0.0

        # Second call with explicit timestamp
        stats.apply_decay(timestamp=100.0, half_life=100.0)
        assert stats.last_update_ts == 100.0
        assert stats.durations["on"] == pytest.approx(50.0)

        # Third call with default (should use current time)
        stats.apply_decay(half_life=100.0)
        # Should update to current time (which is >> 100.0)
        assert stats.last_update_ts > 100.0

    def test_default_timestamp_with_zero_half_life(self: Self) -> None:
        """Default timestamp with zero half-life updates timestamp only."""
        stats = StateStats(durations={"on": 100.0})

        stats.apply_decay(half_life=0.0)

        assert stats.last_update_ts is not None
        assert stats.last_update_ts > 0
        # No decay applied
        assert stats.durations["on"] == 100.0

    def test_default_timestamp_empty_durations(self: Self) -> None:
        """Default timestamp works with empty durations."""
        stats = StateStats()

        stats.apply_decay(half_life=100.0)

        assert stats.last_update_ts is not None
        assert stats.last_update_ts > 0
        assert stats.durations == {}

    def test_explicit_none_timestamp_same_as_default(self: Self) -> None:
        """Explicitly passing None is same as omitting parameter."""
        stats1 = StateStats(durations={"on": 100.0})
        stats2 = StateStats(durations={"on": 100.0})

        # Both should behave identically
        stats1.apply_decay(timestamp=None, half_life=3600.0)
        stats2.apply_decay(half_life=3600.0)

        # Both should have timestamps set
        assert stats1.last_update_ts is not None
        assert stats2.last_update_ts is not None
        # Should be very close (within milliseconds)
        assert abs(stats1.last_update_ts - stats2.last_update_ts) < 0.1


class TestPruneBasicBehavior:
    """Test basic prune functionality."""

    def test_prune_removes_states_below_threshold(self: Self) -> None:
        """Prune removes states with duration below minimum."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0, "idle": 50.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 100.0, "idle": 50.0}
        assert "off" not in stats.durations

    def test_prune_keeps_states_at_or_above_threshold(self: Self) -> None:
        """Prune keeps states with duration at or above minimum."""
        stats = StateStats(durations={"on": 10.0, "off": 10.1, "idle": 100.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 10.0, "off": 10.1, "idle": 100.0}
        # State exactly at threshold is kept

    def test_prune_with_zero_threshold_keeps_all(self: Self) -> None:
        """Prune with threshold 0 removes nothing."""
        stats = StateStats(durations={"on": 0.1, "off": 0.01, "idle": 100.0})
        original = stats.durations.copy()
        stats.prune(min_state_duration=0.0)
        assert stats.durations == original

    def test_prune_empty_durations_safe(self: Self) -> None:
        """Prune on empty durations is safe."""
        stats = StateStats()
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {}

    def test_prune_removes_all_states_when_threshold_high(self: Self) -> None:
        """Prune removes all states when threshold exceeds all durations."""
        stats = StateStats(durations={"on": 10.0, "off": 20.0, "idle": 30.0})
        stats.prune(min_state_duration=100.0)
        assert stats.durations == {}

    def test_prune_single_state_below_threshold(self: Self) -> None:
        """Prune with single state below threshold."""
        stats = StateStats(durations={"on": 5.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {}

    def test_prune_single_state_above_threshold(self: Self) -> None:
        """Prune with single state above threshold."""
        stats = StateStats(durations={"on": 15.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 15.0}


class TestPruneExactBoundary:
    """Test prune behavior at exact threshold boundaries."""

    def test_prune_exact_threshold_kept(self: Self) -> None:
        """State at exact threshold is kept (uses < not <=)."""
        stats = StateStats(durations={"on": 10.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 10.0}

    def test_prune_just_below_threshold(self: Self) -> None:
        """State just below threshold is removed."""
        stats = StateStats(durations={"on": 9.999999})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {}

    def test_prune_just_above_threshold(self: Self) -> None:
        """State just above threshold is kept."""
        stats = StateStats(durations={"on": 10.000001})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 10.000001}


class TestPruneWithDecay:
    """Test prune behavior after decay operations."""

    def test_prune_after_decay_removes_low_weight_states(self: Self) -> None:
        """Prune after decay removes states that decayed below threshold."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0, "idle": 50.0})
        stats.last_update_ts = 0.0

        # After 3 half-lives, durations reduced to 12.5%
        stats.apply_decay(timestamp=3 * 3600.0, half_life=3600.0)

        # on: 12.5, off: 25.0, idle: 6.25
        stats.prune(min_state_duration=10.0)

        assert "idle" not in stats.durations  # 6.25 < 10
        assert "on" in stats.durations  # 12.5 >= 10
        assert "off" in stats.durations  # 25.0 >= 10

    def test_prune_workflow_decay_then_prune(self: Self) -> None:
        """Realistic workflow: decay old data, then prune insignificant states."""
        stats = StateStats(
            durations={"heating": 1000.0, "cooling": 500.0, "idle": 100.0}
        )
        stats.last_update_ts = 0.0

        # Simulate 1 week passing with 1 day half-life
        stats.apply_decay(timestamp=7 * 86400.0, half_life=86400.0)

        # After 7 half-lives: factor = 0.5^7 ≈ 0.0078125
        # heating: ~7.8, cooling: ~3.9, idle: ~0.78
        stats.prune(min_state_duration=5.0)

        # Only heating should remain
        assert list(stats.durations.keys()) == ["heating"]
        assert stats.durations["heating"] == pytest.approx(1000.0 * (0.5**7))


class TestPruneStatePreservation:
    """Test that prune preserves remaining state properties correctly."""

    def test_prune_preserves_exact_durations(self: Self) -> None:
        """Prune doesn't modify durations of kept states."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0, "idle": 50.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations["on"] == 100.0
        assert stats.durations["idle"] == 50.0

    def test_prune_preserves_distribution_ratios(self: Self) -> None:
        """Prune preserves relative probabilities among kept states."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0, "idle": 200.0})

        # Original distribution of on:idle is 1:2
        dist_before = stats.distribution()
        ratio_before = dist_before["on"] / dist_before["idle"]

        stats.prune(min_state_duration=10.0)

        # After pruning off, on:idle ratio should be same
        dist_after = stats.distribution()
        ratio_after = dist_after["on"] / dist_after["idle"]

        assert ratio_before == pytest.approx(ratio_after)

    def test_prune_does_not_modify_last_update_ts(self: Self) -> None:
        """Prune doesn't change last_update_ts."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0})
        stats.last_update_ts = 12345.0

        stats.prune(min_state_duration=10.0)

        assert stats.last_update_ts == 12345.0

    def test_prune_with_none_last_update_ts(self: Self) -> None:
        """Prune works when last_update_ts is None."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0})
        assert stats.last_update_ts is None

        stats.prune(min_state_duration=10.0)

        assert stats.durations == {"on": 100.0}
        assert stats.last_update_ts is None


class TestPruneMultipleStates:
    """Test prune with various state configurations."""

    def test_prune_removes_multiple_states(self: Self) -> None:
        """Prune can remove multiple states in one call."""
        stats = StateStats(
            durations={
                "state1": 100.0,
                "state2": 2.0,
                "state3": 50.0,
                "state4": 3.0,
                "state5": 1.0,
            }
        )
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"state1": 100.0, "state3": 50.0}

    def test_prune_with_many_states(self: Self) -> None:
        """Prune works efficiently with many states."""
        durations = {f"state_{i}": float(i) for i in range(100)}
        stats = StateStats(durations=durations)

        stats.prune(min_state_duration=50.0)

        # Only states 50-99 should remain
        assert len(stats.durations) == 50
        assert all(i >= 50 for i in range(100) if f"state_{i}" in stats.durations)
        assert all(f"state_{i}" not in stats.durations for i in range(50))

    def test_prune_mixed_state_types(self: Self) -> None:
        """Prune works with mixed hashable state types."""
        stats = StateStats(durations={"on": 100.0, 1: 5.0, (0, 0): 50.0, "off": 3.0})
        stats.prune(min_state_duration=10.0)
        assert stats.durations == {"on": 100.0, (0, 0): 50.0}


class TestPruneEdgeCases:
    """Test prune edge cases and corner scenarios."""

    def test_prune_with_negative_durations(self: Self) -> None:
        """Prune handles negative durations (edge case)."""
        stats = StateStats(durations={"on": 100.0, "off": -5.0})
        stats.prune(min_state_duration=10.0)
        # Negative is < 10, should be removed
        assert stats.durations == {"on": 100.0}

    def test_prune_with_zero_duration_states(self: Self) -> None:
        """Prune removes zero-duration states when threshold > 0."""
        stats = StateStats(durations={"on": 100.0, "off": 0.0, "idle": 50.0})
        stats.prune(min_state_duration=0.1)
        assert stats.durations == {"on": 100.0, "idle": 50.0}

    def test_prune_successive_calls_with_different_thresholds(self: Self) -> None:
        """Multiple prune calls with increasing thresholds."""
        stats = StateStats(durations={"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0})

        stats.prune(min_state_duration=15.0)
        assert stats.durations == {"b": 20.0, "c": 30.0, "d": 40.0}

        stats.prune(min_state_duration=25.0)
        assert stats.durations == {"c": 30.0, "d": 40.0}

        stats.prune(min_state_duration=35.0)
        assert stats.durations == {"d": 40.0}

    def test_prune_after_total_becomes_zero(self: Self) -> None:
        """Prune after all states removed results in empty distribution."""
        stats = StateStats(durations={"on": 5.0, "off": 3.0})
        stats.prune(min_state_duration=10.0)

        assert stats.durations == {}
        assert stats.total() == 0.0
        assert stats.distribution() == {}

    def test_prune_very_small_threshold(self: Self) -> None:
        """Prune with very small threshold keeps most states."""
        stats = StateStats(durations={"on": 0.1, "off": 0.01, "idle": 0.001})
        stats.prune(min_state_duration=0.005)
        assert stats.durations == {"on": 0.1, "off": 0.01}

    def test_prune_very_large_threshold(self: Self) -> None:
        """Prune with very large threshold removes all states."""
        stats = StateStats(durations={"on": 1000.0, "off": 2000.0})
        stats.prune(min_state_duration=1e10)
        assert stats.durations == {}


class TestPruneAdaptiveBasicBehavior:
    """Test basic prune_adaptive functionality."""

    def test_prune_adaptive_removes_states_below_relative_threshold(self: Self) -> None:
        """Prune adaptive removes states below epsilon fraction of total."""
        stats = StateStats(durations={"on": 9000.0, "off": 980.0, "idle": 20.0})
        # total = 10000, threshold = max(10000 * 0.003, 20) = 30
        stats.prune_adaptive(epsilon=0.003, absolute_min=20.0)

        assert stats.durations == {"on": 9000.0, "off": 980.0}
        assert "idle" not in stats.durations

    def test_prune_adaptive_removes_states_below_absolute_min(self: Self) -> None:
        """Prune adaptive removes states below absolute minimum."""
        stats = StateStats(durations={"on": 80.0, "off": 15.0, "idle": 5.0})
        # total = 100, threshold = max(100 * 0.003, 20) = 20
        stats.prune_adaptive(epsilon=0.003, absolute_min=20.0)

        assert stats.durations == {"on": 80.0}
        assert "off" not in stats.durations
        assert "idle" not in stats.durations

    def test_prune_adaptive_keeps_states_meeting_both_thresholds(self: Self) -> None:
        """Prune adaptive keeps states meeting both relative and absolute thresholds."""
        stats = StateStats(durations={"on": 1000.0, "off": 500.0, "idle": 100.0})
        # total = 1600, threshold = max(1600 * 0.05, 20) = 80
        stats.prune_adaptive(epsilon=0.05, absolute_min=20.0)

        assert stats.durations == {"on": 1000.0, "off": 500.0, "idle": 100.0}

    def test_prune_adaptive_empty_durations_safe(self: Self) -> None:
        """Prune adaptive on empty durations is safe."""
        stats = StateStats()
        stats.prune_adaptive()
        assert stats.durations == {}

    def test_prune_adaptive_default_parameters(self: Self) -> None:
        """Prune adaptive with default parameters works correctly."""
        stats = StateStats(durations={"on": 10000.0, "off": 25.0, "idle": 15.0})
        # total = 10040, threshold = max(10040 * 0.003, 20) = max(30.12, 20) = 30.12
        stats.prune_adaptive()

        assert "on" in stats.durations
        assert "off" not in stats.durations
        assert "idle" not in stats.durations


class TestPruneAdaptiveThresholdCalculation:
    """Test adaptive threshold calculation logic."""

    def test_prune_adaptive_relative_threshold_dominates_large_dataset(
        self: Self,
    ) -> None:
        """With large total, relative threshold (epsilon) dominates."""
        stats = StateStats(durations={"on": 9000.0, "off": 500.0, "idle": 500.0})
        # total = 10000, epsilon threshold = 10000 * 0.01 = 100
        # absolute = 20, max(100, 20) = 100
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        # idle (500) >= 100, kept; both others also kept
        assert len(stats.durations) == 3

    def test_prune_adaptive_absolute_threshold_dominates_small_dataset(
        self: Self,
    ) -> None:
        """With small total, absolute minimum dominates."""
        stats = StateStats(durations={"on": 100.0, "off": 25.0, "idle": 15.0})
        # total = 140, epsilon threshold = 140 * 0.01 = 1.4
        # absolute = 30, max(1.4, 30) = 30
        stats.prune_adaptive(epsilon=0.01, absolute_min=30.0)

        assert stats.durations == {"on": 100.0}

    def test_prune_adaptive_zero_epsilon_uses_absolute_only(self: Self) -> None:
        """Zero epsilon means only absolute threshold applies."""
        stats = StateStats(durations={"on": 100.0, "off": 50.0, "idle": 25.0})
        # threshold = max(190 * 0, 30) = 30
        stats.prune_adaptive(epsilon=0.0, absolute_min=30.0)

        assert stats.durations == {"on": 100.0, "off": 50.0}
        assert "idle" not in stats.durations

    def test_prune_adaptive_zero_absolute_min_uses_relative_only(self: Self) -> None:
        """Zero absolute_min means only relative threshold applies."""
        stats = StateStats(durations={"on": 100.0, "off": 6.0, "idle": 2.0})
        # total = 108, threshold = max(108 * 0.05, 0) = 5.4
        stats.prune_adaptive(epsilon=0.05, absolute_min=0.0)

        assert stats.durations == {"on": 100.0, "off": 6.0}
        assert "idle" not in stats.durations

    def test_prune_adaptive_at_exact_threshold(self: Self) -> None:
        """State at exact threshold is kept (uses >=)."""
        stats = StateStats(durations={"on": 100.0, "off": 30.0})
        # total = 130, threshold = max(130 * 0.01, 30) = 30
        stats.prune_adaptive(epsilon=0.01, absolute_min=30.0)

        assert stats.durations == {"on": 100.0, "off": 30.0}

    def test_prune_adaptive_just_below_threshold(self: Self) -> None:
        """State just below threshold is removed."""
        stats = StateStats(durations={"on": 100.0, "off": 29.999})
        # threshold = max(..., 30) = 30
        stats.prune_adaptive(epsilon=0.01, absolute_min=30.0)

        assert stats.durations == {"on": 100.0}
        assert "off" not in stats.durations


class TestPruneAdaptiveWithDecay:
    """Test prune_adaptive interaction with decay functionality."""

    def test_prune_adaptive_after_decay_scales_automatically(self: Self) -> None:
        """Prune adaptive automatically adjusts to decayed totals."""
        stats = StateStats(durations={"heating": 500.0, "cooling": 300.0, "idle": 50.0})
        stats.last_update_ts = 0.0

        # After 2 half-lives: all x 0.25
        stats.apply_decay(timestamp=86400.0, half_life=43200.0)

        # New totals: heating=125, cooling=75, idle=12.5, total=212.5
        # threshold = max(212.5 * 0.01, 15) = max(2.125, 15) = 15
        stats.prune_adaptive(epsilon=0.01, absolute_min=15.0)

        assert "heating" in stats.durations
        assert "cooling" in stats.durations
        assert "idle" not in stats.durations

    def test_prune_adaptive_decay_workflow_realistic(self: Self) -> None:
        """Realistic workflow: decay old data, then adaptive prune."""
        stats = StateStats(
            durations={"on": 10000.0, "off": 5000.0, "idle": 1000.0, "error": 200.0}
        )
        stats.last_update_ts = 0.0

        # After 1 week with 1-day half-life: 7 half-lives
        stats.apply_decay(timestamp=7 * 86400.0, half_life=86400.0)

        # Factor = 0.5^7 ≈ 0.0078125
        # on: ~78, off: ~39, idle: ~7.8, error: ~1.56
        # total ≈ 126.36
        stats.prune_adaptive(epsilon=0.01, absolute_min=10.0)

        # threshold = max(126.36 * 0.01, 10) = 10
        # idle (~7.8) and error (~1.56) removed
        assert set(stats.durations.keys()) == {"on", "off"}


class TestPruneAdaptiveStatePreservation:
    """Test that prune_adaptive preserves remaining state properties correctly."""

    def test_prune_adaptive_preserves_exact_durations(self: Self) -> None:
        """Prune adaptive doesn't modify durations of kept states."""
        stats = StateStats(durations={"on": 1000.0, "off": 50.0, "idle": 10.0})
        stats.prune_adaptive(epsilon=0.01, absolute_min=40.0)

        assert stats.durations["on"] == 1000.0
        assert stats.durations["off"] == 50.0

    def test_prune_adaptive_preserves_distribution_ratios(self: Self) -> None:
        """Prune adaptive preserves relative probabilities among kept states."""
        stats = StateStats(durations={"on": 200.0, "off": 100.0, "idle": 5.0})

        # Original ratio of on:off is 2:1
        dist_before = stats.distribution()
        ratio_before = dist_before["on"] / dist_before["off"]

        stats.prune_adaptive(epsilon=0.01, absolute_min=10.0)

        # After pruning idle, on:off ratio should be same
        dist_after = stats.distribution()
        ratio_after = dist_after["on"] / dist_after["off"]

        assert ratio_before == pytest.approx(ratio_after)

    def test_prune_adaptive_does_not_modify_last_update_ts(self: Self) -> None:
        """Prune adaptive doesn't change last_update_ts."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0})
        stats.last_update_ts = 12345.0

        stats.prune_adaptive()

        assert stats.last_update_ts == 12345.0

    def test_prune_adaptive_with_none_last_update_ts(self: Self) -> None:
        """Prune adaptive works when last_update_ts is None."""
        stats = StateStats(durations={"on": 100.0, "off": 5.0})
        assert stats.last_update_ts is None

        stats.prune_adaptive(epsilon=0.1, absolute_min=10.0)

        assert stats.last_update_ts is None


class TestPruneAdaptiveMultipleStates:
    """Test prune_adaptive with various state configurations."""

    def test_prune_adaptive_removes_multiple_states(self: Self) -> None:
        """Prune adaptive can remove multiple states in one call."""
        stats = StateStats(
            durations={
                "state1": 1000.0,
                "state2": 20.0,
                "state3": 500.0,
                "state4": 15.0,
                "state5": 8.0,
            }
        )
        # total = 1543, threshold = max(1543 * 0.01, 20) = max(15.43, 20) = 20
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        assert set(stats.durations.keys()) == {"state1", "state2", "state3"}

    def test_prune_adaptive_with_many_states(self: Self) -> None:
        """Prune adaptive works efficiently with many states."""
        durations = {f"state_{i}": float(i * 10) for i in range(100)}
        stats = StateStats(durations=durations)

        # total = 0+10+20+...+990 = 49500
        # threshold = max(49500 * 0.01, 20) = 495
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        # Only states >= 495 kept: state_50 onwards (500, 510, ..., 990)
        kept_states = list(stats.durations)
        assert all(int(s.split("_")[1]) >= 50 for s in kept_states)
        assert len(kept_states) == 50

    def test_prune_adaptive_mixed_state_types(self: Self) -> None:
        """Prune adaptive works with mixed hashable state types."""
        stats = StateStats(durations={"on": 100.0, 1: 50.0, (0, 0): 25.0, "off": 10.0})
        # total = 185, threshold = max(185 * 0.05, 20) = 20
        stats.prune_adaptive(epsilon=0.05, absolute_min=20.0)

        assert "on" in stats.durations
        assert 1 in stats.durations
        assert (0, 0) in stats.durations
        assert "off" not in stats.durations


class TestPruneAdaptiveEdgeCases:
    """Test prune_adaptive edge cases and corner scenarios."""

    def test_prune_adaptive_all_states_removed(self: Self) -> None:
        """All states can be removed if none meet threshold."""
        stats = StateStats(durations={"on": 5.0, "off": 3.0, "idle": 2.0})
        # total = 10, threshold = max(10 * 0.01, 15) = 15
        stats.prune_adaptive(epsilon=0.01, absolute_min=15.0)

        assert stats.durations == {}

    def test_prune_adaptive_single_state_kept(self: Self) -> None:
        """Single state above threshold is kept."""
        stats = StateStats(durations={"on": 100.0})
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        assert stats.durations == {"on": 100.0}

    def test_prune_adaptive_single_state_removed(self: Self) -> None:
        """Single state below threshold is removed."""
        stats = StateStats(durations={"on": 10.0})
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        assert stats.durations == {}

    def test_prune_adaptive_very_small_epsilon(self: Self) -> None:
        """Very small epsilon keeps more states."""
        stats = StateStats(durations={"on": 1000.0, "off": 10.0, "idle": 5.0})
        # total = 1015, threshold = max(1015 * 0.001, 5) = max(1.015, 5) = 5
        stats.prune_adaptive(epsilon=0.001, absolute_min=5.0)

        assert set(stats.durations.keys()) == {"on", "off", "idle"}

    def test_prune_adaptive_very_large_epsilon(self: Self) -> None:
        """Very large epsilon removes most states."""
        stats = StateStats(durations={"on": 100.0, "off": 50.0, "idle": 30.0})
        # total = 180, threshold = max(180 * 0.5, 20) = 90
        stats.prune_adaptive(epsilon=0.5, absolute_min=20.0)

        assert stats.durations == {"on": 100.0}

    def test_prune_adaptive_successive_calls(self: Self) -> None:
        """Multiple prune_adaptive calls progressively remove states."""
        stats = StateStats(durations={"a": 1000.0, "b": 100.0, "c": 50.0, "d": 25.0})

        # First prune: threshold = max(1175 * 0.02, 20) = 23.5
        stats.prune_adaptive(epsilon=0.02, absolute_min=20.0)
        assert set(stats.durations.keys()) == {"a", "b", "c", "d"}

        # Second prune: threshold = max(1175 * 0.04, 40) = 47
        stats.prune_adaptive(epsilon=0.04, absolute_min=40.0)
        assert set(stats.durations.keys()) == {"a", "b", "c"}

        # Third prune: threshold = max(1150 * 0.1, 50) = 115
        stats.prune_adaptive(epsilon=0.1, absolute_min=50.0)
        assert set(stats.durations.keys()) == {"a"}

    def test_prune_adaptive_after_total_becomes_zero(self: Self) -> None:
        """Prune adaptive handles zero total correctly."""
        stats = StateStats(durations={"on": 5.0})
        stats.prune_adaptive(epsilon=0.01, absolute_min=10.0)

        assert stats.durations == {}
        assert stats.total() == 0.0
        assert stats.distribution() == {}

    def test_prune_adaptive_with_negative_durations(self: Self) -> None:
        """Prune adaptive handles negative durations (edge case)."""
        stats = StateStats(durations={"on": 100.0, "off": -5.0})
        # total = 95, threshold = max(95 * 0.01, 20) = 20
        stats.prune_adaptive(epsilon=0.01, absolute_min=20.0)

        # Negative duration < 20, removed
        assert stats.durations == {"on": 100.0}

    def test_prune_adaptive_comparison_with_fixed_prune(self: Self) -> None:
        """Prune adaptive behaves differently than fixed prune."""
        stats1 = StateStats(durations={"on": 1000.0, "off": 50.0, "idle": 25.0})
        stats2 = StateStats(durations={"on": 1000.0, "off": 50.0, "idle": 25.0})

        # Fixed prune with threshold 30
        stats1.prune(min_state_duration=30.0)

        # Adaptive prune that should give different result
        # total = 1075, threshold = max(1075 * 0.04, 30) = 43
        stats2.prune_adaptive(epsilon=0.04, absolute_min=30.0)

        # Fixed keeps on and off (both >= 30)
        assert set(stats1.durations.keys()) == {"on", "off"}

        # Adaptive keeps only on (off's 50 < 43 is false, so off kept)
        assert set(stats2.durations.keys()) == {"on", "off"}

        # Use different threshold to show difference
        stats3 = StateStats(durations={"on": 1000.0, "off": 50.0, "idle": 25.0})
        # threshold = max(1075 * 0.05, 30) = 53.75
        stats3.prune_adaptive(epsilon=0.05, absolute_min=30.0)
        assert set(stats3.durations.keys()) == {"on"}


class TestCheckDriftBasicBehavior:
    """Test basic drift detection functionality."""

    def test_check_drift_establishes_baseline_first_call(self: Self) -> None:
        """First call to check_drift establishes baseline without detecting drift."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})

        drift = stats.check_drift(now_ts=1000.0)

        assert drift is False
        assert stats.baseline == {"on": 0.75, "off": 0.25}

    def test_check_drift_detects_significant_change(self: Self) -> None:
        """Detects drift when distribution changes significantly."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 0.0

        # Change pattern: now mostly 'off'
        stats.durations = {"on": 500.0, "off": 3500.0}

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is True
        assert stats.last_drift_ts == 5000.0
        # Baseline updated to new pattern
        assert stats.baseline == pytest.approx({"on": 0.125, "off": 0.875})

    def test_check_drift_no_detection_for_small_changes(self: Self) -> None:
        """Does not detect drift for small distribution changes."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 0.0

        # Small change: 73% vs 75%
        stats.durations = {"on": 2900.0, "off": 1100.0}

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is False

    def test_check_drift_returns_false_for_insufficient_data(self: Self) -> None:
        """Returns False when total duration below min_support."""
        stats = StateStats(durations={"on": 100.0, "off": 50.0})

        drift = stats.check_drift(now_ts=1000.0, min_support=1000.0)

        assert drift is False
        # Baseline not established due to insufficient data
        assert stats.baseline is None

    def test_check_drift_respects_cooldown_period(self: Self) -> None:
        """Does not detect drift during cooldown period."""
        stats = StateStats(durations={"on": 1000.0, "off": 3000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 1000.0

        # Significant change but within cooldown
        stats.durations = {"on": 500.0, "off": 3500.0}

        drift = stats.check_drift(now_ts=1500.0, cooldown=3600.0)

        assert drift is False
        # Baseline not updated
        assert stats.baseline == {"on": 0.75, "off": 0.25}

    def test_check_drift_detects_after_cooldown_expires(self: Self) -> None:
        """Detects drift after cooldown period expires."""
        stats = StateStats(durations={"on": 1000.0, "off": 3000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 1000.0

        # Significant change after cooldown
        stats.durations = {"on": 500.0, "off": 3500.0}

        drift = stats.check_drift(now_ts=5000.0, cooldown=3600.0)

        assert drift is True


class TestCheckDriftThresholdSensitivity:
    """Test drift detection with different threshold values."""

    def test_check_drift_high_threshold_less_sensitive(self: Self) -> None:
        """High threshold requires larger changes to detect drift."""
        stats = StateStats(durations={"on": 2000.0, "off": 2000.0})
        stats.baseline = {"on": 0.7, "off": 0.3}
        stats.last_drift_ts = 0.0

        # Moderate change: 50/50 from 70/30
        drift = stats.check_drift(now_ts=5000.0, threshold=0.5)

        assert drift is False  # JS divergence < 0.5

    def test_check_drift_low_threshold_more_sensitive(self: Self) -> None:
        """Low threshold detects smaller distribution changes."""
        stats = StateStats(durations={"on": 2000.0, "off": 2000.0})
        stats.baseline = {"on": 0.7, "off": 0.3}
        stats.last_drift_ts = 0.0

        # Larger change: 50/50 from 70/30
        drift = stats.check_drift(
            now_ts=5000.0, min_support=1000.0, threshold=0.02, cooldown=1000.0
        )

        assert drift is True  # Low threshold catches moderate changes

    def test_check_drift_zero_threshold_detects_any_change(self: Self) -> None:
        """Zero threshold detects any distribution change."""
        stats = StateStats(durations={"on": 3001.0, "off": 1000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 0.0

        # Tiny change
        drift = stats.check_drift(now_ts=5000.0, threshold=0.0)

        assert drift is True


class TestCheckDriftWithNewStates:
    """Test drift detection when states appear or disappear."""

    def test_check_drift_detects_new_state_appearance(self: Self) -> None:
        """Detects drift when a new state appears."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 0.0

        # New state 'idle' appears with significant proportion
        stats.durations = {"on": 2000.0, "off": 500.0, "idle": 1500.0}

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is True
        assert "idle" in stats.baseline

    def test_check_drift_detects_state_disappearance(self: Self) -> None:
        """Detects drift when a state disappears."""
        stats = StateStats(durations={"on": 2000.0, "off": 1000.0, "idle": 1000.0})
        stats.baseline = {"on": 0.5, "off": 0.25, "idle": 0.25}
        stats.last_drift_ts = 0.0

        # 'idle' state removed
        stats.durations = {"on": 3500.0, "off": 500.0}

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is True
        assert "idle" not in stats.baseline

    def test_check_drift_with_completely_different_states(self: Self) -> None:
        """Detects drift when states are completely replaced."""
        stats = StateStats(durations={"heating": 3000.0, "cooling": 1000.0})
        stats.baseline = {"heating": 0.75, "cooling": 0.25}
        stats.last_drift_ts = 0.0

        # Completely different states
        stats.durations = {"on": 3000.0, "off": 1000.0}

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is True
        assert set(stats.baseline.keys()) == {"on", "off"}


class TestCheckDriftMinSupport:
    """Test drift detection with minimum support requirements."""

    def test_check_drift_default_min_support(self: Self) -> None:
        """Default min_support is 3600 seconds."""
        stats = StateStats(durations={"on": 3500.0, "off": 100.0})

        drift = stats.check_drift(now_ts=1000.0)

        assert drift is False  # 3600 < 3600 (default)

    def test_check_drift_meets_min_support(self: Self) -> None:
        """Establishes baseline when meeting min_support."""
        stats = StateStats(durations={"on": 3600.0, "off": 100.0})

        drift = stats.check_drift(now_ts=1000.0, min_support=3600.0)

        assert drift is False  # First call establishes baseline
        assert stats.baseline is not None

    def test_check_drift_custom_min_support(self: Self) -> None:
        """Custom min_support allows earlier drift detection."""
        stats = StateStats(durations={"on": 1500.0, "off": 500.0})

        drift = stats.check_drift(now_ts=1000.0, min_support=1000.0)

        assert drift is False
        assert stats.baseline is not None  # Baseline established


class TestCheckDriftCooldownBehavior:
    """Test cooldown period behavior in drift detection."""

    def test_check_drift_default_cooldown(self: Self) -> None:
        """Default cooldown is 3600 seconds."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})
        stats.baseline = {"on": 0.2, "off": 0.8}
        stats.last_drift_ts = 1000.0

        # Significant change but 3500s < 3600s default cooldown
        drift = stats.check_drift(now_ts=4500.0)

        assert drift is False

    def test_check_drift_custom_short_cooldown(self: Self) -> None:
        """Short cooldown allows frequent drift detection."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})
        stats.baseline = {"on": 0.2, "off": 0.8}
        stats.last_drift_ts = 1000.0

        # Significant change after short cooldown
        drift = stats.check_drift(now_ts=1100.0, cooldown=60.0)

        assert drift is True

    def test_check_drift_zero_cooldown_allows_immediate_detection(self: Self) -> None:
        """Zero cooldown allows immediate successive detections."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})
        stats.baseline = {"on": 0.2, "off": 0.8}
        stats.last_drift_ts = 1000.0

        drift = stats.check_drift(now_ts=1000.0, cooldown=0.0)

        assert drift is True

    def test_check_drift_cooldown_at_exact_boundary(self: Self) -> None:
        """Detects drift at exact cooldown boundary."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})
        stats.baseline = {"on": 0.2, "off": 0.8}
        stats.last_drift_ts = 1000.0

        # Exactly at cooldown boundary
        drift = stats.check_drift(now_ts=4600.0, cooldown=3600.0)

        assert drift is True


class TestCheckDriftSuccessiveDetections:
    """Test multiple successive drift detections."""

    def test_check_drift_successive_detections_update_baseline(self: Self) -> None:
        """Successive drift detections update baseline each time."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})

        # First detection establishes baseline
        drift1 = stats.check_drift(now_ts=1000.0)
        assert drift1 is False
        baseline1 = stats.baseline.copy()

        # Change pattern
        stats.durations = {"on": 500.0, "off": 3500.0}
        drift2 = stats.check_drift(now_ts=5000.0, threshold=0.15)
        assert drift2 is True
        baseline2 = stats.baseline.copy()

        # Baseline updated
        assert baseline1 != baseline2

        # Change pattern again - larger change from 12.5/87.5
        stats.durations = {"on": 3500.0, "off": 500.0}
        drift3 = stats.check_drift(now_ts=10000.0, threshold=0.15)
        assert drift3 is True

        # Baseline updated again
        assert stats.baseline != baseline2

    def test_check_drift_resets_last_drift_ts_on_detection(self: Self) -> None:
        """Drift detection updates last_drift_ts."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
        stats.baseline = {"on": 0.2, "off": 0.8}
        stats.last_drift_ts = 0.0

        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is True
        assert stats.last_drift_ts == 5000.0


class TestJSDivergence:
    """Test Jensen-Shannon divergence calculation."""

    def test_js_divergence_identical_distributions(self: Self) -> None:
        """JS divergence is 0 for identical distributions."""
        stats = StateStats()
        p = {"on": 0.7, "off": 0.3}
        q = {"on": 0.7, "off": 0.3}

        js = stats._js_divergence(p, q)

        assert js == pytest.approx(0.0, abs=1e-10)

    def test_js_divergence_symmetric(self: Self) -> None:
        """JS divergence is symmetric: JS(P,Q) = JS(Q,P)."""
        stats = StateStats()
        p = {"on": 0.7, "off": 0.3}
        q = {"on": 0.4, "off": 0.6}

        js_pq = stats._js_divergence(p, q)
        js_qp = stats._js_divergence(q, p)

        assert js_pq == pytest.approx(js_qp)

    def test_js_divergence_moderate_difference(self: Self) -> None:
        """JS divergence for moderate distribution difference."""
        stats = StateStats()
        p = {"on": 0.7, "off": 0.3}
        q = {"on": 0.5, "off": 0.5}

        js = stats._js_divergence(p, q)

        # Should be small but measurable
        assert 0.0 < js < 0.2

    def test_js_divergence_large_difference(self: Self) -> None:
        """JS divergence for large distribution difference."""
        stats = StateStats()
        p = {"on": 0.9, "off": 0.1}
        q = {"on": 0.1, "off": 0.9}

        js = stats._js_divergence(p, q)

        # Should be substantial
        assert js > 0.3

    def test_js_divergence_completely_different_states(self: Self) -> None:
        """JS divergence is maximum for completely different states."""
        stats = StateStats()
        p = {"on": 1.0}
        q = {"off": 1.0}

        js = stats._js_divergence(p, q)

        # Should be close to maximum (1.0)
        assert js > 0.9

    def test_js_divergence_handles_missing_states(self: Self) -> None:
        """JS divergence handles distributions with different state support."""
        stats = StateStats()
        p = {"on": 0.5, "off": 0.5}
        q = {"on": 0.3, "off": 0.3, "idle": 0.4}

        js = stats._js_divergence(p, q)

        # Should compute without error
        assert 0.0 < js < 1.0

    def test_js_divergence_with_very_small_probabilities(self: Self) -> None:
        """JS divergence handles very small probabilities via smoothing."""
        stats = StateStats()
        p = {"on": 0.99999, "off": 0.00001}
        q = {"on": 0.00001, "off": 0.99999}

        js = stats._js_divergence(p, q)

        # Should compute large divergence (nearly opposite distributions)
        assert js > 0.5

    def test_js_divergence_three_states(self: Self) -> None:
        """JS divergence works with three states."""
        stats = StateStats()
        p = {"on": 0.5, "off": 0.3, "idle": 0.2}
        q = {"on": 0.3, "off": 0.5, "idle": 0.2}

        js = stats._js_divergence(p, q)

        # Small difference (only on/off swapped)
        assert 0.0 < js < 0.1

    def test_js_divergence_many_states(self: Self) -> None:
        """JS divergence works with many states."""
        stats = StateStats()
        # Uniform distribution over 10 states
        p = {f"state_{i}": 0.1 for i in range(10)}
        # Skewed distribution
        q = {"state_0": 0.5}
        q.update({f"state_{i}": 0.5 / 9 for i in range(1, 10)})

        js = stats._js_divergence(p, q)

        # Should show significant divergence
        assert js > 0.1


class TestCheckDriftEdgeCases:
    """Test edge cases in drift detection."""

    def test_check_drift_empty_durations(self: Self) -> None:
        """Handles empty durations gracefully."""
        stats = StateStats()

        drift = stats.check_drift(now_ts=1000.0, min_support=0.0)

        assert drift is False

    def test_check_drift_single_state(self: Self) -> None:
        """Handles single state distribution."""
        stats = StateStats(durations={"on": 4000.0})

        drift = stats.check_drift(now_ts=1000.0)

        assert drift is False
        assert stats.baseline == {"on": 1.0}

    def test_check_drift_after_prune_removes_all_states(self: Self) -> None:
        """Handles case where pruning removes all states."""
        stats = StateStats(durations={"on": 10.0, "off": 5.0})
        stats.baseline = {"on": 0.8, "off": 0.2}
        stats.last_drift_ts = 0.0

        # Prune all states
        stats.prune(min_state_duration=100.0)

        # Empty durations creates empty distribution
        drift = stats.check_drift(now_ts=5000.0, min_support=0.0)

        # When current distribution is empty but baseline exists,
        # drift is detected (distribution changed from baseline to empty)
        assert drift is True
        # Baseline updated to empty
        assert stats.baseline == {}

    def test_check_drift_negative_timestamp(self: Self) -> None:
        """Handles negative timestamps."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})

        drift = stats.check_drift(now_ts=-1000.0)

        assert drift is False
        assert stats.baseline is not None

    def test_check_drift_very_large_divergence(self: Self) -> None:
        """Handles very large divergence values."""
        stats = StateStats(durations={"on": 4000.0})
        stats.baseline = {"off": 1.0}
        stats.last_drift_ts = 0.0

        drift = stats.check_drift(now_ts=5000.0, threshold=0.5)

        assert drift is True  # Maximum divergence

    def test_check_drift_preserves_baseline_when_no_drift(self: Self) -> None:
        """Baseline unchanged when drift not detected."""
        stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 0.0

        # Same distribution
        drift = stats.check_drift(now_ts=5000.0, threshold=0.15)

        assert drift is False
        assert stats.baseline == {"on": 0.75, "off": 0.25}


class TestStateStatsSerialization:
    """Tests for StateStats serialization (to_dict/from_dict)."""

    def test_to_dict_empty_stats(self: Self) -> None:
        """Test serializing an empty StateStats."""
        stats = StateStats()
        data = stats.to_dict()

        assert data["durations"] == {}
        assert data["last_update_ts"] is None
        assert data["baseline"] is None
        assert data["last_drift_ts"] == 0.0
        assert data["fast_decay_updates"] == 0

    def test_to_dict_with_durations(self: Self) -> None:
        """Test serializing StateStats with durations."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        data = stats.to_dict()

        assert data["durations"] == {"on": 100.0, "off": 200.0}
        assert data["last_update_ts"] is None
        assert data["baseline"] is None

    def test_to_dict_with_all_fields(self: Self) -> None:
        """Test serializing StateStats with all fields populated."""
        stats = StateStats(durations={"on": 300.0, "off": 100.0})
        stats.last_update_ts = 1234567890.0
        stats.baseline = {"on": 0.75, "off": 0.25}
        stats.last_drift_ts = 1234567800.0
        stats.fast_decay_updates = 5

        data = stats.to_dict()

        assert data["durations"] == {"on": 300.0, "off": 100.0}
        assert data["last_update_ts"] == 1234567890.0
        assert data["baseline"] == {"on": 0.75, "off": 0.25}
        assert data["last_drift_ts"] == 1234567800.0
        assert data["fast_decay_updates"] == 5

    def test_to_dict_baseline_is_none(self: Self) -> None:
        """Test that None baseline is preserved in serialization."""
        stats = StateStats(durations={"on": 100.0})
        stats.baseline = None

        data = stats.to_dict()

        assert data["baseline"] is None

    def test_from_dict_empty_data(self: Self) -> None:
        """Test deserializing empty StateStats."""
        data = {
            "durations": {},
            "last_update_ts": None,
            "baseline": None,
            "last_drift_ts": 0.0,
            "fast_decay_updates": 0,
        }

        stats = StateStats.from_dict(data)

        assert stats.durations == {}
        assert stats.last_update_ts is None
        assert stats.baseline is None
        assert stats.last_drift_ts == 0.0
        assert stats.fast_decay_updates == 0

    def test_from_dict_with_durations(self: Self) -> None:
        """Test deserializing StateStats with durations."""
        data = {
            "durations": {"on": 150.0, "off": 250.0},
            "last_update_ts": None,
            "baseline": None,
            "last_drift_ts": 0.0,
            "fast_decay_updates": 0,
        }

        stats = StateStats.from_dict(data)

        assert stats.durations == {"on": 150.0, "off": 250.0}

    def test_from_dict_with_all_fields(self: Self) -> None:
        """Test deserializing StateStats with all fields."""
        data = {
            "durations": {"heating": 400.0, "cooling": 200.0, "idle": 100.0},
            "last_update_ts": 9876543210.0,
            "baseline": {"heating": 0.6, "cooling": 0.3, "idle": 0.1},
            "last_drift_ts": 9876543100.0,
            "fast_decay_updates": 12,
        }

        stats = StateStats.from_dict(data)

        assert stats.durations == {"heating": 400.0, "cooling": 200.0, "idle": 100.0}
        assert stats.last_update_ts == 9876543210.0
        assert stats.baseline == {"heating": 0.6, "cooling": 0.3, "idle": 0.1}
        assert stats.last_drift_ts == 9876543100.0
        assert stats.fast_decay_updates == 12

    def test_from_dict_missing_optional_fields(self: Self) -> None:
        """Test deserializing with missing optional fields uses defaults."""
        data = {"durations": {"on": 100.0}}

        stats = StateStats.from_dict(data)

        assert stats.durations == {"on": 100.0}
        assert stats.last_update_ts is None
        assert stats.baseline is None
        assert stats.last_drift_ts == 0.0
        assert stats.fast_decay_updates == 0

    def test_roundtrip_empty_stats(self: Self) -> None:
        """Test round-trip serialization of empty StateStats."""
        original = StateStats()
        data = original.to_dict()
        restored = StateStats.from_dict(data)

        assert restored.durations == original.durations
        assert restored.last_update_ts == original.last_update_ts
        assert restored.baseline == original.baseline
        assert restored.last_drift_ts == original.last_drift_ts
        assert restored.fast_decay_updates == original.fast_decay_updates

    def test_roundtrip_with_durations(self: Self) -> None:
        """Test round-trip serialization with durations."""
        original = StateStats(durations={"on": 123.45, "off": 678.90})
        data = original.to_dict()
        restored = StateStats.from_dict(data)

        assert restored.durations == original.durations
        assert restored.total() == original.total()

    def test_roundtrip_with_all_fields(self: Self) -> None:
        """Test round-trip serialization with all fields populated."""
        original = StateStats(durations={"on": 500.0, "off": 300.0})
        original.last_update_ts = 1609459200.0
        original.baseline = {"on": 0.625, "off": 0.375}
        original.last_drift_ts = 1609459100.0
        original.fast_decay_updates = 7

        data = original.to_dict()
        restored = StateStats.from_dict(data)

        assert restored.durations == original.durations
        assert restored.last_update_ts == original.last_update_ts
        assert restored.baseline == original.baseline
        assert restored.last_drift_ts == original.last_drift_ts
        assert restored.fast_decay_updates == original.fast_decay_updates

    def test_roundtrip_preserves_distribution(self: Self) -> None:
        """Test that distribution is preserved after round-trip."""
        original = StateStats(durations={"on": 700.0, "off": 300.0})
        original_dist = original.distribution()

        data = original.to_dict()
        restored = StateStats.from_dict(data)
        restored_dist = restored.distribution()

        assert restored_dist == original_dist

    def test_serialized_data_is_json_compatible(self: Self) -> None:
        """Test that serialized data can be JSON-encoded."""
        stats = StateStats(durations={"on": 100.0, "off": 200.0})
        stats.last_update_ts = 1234567890.0
        stats.baseline = {"on": 0.33, "off": 0.67}

        data = stats.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Verify we can restore from parsed JSON
        restored = StateStats.from_dict(parsed)
        assert restored.durations == stats.durations
        assert restored.last_update_ts == stats.last_update_ts
        assert restored.baseline == stats.baseline

    def test_roundtrip_after_decay(self: Self) -> None:
        """Test serialization works correctly after decay operations."""
        stats = StateStats(durations={"on": 1000.0, "off": 500.0})
        stats.last_update_ts = 0.0
        stats.apply_decay(timestamp=3600.0, half_life=3600.0)

        # After one half-life, durations should be halved
        data = stats.to_dict()
        restored = StateStats.from_dict(data)

        assert restored.durations["on"] == pytest.approx(500.0)
        assert restored.durations["off"] == pytest.approx(250.0)
        assert restored.last_update_ts == 3600.0

    def test_roundtrip_after_drift_detection(self: Self) -> None:
        """Test serialization preserves drift detection state."""
        stats = StateStats(durations={"on": 4000.0, "off": 1000.0})
        stats.check_drift(now_ts=1000.0)  # Establish baseline

        # Change distribution
        stats.durations = {"on": 1000.0, "off": 4000.0}
        stats.check_drift(now_ts=5000.0, threshold=0.15)  # Detect drift

        data = stats.to_dict()
        restored = StateStats.from_dict(data)

        assert restored.baseline == stats.baseline
        assert restored.last_drift_ts == stats.last_drift_ts

    def test_multiple_stats_serialization(self: Self) -> None:
        """Test serializing multiple different StateStats instances."""
        stats_list = [
            StateStats(),
            StateStats(durations={"on": 100.0}),
            StateStats(durations={"on": 200.0, "off": 300.0, "idle": 50.0}),
        ]

        # Serialize all
        serialized = [s.to_dict() for s in stats_list]

        # Deserialize all
        restored = [StateStats.from_dict(data) for data in serialized]

        # Verify all match
        for original, restored_stats in zip(stats_list, restored, strict=True):
            assert restored_stats.durations == original.durations
            assert restored_stats.total() == original.total()
