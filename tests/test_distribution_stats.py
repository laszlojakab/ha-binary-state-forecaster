"""
Unit tests for DistributionStats.

Comprehensive tests for the DistributionStats class, covering distribution
management, support aggregation, probability calculations, entropy, and decay.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)


class TestDistributionStatsInitialization:
    """Tests for DistributionStats initialization."""

    def test_create_empty_distribution(self: Self) -> None:
        """Test creating an empty distribution."""
        dist = DistributionStats()
        assert dist.total_support() == 0.0
        assert dist.is_empty()

    def test_empty_states_set(self: Self) -> None:
        """Test empty distribution has no states."""
        dist = DistributionStats()
        assert dist.states() == set()

    def test_empty_distribution_dict(self: Self) -> None:
        """Test empty distribution returns empty dict."""
        dist = DistributionStats()
        assert dist.distribution() == {}


class TestDistributionStatsUpdate:
    """Tests for DistributionStats.update method."""

    def test_update_single_state(self: Self) -> None:
        """Test updating a single state."""
        dist = DistributionStats()
        dist.update("on", 1.0)
        assert dist.support("on") == 1.0
        assert dist.total_support() == 1.0

    def test_update_multiple_states(self: Self) -> None:
        """Test updating multiple different states."""
        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("off", 3.0)
        assert dist.total_support() == 5.0
        assert dist.support("on") == 2.0
        assert dist.support("off") == 3.0

    def test_update_same_state_twice(self: Self) -> None:
        """Test accumulating support for same state."""
        dist = DistributionStats()
        dist.update("on", 1.0)
        dist.update("on", 2.0)
        assert dist.support("on") == 3.0

    def test_update_default_weight(self: Self) -> None:
        """Test update with default weight 1.0."""
        dist = DistributionStats()
        dist.update("on")
        assert dist.support("on") == 1.0

    def test_update_zero_weight(self: Self) -> None:
        """Test update with zero weight."""
        dist = DistributionStats()
        dist.update("on", 0.0)
        assert dist.support("on") == 0.0

    def test_update_fractional_weights(self: Self) -> None:
        """Test updates with fractional weights."""
        dist = DistributionStats()
        dist.update("a", 0.5)
        dist.update("b", 0.3)
        dist.update("c", 0.2)
        assert abs(dist.total_support() - 1.0) < 1e-9

    def test_update_creates_new_states(self: Self) -> None:
        """Test that update creates entries for new states."""
        dist = DistributionStats()
        dist.update("state1")
        dist.update("state2")
        dist.update("state3")
        assert len(dist.states()) == 3


class TestDistributionStatsTotalSupport:
    """Tests for DistributionStats.total_support method."""

    def test_total_support_empty(self: Self) -> None:
        """Test total_support on empty distribution."""
        dist = DistributionStats()
        assert dist.total_support() == 0.0

    def test_total_support_single_state(self: Self) -> None:
        """Test total_support with single state."""
        dist = DistributionStats()
        dist.update("on", 5.0)
        assert dist.total_support() == 5.0

    def test_total_support_multiple_states(self: Self) -> None:
        """Test total_support aggregates all states."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 2.0)
        dist.update("c", 3.0)
        assert dist.total_support() == 6.0

    def test_total_support_fractional(self: Self) -> None:
        """Test total_support with fractional values."""
        dist = DistributionStats()
        dist.update("a", 0.25)
        dist.update("b", 0.75)
        assert abs(dist.total_support() - 1.0) < 1e-9


class TestDistributionStatsSupport:
    """Tests for DistributionStats.support method."""

    def test_support_existing_state(self: Self) -> None:
        """Test getting support for existing state."""
        dist = DistributionStats()
        dist.update("on", 5.0)
        assert dist.support("on") == 5.0

    def test_support_nonexistent_state(self: Self) -> None:
        """Test support returns 0.0 for nonexistent state."""
        dist = DistributionStats()
        dist.update("on", 5.0)
        assert dist.support("off") == 0.0

    def test_support_accumulates(self: Self) -> None:
        """Test support accumulates across updates."""
        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("on", 3.0)
        dist.update("on", 5.0)
        assert dist.support("on") == 10.0


class TestDistributionStatsDistribution:
    """Tests for DistributionStats.distribution method."""

    def test_distribution_empty_returns_empty_dict(self: Self) -> None:
        """Test distribution on empty returns empty dict."""
        dist = DistributionStats()
        assert dist.distribution() == {}

    def test_distribution_single_state(self: Self) -> None:
        """Test distribution with single state gives probability 1.0."""
        dist = DistributionStats()
        dist.update("on", 10.0)
        result = dist.distribution()
        assert len(result) == 1
        assert abs(result["on"] - 1.0) < 1e-9

    def test_distribution_two_states_equal(self: Self) -> None:
        """Test distribution with two equal states."""
        dist = DistributionStats()
        dist.update("a", 5.0)
        dist.update("b", 5.0)
        result = dist.distribution()
        assert abs(result["a"] - 0.5) < 1e-9
        assert abs(result["b"] - 0.5) < 1e-9

    def test_distribution_normalizes_correctly(self: Self) -> None:
        """Test distribution normalizes to probabilities."""
        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("off", 1.0)
        dist.update("unknown", 1.0)
        result = dist.distribution()
        assert abs(result["on"] - 0.5) < 1e-9
        assert abs(result["off"] - 0.25) < 1e-9
        assert abs(result["unknown"] - 0.25) < 1e-9

    def test_distribution_sums_to_one(self: Self) -> None:
        """Test probabilities sum to 1.0."""
        dist = DistributionStats()
        for i in range(10):
            dist.update(f"state_{i}", float(i + 1))
        result = dist.distribution()
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_distribution_with_zero_total_support(self: Self) -> None:
        """Test distribution with zero total support returns empty."""
        dist = DistributionStats()
        dist.update("a", 0.0)
        dist.update("b", 0.0)
        assert dist.distribution() == {}


class TestDistributionStatsIsConfident:
    """Tests for DistributionStats.is_confident method."""

    def test_is_confident_below_threshold(self: Self) -> None:
        """Test is_confident returns False below threshold."""
        dist = DistributionStats()
        dist.update("on", 5.0)
        assert not dist.is_confident(10.0)

    def test_is_confident_at_threshold(self: Self) -> None:
        """Test is_confident returns True at threshold."""
        dist = DistributionStats()
        dist.update("on", 10.0)
        assert dist.is_confident(10.0)

    def test_is_confident_above_threshold(self: Self) -> None:
        """Test is_confident returns True above threshold."""
        dist = DistributionStats()
        dist.update("on", 15.0)
        assert dist.is_confident(10.0)

    def test_is_confident_empty_distribution(self: Self) -> None:
        """Test is_confident on empty distribution."""
        dist = DistributionStats()
        assert not dist.is_confident(0.1)


class TestDistributionStatsActiveStates:
    """Tests for DistributionStats.active_states method."""

    def test_active_states_empty(self: Self) -> None:
        """Test active_states on empty distribution."""
        dist = DistributionStats()
        assert dist.active_states(1.0) == set()

    def test_active_states_all_below_threshold(self: Self) -> None:
        """Test active_states when all states below threshold."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 2.0)
        assert dist.active_states(10.0) == set()

    def test_active_states_some_above_threshold(self: Self) -> None:
        """Test active_states returns states above threshold."""
        dist = DistributionStats()
        dist.update("a", 5.0)
        dist.update("b", 15.0)
        dist.update("c", 20.0)
        active = dist.active_states(10.0)
        assert active == {"b", "c"}

    def test_active_states_at_threshold(self: Self) -> None:
        """Test active_states includes states at threshold."""
        dist = DistributionStats()
        dist.update("a", 10.0)
        assert dist.active_states(10.0) == {"a"}


class TestDistributionStatsEntropy:
    """Tests for DistributionStats.entropy method."""

    def test_entropy_empty_distribution(self: Self) -> None:
        """Test entropy of empty distribution."""
        dist = DistributionStats()
        assert dist.entropy() == 0.0

    def test_entropy_single_state(self: Self) -> None:
        """Test entropy with single state is zero."""
        dist = DistributionStats()
        dist.update("on", 10.0)
        assert abs(dist.entropy() - 0.0) < 1e-9

    def test_entropy_two_equal_states(self: Self) -> None:
        """Test entropy with two equal probability states."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 1.0)
        # Entropy = -0.5 * ln(0.5) - 0.5 * ln(0.5) = ln(2) ≈ 0.693
        entropy = dist.entropy()
        assert abs(entropy - 0.693147) < 0.001

    def test_entropy_three_equal_states(self: Self) -> None:
        """Test entropy with three equal probability states."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 1.0)
        dist.update("c", 1.0)
        # Entropy = 3 * (-1/3 * ln(1/3)) = ln(3) ≈ 1.099
        entropy = dist.entropy()
        assert abs(entropy - 1.0986) < 0.01

    def test_entropy_non_negative(self: Self) -> None:
        """Test entropy is always non-negative."""
        dist = DistributionStats()
        for i in range(1, 11):
            dist.update(f"state_{i}", float(i))
        assert dist.entropy() >= 0.0


class TestDistributionStatsMaxProbability:
    """Tests for DistributionStats.max_probability method."""

    def test_max_probability_empty(self: Self) -> None:
        """Test max_probability on empty distribution."""
        dist = DistributionStats()
        assert dist.max_probability() == 0.0

    def test_max_probability_single_state(self: Self) -> None:
        """Test max_probability with single state is 1.0."""
        dist = DistributionStats()
        dist.update("on", 10.0)
        assert abs(dist.max_probability() - 1.0) < 1e-9

    def test_max_probability_multiple_states(self: Self) -> None:
        """Test max_probability with multiple states."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 2.0)
        dist.update("c", 3.0)
        # b has max prob of 3/6 = 0.5
        assert abs(dist.max_probability() - 0.5) < 1e-9


class TestDistributionStatsApplyDecay:
    """Tests for DistributionStats.apply_decay method."""

    def test_apply_decay_single_state(self: Self) -> None:
        """Test decay on single state."""
        dist = DistributionStats()
        dist.update("on", 10.0)
        dist.apply_decay(0.5)
        assert dist.support("on") == 5.0

    def test_apply_decay_multiple_states(self: Self) -> None:
        """Test decay applies to all states."""
        dist = DistributionStats()
        dist.update("a", 10.0)
        dist.update("b", 20.0)
        dist.apply_decay(0.5)
        assert dist.support("a") == 5.0
        assert dist.support("b") == 10.0

    def test_apply_decay_preserves_ratios(self: Self) -> None:
        """Test decay preserves relative probabilities."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 2.0)
        dist_before = dist.distribution()
        dist.apply_decay(0.7)
        dist_after = dist.distribution()
        # Probabilities should remain same
        for state in ["a", "b"]:
            assert abs(dist_before[state] - dist_after[state]) < 1e-9


class TestDistributionStatsStates:
    """Tests for DistributionStats.states method."""

    def test_states_empty_distribution(self: Self) -> None:
        """Test states on empty distribution."""
        dist = DistributionStats()
        assert dist.states() == set()

    def test_states_single_state(self: Self) -> None:
        """Test states with single state."""
        dist = DistributionStats()
        dist.update("on")
        assert dist.states() == {"on"}

    def test_states_multiple_states(self: Self) -> None:
        """Test states returns all observed states."""
        dist = DistributionStats()
        dist.update("a")
        dist.update("b")
        dist.update("c")
        assert dist.states() == {"a", "b", "c"}


class TestDistributionStatsIsEmpty:
    """Tests for DistributionStats.is_empty method."""

    def test_is_empty_initial(self: Self) -> None:
        """Test is_empty on newly created distribution."""
        dist = DistributionStats()
        assert dist.is_empty()

    def test_is_empty_after_update(self: Self) -> None:
        """Test is_empty returns False after update."""
        dist = DistributionStats()
        dist.update("on")
        assert not dist.is_empty()

    def test_is_empty_with_zero_weight(self: Self) -> None:
        """Test is_empty after updating with zero weight."""
        dist = DistributionStats()
        dist.update("on", 0.0)
        # Still not empty (state exists)
        assert not dist.is_empty()


class TestDistributionStatsPrune:
    """Tests for DistributionStats.prune method."""

    def test_prune_removes_below_threshold(self: Self) -> None:
        """Test prune removes states below threshold."""
        dist = DistributionStats()
        dist.update("a", 5.0)
        dist.update("b", 15.0)
        dist.prune(10.0)
        assert dist.states() == {"b"}

    def test_prune_keeps_at_threshold(self: Self) -> None:
        """Test prune keeps states at threshold."""
        dist = DistributionStats()
        dist.update("a", 10.0)
        dist.prune(10.0)
        assert dist.states() == {"a"}

    def test_prune_empty_distribution(self: Self) -> None:
        """Test prune on empty distribution."""
        dist = DistributionStats()
        dist.prune(10.0)
        assert dist.is_empty()

    def test_prune_all_states(self: Self) -> None:
        """Test pruning removes all states when threshold high."""
        dist = DistributionStats()
        dist.update("a", 1.0)
        dist.update("b", 2.0)
        dist.prune(100.0)
        assert dist.is_empty()


class TestDistributionStatsPruneAdaptive:
    """Tests for DistributionStats.prune_adaptive method."""

    def test_prune_adaptive_default_params(self: Self) -> None:
        """Test prune_adaptive with default parameters."""
        dist = DistributionStats()
        dist.update("a", 1000.0)
        dist.update("b", 100.0)
        dist.update("c", 10.0)
        # Threshold = max(1110 * 0.003, 20.0) = max(3.33, 20) = 20.0
        dist.prune_adaptive()
        # Only 'a' and 'b' should remain
        assert "a" in dist.states()
        assert "b" in dist.states()
        # 'c' with 10.0 should be removed
        assert "c" not in dist.states()

    def test_prune_adaptive_custom_epsilon(self: Self) -> None:
        """Test prune_adaptive with custom epsilon."""
        dist = DistributionStats()
        for i in range(1, 11):
            dist.update(f"state_{i}", float(i * 10))
        # Total = 550, threshold = max(550 * 0.01, 20.0) = max(5.5, 20) = 20.0
        dist.prune_adaptive(epsilon=0.01, absolute_min=20.0)
        # States with 20+ should remain (states 2-10)
        assert "state_1" not in dist.states()  # 10 < 20

    def test_prune_adaptive_custom_absolute_min(self: Self) -> None:
        """Test prune_adaptive respects absolute minimum."""
        dist = DistributionStats()
        dist.update("a", 1000.0)
        dist.update("b", 10.0)
        # With low epsilon, threshold might be < 10
        # But absolute_min=100 forces threshold=100
        dist.prune_adaptive(epsilon=0.001, absolute_min=100.0)
        # Only 'a' should remain (1000 >= 100)
        assert "a" in dist.states()
        assert "b" not in dist.states()


class TestDistributionStatsIntegration:
    """Integration tests with multiple operations."""

    def test_workflow_update_distribute_prune(self: Self) -> None:
        """Test complete workflow: update, get distribution, then prune."""
        dist = DistributionStats()
        dist.update("on", 50.0)
        dist.update("off", 20.0)
        dist.update("unknown", 5.0)

        # Check distribution before pruning
        dist_before = dist.distribution()
        assert abs(dist_before["on"] - 50 / 75) < 1e-9

        # Prune and check
        dist.prune(10.0)
        assert dist.states() == {"on", "off"}

    def test_workflow_decay_then_stable(self: Self) -> None:
        """Test decay followed by getting stable distribution."""
        dist = DistributionStats()
        dist.update("a", 10.0)
        dist.update("b", 10.0)

        # Apply decay
        dist.apply_decay(0.8)
        # Total now 16.0

        # Distribution should maintain ratios
        dist_after = dist.distribution()
        assert abs(dist_after["a"] - 0.5) < 1e-9
        assert abs(dist_after["b"] - 0.5) < 1e-9

    def test_many_states_operations(self: Self) -> None:
        """Test with many states."""
        dist = DistributionStats()
        for i in range(100):
            dist.update(f"state_{i}", float(i + 1))

        # Check total support
        total = sum(i + 1 for i in range(100))
        assert abs(dist.total_support() - total) < 1e-6

        # Check distribution sums to 1
        dist_dict = dist.distribution()
        prob_sum = sum(dist_dict.values())
        assert abs(prob_sum - 1.0) < 1e-9
