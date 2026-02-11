"""
Unit tests for BaselineDistribution.

Comprehensive tests covering initialization, updating, decay, and distribution
computation for the BaselineDistribution class.
"""

import json
from typing import Self

from custom_components.discrete_state_forecaster.model.learning.baseline_distribution import (
    BaselineDistribution,
)


class TestBaselineDistributionInitialization:
    """Tests for BaselineDistribution initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating BaselineDistribution with default parameters."""
        baseline = BaselineDistribution(half_life=50.0)
        dist = baseline.distribution()
        assert dist == {}

    def test_create_with_custom_epsilon(self: Self) -> None:
        """Test creating with custom epsilon."""
        baseline = BaselineDistribution(half_life=50.0, epsilon=1e-6)
        assert baseline is not None


class TestBaselineDistributionUpdate:
    """Tests for BaselineDistribution.update method."""

    def test_first_update(self: Self) -> None:
        """Test first update copies distribution."""
        baseline = BaselineDistribution(half_life=50.0)
        dist = {"on": 0.7, "off": 0.3}
        baseline.update(dist, 100.0)

        result = baseline.distribution()
        assert "on" in result
        assert "off" in result
        assert abs(result["on"] - 0.7) < 0.01

    def test_multiple_updates_same_dist(self: Self) -> None:
        """Test multiple updates with same distribution stabilize."""
        baseline = BaselineDistribution(half_life=50.0)
        dist = {"on": 0.6, "off": 0.4}

        for i in range(10):
            baseline.update(dist, 100.0 + i * 10.0)

        result = baseline.distribution()
        assert abs(result["on"] - 0.6) < 0.02

    def test_update_with_zero_time_delta(self: Self) -> None:
        """Test that update with zero time delta is ignored."""
        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 0.7, "off": 0.3}, 100.0)
        result1 = baseline.distribution()

        # Update at same timestamp
        baseline.update({"on": 0.3, "off": 0.7}, 100.0)
        result2 = baseline.distribution()

        assert result1 == result2

    def test_update_with_negative_time_delta(self: Self) -> None:
        """Test that update with negative time delta is ignored."""
        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 0.7, "off": 0.3}, 100.0)
        result1 = baseline.distribution()

        # Update with earlier timestamp
        baseline.update({"on": 0.3, "off": 0.7}, 50.0)
        result2 = baseline.distribution()

        assert result1 == result2


class TestBaselineDistributionDecay:
    """Tests for exponential decay behavior."""

    def test_decay_reduces_old_weights(self: Self) -> None:
        """Test that old weights decay over time."""
        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 1.0}, 100.0)

        # After one half-life, mix should be ~0.5
        baseline.update({"off": 1.0}, 150.0)

        result = baseline.distribution()
        # Both states should have some mass but off should dominate
        assert "on" in result
        assert "off" in result

    def test_pruning_removes_small_weights(self: Self) -> None:
        """Test that very small weights are pruned."""
        baseline = BaselineDistribution(half_life=10.0, prune_threshold=1e-6)
        baseline.update({"on": 1.0}, 100.0)

        # After many half-lives, on should be pruned
        for i in range(20):
            baseline.update({"off": 1.0}, 110.0 + i * 10.0)

        result = baseline.distribution()
        # "on" should have been pruned
        assert "on" not in result or result.get("on", 0) < 0.01


class TestBaselineDistributionDistribution:
    """Tests for distribution() method."""

    def test_distribution_sums_to_one(self: Self) -> None:
        """Test that distribution sums to approximately 1.0."""
        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 0.6, "off": 0.4}, 100.0)

        result = baseline.distribution()
        total = sum(result.values())

        assert abs(total - 1.0) < 1e-6

    def test_distribution_with_laplace_smoothing(self: Self) -> None:
        """Test that Laplace smoothing prevents zero probabilities."""
        baseline = BaselineDistribution(half_life=50.0, epsilon=1e-9)
        baseline.update({"on": 1.0, "off": 0.0}, 100.0)

        result = baseline.distribution()
        # Even with zero probability, should have tiny value
        assert result["off"] > 0

    def test_empty_distribution_before_first_update(self: Self) -> None:
        """Test that distribution is empty before first update."""
        baseline = BaselineDistribution(half_life=50.0)
        result = baseline.distribution()
        assert result == {}


class TestBaselineDistributionEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_state_distribution(self: Self) -> None:
        """Test with only one state."""
        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 1.0}, 100.0)

        result = baseline.distribution()
        assert abs(result["on"] - 1.0) < 1e-6

    def test_many_states(self: Self) -> None:
        """Test with many states."""
        baseline = BaselineDistribution(half_life=50.0)
        dist = {f"state{i}": 1.0 / 10 for i in range(10)}

        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)

        result = baseline.distribution()
        assert len(result) == 10
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_changing_distributions(self: Self) -> None:
        """Test with gradually changing distributions."""
        baseline = BaselineDistribution(half_life=50.0)

        # Start with on=0.8
        baseline.update({"on": 0.8, "off": 0.2}, 100.0)

        # Gradually shift to off=0.8 over longer period
        for i in range(20):
            ratio = i / 20.0
            baseline.update(
                {"on": 0.8 - 0.6 * ratio, "off": 0.2 + 0.6 * ratio},
                110.0 + i * 50.0,
            )

        result = baseline.distribution()
        # Should have shifted towards off
        assert result["off"] > 0.4


class TestBaselineDistributionSerialization:
    """Tests for BaselineDistribution.to_dict and from_dict."""

    def test_to_dict_json_serializable(self: Self) -> None:
        

        baseline = BaselineDistribution(half_life=50.0)
        baseline.update({"on": 0.7, "off": 0.3}, 100.0)

        data = baseline.to_dict()

        # Should be JSON serializable
        dumped = json.dumps(data)
        assert isinstance(dumped, str)

    def test_from_dict_roundtrip(self: Self) -> None:
        baseline = BaselineDistribution(half_life=20.0, epsilon=1e-6, prune_threshold=1e-8)
        baseline.update({"on": 1.0}, 50.0)
        baseline.update({"off": 1.0}, 70.0)

        data = baseline.to_dict()
        new = BaselineDistribution.from_dict(data)

        # Check attributes
        assert abs(new._half_life - baseline._half_life) < 1e-12
        assert abs(new._epsilon - baseline._epsilon) < 1e-12
        assert abs(new._prune_threshold - baseline._prune_threshold) < 1e-12
        assert new._last_ts == baseline._last_ts

        # Check distributions are similar
        old_dist = baseline.distribution()
        new_dist = new.distribution()

        for k in set(old_dist.keys()).union(new_dist.keys()):
            assert abs(old_dist.get(k, 0.0) - new_dist.get(k, 0.0)) < 1e-6
