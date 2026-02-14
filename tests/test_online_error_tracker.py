"""
Unit tests for OnlineErrorTracker.

Comprehensive tests for the OnlineErrorTracker class, covering initialization,
error tracking, exponential decay, and statistical calculations.
"""

import math
from typing import Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker import (
    OnlineErrorTracker,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_hyper_parameters import (  # noqa: E501
    OnlineErrorTrackerHyperParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_runtime_parameters import (
    OnlineErrorTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)


def create_test_hp() -> OnlineErrorTrackerHyperParameters:
    """Create OnlineErrorTrackerHyperParameters for testing purposes."""
    base_hp = ForecasterEngineHyperParameters(
        half_life=50.0,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    return OnlineErrorTrackerHyperParameters(
        hyper_parameters=base_hp,
    )


def create_test_rp(
    error_half_life_factor: float = 1.0,
) -> OnlineErrorTrackerRuntimeParameters:
    """Create OnlineErrorTrackerRuntimeParameters for testing purposes."""
    return OnlineErrorTrackerRuntimeParameters(
        error_half_life_factor=error_half_life_factor
    )


class TestOnlineErrorTrackerInitialization:
    """Tests for OnlineErrorTracker initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating OnlineErrorTracker with default hyper-parameters."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)
        assert tracker.mean == 0.0
        assert tracker.std >= 0.0

    def test_initial_mean_is_zero(self: Self) -> None:
        """Test that initial mean error is zero."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)
        assert tracker.mean == 0.0

    def test_initial_std_is_small(self: Self) -> None:
        """Test that initial std is very small (floor value)."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)
        # Should be sqrt(max(0.0, 1e-12)) = sqrt(1e-12)
        assert tracker.std < 1e-5


class TestOnlineErrorTrackerUpdate:
    """Tests for OnlineErrorTracker.update method."""

    def test_first_update_perfect_prediction(self: Self) -> None:
        """Test first update with perfect prediction (p=1.0)."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        # Error should be -log(1.0) = 0
        assert abs(tracker.mean - 0.0) < 1e-9

    def test_first_update_wrong_prediction(self: Self) -> None:
        """Test first update with completely wrong prediction."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "off", 100.0)

        # Error should be -log(1e-12) which is large
        assert tracker.mean > 10.0

    def test_first_update_uncertain_prediction(self: Self) -> None:
        """Test first update with uncertain prediction."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)
        dist.update("off", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        # Error should be -log(0.5) ≈ 0.693
        expected_error = -math.log(0.5)
        assert abs(tracker.mean - expected_error) < 1e-6

    def test_multiple_updates_same_prediction(self: Self) -> None:
        """Test multiple updates with same prediction quality."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=1.0)
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("off", 1.0)

        # Update multiple times with same prediction
        for i in range(5):
            tracker.update(dist.distribution, "on", 100.0 + i * 10.0)

        # Mean should stabilize around -log(2/3)
        expected = -math.log(2.0 / 3.0)
        assert abs(tracker.mean - expected) < 0.01

    def test_update_with_zero_time_delta(self: Self) -> None:
        """Test that update with zero time delta doesn't change statistics."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)
        mean_before = tracker.mean

        # Update again at same timestamp
        tracker.update(dist.distribution, "on", 100.0)
        mean_after = tracker.mean

        assert mean_before == mean_after

    def test_update_with_negative_time_delta(self: Self) -> None:
        """Test that update with negative time delta doesn't change statistics."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)
        mean_before = tracker.mean

        # Update with earlier timestamp
        tracker.update(dist.distribution, "on", 50.0)
        mean_after = tracker.mean

        assert mean_before == mean_after


class TestOnlineErrorTrackerMeanProperty:
    """Tests for OnlineErrorTracker.mean property."""

    def test_mean_after_single_update(self: Self) -> None:
        """Test mean after single update."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 3.0)
        dist.update("off", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        # Mean should be -log(0.75)
        expected = -math.log(0.75)
        assert abs(tracker.mean - expected) < 1e-9

    def test_mean_multiple_calls(self: Self) -> None:
        """Test that mean returns same value on multiple calls."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        mean1 = tracker.mean
        mean2 = tracker.mean
        assert mean1 == mean2


class TestOnlineErrorTrackerStdProperty:
    """Tests for OnlineErrorTracker.std property."""

    def test_std_after_first_update(self: Self) -> None:
        """Test std after first update is near zero."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        # After first update, variance is 0
        assert tracker.std < 1e-5

    def test_std_with_varying_errors(self: Self) -> None:
        """Test std increases with varying prediction quality."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=10.0)
        tracker = OnlineErrorTracker(hp, rp)

        # Perfect prediction
        dist1 = DistributionStats()
        dist1.update("on", 1.0)
        tracker.update(dist1.distribution, "on", 100.0)

        # Bad prediction
        dist2 = DistributionStats()
        dist2.update("on", 0.1)
        dist2.update("off", 0.9)
        tracker.update(dist2.distribution, "on", 200.0)

        # Should have some variance now
        assert tracker.std > 0.1

    def test_std_multiple_calls(self: Self) -> None:
        """Test that std returns same value on multiple calls."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)
        dist.update("off", 1.0)

        tracker.update(dist.distribution, "on", 100.0)
        tracker.update(dist.distribution, "off", 150.0)

        std1 = tracker.std
        std2 = tracker.std
        assert std1 == std2


class TestOnlineErrorTrackerDecay:
    """Tests for exponential decay behavior."""

    def test_decay_reduces_old_error_influence(self: Self) -> None:
        """Test that old errors have less influence after time passes."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=1.0)
        tracker = OnlineErrorTracker(hp, rp)

        # Large error
        dist_bad = DistributionStats()
        dist_bad.update("on", 0.01)
        dist_bad.update("off", 0.99)
        tracker.update(dist_bad.distribution, "on", 0.0)

        mean_after_bad = tracker.mean

        # After significant time, add perfect predictions
        dist_good = DistributionStats()
        dist_good.update("on", 1.0)

        # Multiple updates after 1 half-life
        tracker.update(dist_good.distribution, "on", 50.0)
        tracker.update(dist_good.distribution, "on", 100.0)
        tracker.update(dist_good.distribution, "on", 150.0)

        mean_after_good = tracker.mean

        # Mean should decrease significantly
        assert mean_after_good < mean_after_bad * 0.5

    def test_decay_with_different_half_lives(self: Self) -> None:
        """Test decay behavior with different half-life factors."""
        # Fast decay
        hp_fast = create_test_hp()
        rp_fast = create_test_rp(error_half_life_factor=0.1)
        tracker_fast = OnlineErrorTracker(hp_fast, rp_fast)

        # Slow decay
        hp_slow = create_test_hp()
        rp_slow = create_test_rp(error_half_life_factor=10.0)
        tracker_slow = OnlineErrorTracker(hp_slow, rp_slow)

        # Initial bad error
        dist_bad = DistributionStats()
        dist_bad.update("on", 0.1)
        dist_bad.update("off", 0.9)

        tracker_fast.update(dist_bad.distribution, "on", 0.0)
        tracker_slow.update(dist_bad.distribution, "on", 0.0)

        # Good prediction after time
        dist_good = DistributionStats()
        dist_good.update("on", 1.0)

        tracker_fast.update(dist_good.distribution, "on", 50.0)
        tracker_slow.update(dist_good.distribution, "on", 50.0)

        # Fast decay should adapt quicker
        assert tracker_fast.mean < tracker_slow.mean


class TestOnlineErrorTrackerEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_prediction_with_zero_probability(self: Self) -> None:
        """Test handling of prediction with zero probability for true state."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        # Predict "on" but observe "off" (not in distribution)
        tracker.update(dist.distribution, "off", 100.0)

        # Should clamp to 1e-12 and compute -log(1e-12)
        expected = -math.log(1e-12)
        assert abs(tracker.mean - expected) < 1e-6

    def test_very_confident_correct_prediction(self: Self) -> None:
        """Test with very confident correct prediction."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1000.0)
        dist.update("off", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        # Error should be very small
        assert tracker.mean < 0.01

    def test_sequence_of_perfect_predictions(self: Self) -> None:
        """Test sequence of perfect predictions."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        for i in range(10):
            tracker.update(dist.distribution, "on", 100.0 + i * 10.0)

        # Mean should be 0
        assert abs(tracker.mean - 0.0) < 1e-9

    def test_sequence_of_wrong_predictions(self: Self) -> None:
        """Test sequence of completely wrong predictions."""
        hp = create_test_hp()
        rp = create_test_rp()
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        for i in range(10):
            tracker.update(dist.distribution, "off", 100.0 + i * 10.0)

        # Mean should be -log(1e-12)
        expected = -math.log(1e-12)
        assert abs(tracker.mean - expected) < 1e-3

    def test_alternating_predictions(self: Self) -> None:
        """Test alternating between good and bad predictions."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=10.0)
        tracker = OnlineErrorTracker(hp, rp)

        dist_good = DistributionStats()
        dist_good.update("on", 0.9)
        dist_good.update("off", 0.1)

        dist_bad = DistributionStats()
        dist_bad.update("on", 0.1)
        dist_bad.update("off", 0.9)

        for i in range(20):
            if i % 2 == 0:
                tracker.update(dist_good.distribution, "on", 100.0 + i * 10.0)
            else:
                tracker.update(dist_bad.distribution, "on", 100.0 + i * 10.0)

        # Should have some mean and variance
        assert tracker.mean > 0.0
        assert tracker.std > 0.0


class TestOnlineErrorTrackerIntegration:
    """Integration tests with complete workflows."""

    def test_complete_tracking_workflow(self: Self) -> None:
        """Test complete workflow of tracking errors over time."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=1.0)
        tracker = OnlineErrorTracker(hp, rp)

        # Start with uncertain predictions
        dist_uncertain = DistributionStats()
        dist_uncertain.update("on", 1.0)
        dist_uncertain.update("off", 1.0)

        for i in range(10):
            tracker.update(dist_uncertain.distribution, "on", i * 10.0)

        mean_uncertain = tracker.mean

        # Improve to confident predictions
        dist_confident = DistributionStats()
        dist_confident.update("on", 9.0)
        dist_confident.update("off", 1.0)

        for i in range(10, 20):
            tracker.update(dist_confident.distribution, "on", i * 10.0)

        mean_confident = tracker.mean

        # Mean should decrease (less error)
        assert mean_confident < mean_uncertain


class TestOnlineErrorTrackerSerialization:
    """Tests for OnlineErrorTracker serialization helpers."""

    def test_to_dict_contains_expected_keys(self: Self) -> None:
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=1.0)
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 1.0)

        tracker.update(dist.distribution, "on", 100.0)

        data = tracker.to_dict()
        assert isinstance(data, dict)
        assert set(["mean", "var", "last_ts"]).issubset(set(data.keys()))

    def test_from_dict_restores_tracker_state(self: Self) -> None:
        base_hp = ForecasterEngineHyperParameters(
            half_life=60.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.9,
        )
        hp = OnlineErrorTrackerHyperParameters(hyper_parameters=base_hp)
        rp = create_test_rp(error_half_life_factor=0.5)
        tracker = OnlineErrorTracker(hp, rp)

        dist = DistributionStats()
        dist.update("on", 2.0)
        dist.update("off", 1.0)

        tracker.update(dist.distribution, "on", 100.0)
        tracker.update(dist.distribution, "on", 150.0)

        data = tracker.to_dict()

        restored = OnlineErrorTracker.from_dict(data, hp, rp)

        # Numeric values restored
        assert abs(restored.mean - tracker.mean) < 1e-12
        assert abs(restored._var - tracker._var) < 1e-12
        assert restored._last_ts == tracker._last_ts

        # Hyper-parameters reconstructed correctly
        assert (
            abs(
                restored._parameters.error_half_life
                - tracker._parameters.error_half_life
            )
            < 1e-12
        )

    def test_realistic_prediction_sequence(self: Self) -> None:
        """Test with realistic sequence of varying prediction quality."""
        hp = create_test_hp()
        rp = create_test_rp(error_half_life_factor=2.0)
        tracker = OnlineErrorTracker(hp, rp)

        # Sequence of predictions with varying confidence
        predictions = [
            (0.8, "on"),
            (0.9, "on"),
            (0.7, "on"),
            (0.95, "on"),
            (0.6, "off"),  # Wrong prediction
            (0.85, "on"),
            (0.9, "on"),
        ]

        for i, (prob_on, true_state) in enumerate(predictions):
            dist = DistributionStats()
            dist.update("on", prob_on)
            dist.update("off", 1.0 - prob_on)
            tracker.update(dist.distribution, true_state, i * 100.0)

        # Should have reasonable mean and std
        assert tracker.mean > 0.0
        assert tracker.std > 0.0
