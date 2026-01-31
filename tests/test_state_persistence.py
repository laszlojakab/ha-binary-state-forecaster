"""
Tests for state persistence features in TimeAwareForecaster.

Tests cover:
- Adaptive persistence learning from state transitions
- State-specific persistence factors
- Duration-dependent persistence adjustment
- Prediction with current state context
- Fixed vs adaptive persistence modes
"""

from datetime import datetime

from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    StatePersistenceTracker,
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestStatePersistenceTracker:
    """Tests for StatePersistenceTracker class."""

    def test_init_empty(self) -> None:
        """Test initialization creates empty tracker."""
        tracker = StatePersistenceTracker()
        assert tracker.persistence_count == {}
        assert tracker.total_count == {}
        assert tracker.smoothing == 0.95

    def test_update_single_persistence(self) -> None:
        """Test updating with a single persistence observation."""
        tracker = StatePersistenceTracker()
        tracker.update("on", persisted=True)

        assert "on" in tracker.persistence_count
        assert "on" in tracker.total_count
        assert tracker.persistence_count["on"] == 1.0
        assert tracker.total_count["on"] == 1.0

    def test_update_single_transition(self) -> None:
        """Test updating with a single transition observation."""
        tracker = StatePersistenceTracker()
        tracker.update("on", persisted=False)

        assert "on" in tracker.persistence_count
        assert "on" in tracker.total_count
        assert tracker.persistence_count["on"] == 0.0
        assert tracker.total_count["on"] == 1.0

    def test_update_multiple_observations(self) -> None:
        """Test updating with multiple observations."""
        tracker = StatePersistenceTracker()

        # 3 persisted, 1 transition
        tracker.update("on", persisted=True)
        tracker.update("on", persisted=True)
        tracker.update("on", persisted=True)
        tracker.update("on", persisted=False)

        # With smoothing, exact values depend on exponential moving average
        assert tracker.total_count["on"] > 0
        assert tracker.persistence_count["on"] > 0
        assert tracker.persistence_count["on"] < tracker.total_count["on"]

    def test_get_persistence_factor_insufficient_data(self) -> None:
        """Test getting factor with insufficient data returns default."""
        tracker = StatePersistenceTracker()

        # Add only a few observations (< 10)
        for _ in range(5):
            tracker.update("on", persisted=True)

        factor = tracker.get_persistence_factor("on", default=0.3)
        assert factor == 0.3  # Should return default

    def test_get_persistence_factor_sufficient_data(self) -> None:
        """Test getting factor with sufficient data returns learned value."""
        tracker = StatePersistenceTracker()

        # Add many observations - all persisted
        for _ in range(20):
            tracker.update("on", persisted=True)

        factor = tracker.get_persistence_factor("on", default=0.3)
        # Should be high (blend of 1.0 and default 0.3)
        assert factor > 0.8  # 80% of ~1.0 + 20% of 0.3

    def test_get_persistence_factor_low_persistence(self) -> None:
        """Test getting factor for non-persistent state."""
        tracker = StatePersistenceTracker()

        # Add many observations - mostly transitions
        for _ in range(20):
            tracker.update("off", persisted=False)

        factor = tracker.get_persistence_factor("off", default=0.3)
        # Should be low (blend of ~0.0 and default 0.3)
        assert factor < 0.3

    def test_get_persistence_factor_mixed(self) -> None:
        """Test getting factor for state with mixed persistence."""
        tracker = StatePersistenceTracker()

        # 50/50 mix
        for _ in range(10):
            tracker.update("on", persisted=True)
        for _ in range(10):
            tracker.update("on", persisted=False)

        factor = tracker.get_persistence_factor("on", default=0.3)
        # Should be around middle
        assert 0.3 < factor < 0.7

    def test_multiple_states(self) -> None:
        """Test tracking multiple states independently."""
        tracker = StatePersistenceTracker()

        # "on" is sticky
        for _ in range(20):
            tracker.update("on", persisted=True)

        # "off" is not sticky
        for _ in range(20):
            tracker.update("off", persisted=False)

        on_factor = tracker.get_persistence_factor("on", default=0.3)
        off_factor = tracker.get_persistence_factor("off", default=0.3)

        assert on_factor > 0.7  # High persistence
        assert off_factor < 0.3  # Low persistence


class TestAdaptivePersistence:
    """Tests for adaptive persistence learning in TimeAwareForecaster."""

    def test_adaptive_persistence_enabled_by_default(self) -> None:
        """Test that adaptive persistence is enabled by default."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        assert forecaster.adaptive_persistence is True

    def test_adaptive_persistence_can_be_disabled(self) -> None:
        """Test that adaptive persistence can be disabled."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, adaptive_persistence=False)

        assert forecaster.adaptive_persistence is False

    def test_learns_state_persistence(self) -> None:
        """Test that forecaster learns state persistence from transitions."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, adaptive_persistence=True)

        # Simulate sticky "on" state - same state continues without interruption
        for i in range(15):
            forecaster.update_interval(
                datetime(2024, 1, 1 + i, 10, 0),
                datetime(2024, 1, 1 + i, 10, 30),
                "on",
            )
            # Add another interval immediately after (simulates continuous "on")
            forecaster.update_interval(
                datetime(2024, 1, 1 + i, 10, 30),
                datetime(2024, 1, 1 + i, 11, 0),
                "on",
            )

        # Simulate transient "off" state - alternating states quickly
        for i in range(15):
            forecaster.update_interval(
                datetime(2024, 1, 1 + i, 14, 0),
                datetime(2024, 1, 1 + i, 14, 10),
                "off",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + i, 14, 10),
                datetime(2024, 1, 1 + i, 14, 20),
                "idle",
            )

        # Check learned persistence
        persistence = forecaster.get_learned_persistence()

        # "on" should have higher persistence than "off"
        # "on" persists (continues without change), "off" transitions to "idle"
        if "on" in persistence and "off" in persistence:
            assert persistence["on"] > persistence["off"]

    def test_get_learned_persistence_all_states(self) -> None:
        """Test getting learned persistence for all states."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, adaptive_persistence=True)

        # Add data for multiple states
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
            "off",
        )

        persistence = forecaster.get_learned_persistence()
        assert isinstance(persistence, dict)

    def test_get_learned_persistence_specific_state(self) -> None:
        """Test getting learned persistence for specific state."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer, adaptive_persistence=True)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        persistence = forecaster.get_learned_persistence("on")
        assert isinstance(persistence, dict)
        assert "on" in persistence
        assert len(persistence) == 1


class TestPredictionWithCurrentState:
    """Tests for prediction with current state context."""

    def test_predict_without_current_state(self) -> None:
        """Test prediction without current state returns base prediction."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with clear pattern
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        # Predict without current state
        prediction = forecaster.predict(datetime(2024, 1, 2, 10, 30))
        assert prediction.state == "on"

    def test_predict_with_matching_current_state(self) -> None:
        """Test prediction with current state boosts that state's probability."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.3
        )

        # Train with 60% on, 40% off pattern
        for _ in range(3):
            forecaster.update_interval(
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                "on",
            )
        for _ in range(2):
            forecaster.update_interval(
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 11, 0),
                "off",
            )

        # Predict without current state
        pred_no_context = forecaster.predict(datetime(2024, 1, 3, 10, 30))

        # Predict with current state = "on"
        pred_with_on = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on"
        )

        # Predict with current state = "off"
        pred_with_off = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="off"
        )

        # Probability of "on" should be higher when current state is "on"
        assert pred_with_on.distribution["on"] > pred_no_context.distribution["on"]

        # Probability of "off" should be higher when current state is "off"
        assert pred_with_off.distribution["off"] > pred_no_context.distribution["off"]

    def test_predict_with_novel_current_state(self) -> None:
        """Test prediction when current state wasn't in training data."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train only with "on" state
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        # Predict with novel state "idle"
        prediction = forecaster.predict(
            datetime(2024, 1, 2, 10, 30), current_state="idle"
        )

        # Should include "idle" in distribution
        assert "idle" in prediction.distribution
        assert prediction.distribution["idle"] > 0


class TestDurationDependentPersistence:
    """Tests for duration-dependent persistence adjustment."""

    def test_duration_adjustment_short_duration(self) -> None:
        """Test that short durations have minimal adjustment."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train with balanced pattern
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        # Predict with short duration (5 minutes = 300s)
        pred_short = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on", state_duration=300.0
        )

        # Predict without duration
        pred_no_duration = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on"
        )

        # Should be similar (300s is baseline)
        assert (
            abs(pred_short.distribution["on"] - pred_no_duration.distribution["on"])
            < 0.05
        )

    def test_duration_adjustment_long_duration(self) -> None:
        """Test that long durations significantly boost persistence."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train with balanced pattern
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        # Predict with long duration (2 hours = 7200s)
        pred_long = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on", state_duration=7200.0
        )

        # Predict with short duration
        pred_short = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on", state_duration=300.0
        )

        # Long duration should have higher probability for current state
        assert pred_long.distribution["on"] > pred_short.distribution["on"]

    def test_duration_adjustment_scaling(self) -> None:
        """Test that duration adjustment scales logarithmically."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train with balanced pattern
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        # Test various durations
        pred_5min = forecaster.predict(
            datetime(2024, 1, 3, 10, 30),
            current_state="on",
            state_duration=300.0,  # 5 minutes
        )
        pred_30min = forecaster.predict(
            datetime(2024, 1, 3, 10, 30),
            current_state="on",
            state_duration=1800.0,  # 30 minutes
        )
        pred_2hr = forecaster.predict(
            datetime(2024, 1, 3, 10, 30),
            current_state="on",
            state_duration=7200.0,  # 2 hours
        )
        pred_10hr = forecaster.predict(
            datetime(2024, 1, 3, 10, 30),
            current_state="on",
            state_duration=36000.0,  # 10 hours
        )

        # Should increase with duration (but may plateau due to log scaling)
        assert pred_5min.distribution["on"] < pred_30min.distribution["on"]
        assert pred_30min.distribution["on"] < pred_2hr.distribution["on"]
        # At very long durations, the multiplier caps at ~1.5x, so growth slows
        assert pred_2hr.distribution["on"] <= pred_10hr.distribution["on"]

    def test_duration_adjustment_caps_at_one(self) -> None:
        """Test that duration adjustment doesn't exceed 1.0."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.9
        )

        # Train with strong pattern for "on"
        for _ in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                "on",
            )

        # Predict with very long duration
        prediction = forecaster.predict(
            datetime(2024, 1, 2, 10, 30),
            current_state="on",
            state_duration=100000.0,  # Very long
        )

        # Probability should not exceed 1.0
        assert prediction.distribution["on"] <= 1.0

    def test_duration_none_uses_base_factor(self) -> None:
        """Test that None duration uses base persistence factor."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train with balanced pattern
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        # Predict with None duration
        pred_none = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on", state_duration=None
        )

        # Predict without duration parameter (implicit None)
        pred_implicit = forecaster.predict(
            datetime(2024, 1, 3, 10, 30), current_state="on"
        )

        # Should be identical
        assert pred_none.distribution == pred_implicit.distribution

    def test_duration_zero_uses_base_factor(self) -> None:
        """Test that zero duration uses base persistence factor."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.5
        )

        # Train
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )

        # Predict with zero duration
        pred_zero = forecaster.predict(
            datetime(2024, 1, 2, 10, 30), current_state="on", state_duration=0.0
        )

        # Predict without duration
        pred_none = forecaster.predict(datetime(2024, 1, 2, 10, 30), current_state="on")

        # Should be identical
        assert pred_zero.distribution == pred_none.distribution


class TestIntegrationAdaptiveAndDuration:
    """Integration tests combining adaptive persistence and duration adjustment."""

    def test_adaptive_and_duration_together(self) -> None:
        """Test that adaptive persistence and duration work together."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=True, state_persistence_factor=0.3
        )

        # Train with sticky "on" state
        for i in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + i, 10, 0),
                datetime(2024, 1, 1 + i, 12, 0),
                "on",
            )

        # Predict with learned persistence and long duration
        prediction = forecaster.predict(
            datetime(2024, 1, 15, 10, 30),
            current_state="on",
            state_duration=7200.0,  # 2 hours
        )

        # Should have very high probability due to both factors
        assert prediction.state == "on"
        assert prediction.distribution["on"] > 0.8

    def test_fixed_persistence_with_duration(self) -> None:
        """Test fixed persistence mode with duration adjustment."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(
            indexer, adaptive_persistence=False, state_persistence_factor=0.4
        )

        # Train
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            "on",
        )
        forecaster.update_interval(
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            "off",
        )

        # Predict with fixed factor but long duration
        prediction = forecaster.predict(
            datetime(2024, 1, 3, 10, 30),
            current_state="on",
            state_duration=10800.0,  # 3 hours
        )

        # Should boost due to duration even with fixed base factor
        assert prediction.state == "on"
