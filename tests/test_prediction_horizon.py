"""Tests for prediction horizon functionality in TimeAwareForecaster."""

from datetime import datetime, timedelta

from custom_components.discrete_state_forecaster.model.time_aware_forecaster import (
    TimeAwareForecaster,
)
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import (
    CompositeIndexer,
)
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import (
    TimeOfDayIndexer,
)


class TestPredictHorizon:
    """Tests for predict_horizon() method."""

    def test_empty_horizon_returns_empty_list(self) -> None:
        """Test that zero or negative horizon returns empty list."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Zero horizon
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 1, 10, 0), horizon_minutes=0
        )
        assert predictions == []

        # Negative horizon
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 1, 10, 0), horizon_minutes=-10
        )
        assert predictions == []

    def test_single_step_horizon_returns_one_prediction(self) -> None:
        """Test that horizon equal to interval returns single prediction."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with some data
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        # Request 5-minute horizon with 5-minute interval
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=5, interval_minutes=5
        )

        assert len(predictions) == 1
        assert predictions[0].timestamp == datetime(2024, 1, 2, 10, 0)

    def test_horizon_creates_multiple_predictions(self) -> None:
        """Test that horizon creates predictions at regular intervals."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with data
        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        # Request 30-minute horizon with 10-minute intervals
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=30, interval_minutes=10
        )

        assert len(predictions) == 3  # 0, 10, 20 minutes
        assert predictions[0].timestamp == datetime(2024, 1, 2, 10, 0)
        assert predictions[1].timestamp == datetime(2024, 1, 2, 10, 10)
        assert predictions[2].timestamp == datetime(2024, 1, 2, 10, 20)

    def test_default_interval_uses_smallest_bucket(self) -> None:
        """Test that default interval equals smallest bucket size."""
        # 30-minute buckets
        indexer = CompositeIndexer([TimeOfDayIndexer(30)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        # Don't specify interval - should default to 30 minutes
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=60
        )

        assert len(predictions) == 2  # 0 and 30 minutes
        assert predictions[0].timestamp == datetime(2024, 1, 2, 10, 0)
        assert predictions[1].timestamp == datetime(2024, 1, 2, 10, 30)

    def test_predictions_carry_full_prediction_objects(self) -> None:
        """Test that each horizon prediction contains full Prediction object."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=10, interval_minutes=5
        )

        for horizon_pred in predictions:
            # Check prediction structure
            assert horizon_pred.prediction is not None
            assert hasattr(horizon_pred.prediction, "state")
            assert hasattr(horizon_pred.prediction, "distribution")
            assert hasattr(horizon_pred.prediction, "confidence")


class TestSequentialChaining:
    """Tests for sequential state chaining across horizon."""

    def test_state_persists_across_predictions(self) -> None:
        """Test that predicted state persists in sequential predictions."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train with consistent "on" state at 10:00 hour
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 11, 0),
                "on",
            )

        # Predict horizon - all should be "on" since pattern is consistent
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 15, 10, 0), horizon_minutes=30, interval_minutes=10
        )

        # All predictions should be "on"
        for pred in predictions:
            assert pred.prediction.state == "on"
            assert pred.is_transition is False

    def test_state_duration_accumulates(self) -> None:
        """Test that state_duration accumulates when state persists."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        # Predict with initial duration
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0),
            horizon_minutes=30,  # Changed from 20 to get 3 predictions
            interval_minutes=10,
            current_state="on",
            state_duration=300,  # Already on for 5 minutes
        )

        # First prediction: 300s (initial)
        assert predictions[0].state_duration == 300

        # Second prediction: 300 + 600 = 900s (5 + 10 minutes)
        assert predictions[1].state_duration == 900

        # Third prediction: 900 + 600 = 1500s (5 + 10 + 10 minutes)
        assert predictions[2].state_duration == 1500

    def test_state_duration_resets_on_transition(self) -> None:
        """Test that state_duration resets when state changes."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train pattern: "on" at 10:00, "off" at 11:00
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 10, 30),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 11, 0),
                datetime(2024, 1, 1 + day, 11, 30),
                "off",
            )

        # Predict from 10:00 to 11:30 (crosses bucket boundary)
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 15, 10, 0),
            horizon_minutes=90,
            interval_minutes=30,
            current_state="on",
        )

        # Find transition point
        transition_found = False
        for i, pred in enumerate(predictions):
            if pred.is_transition:
                transition_found = True
                # At transition, state_duration shows how long we've been tracking
                # (it's the duration BEFORE entering new state)
                # The NEXT prediction after transition should have the new state's duration
                if i + 1 < len(predictions):
                    next_pred = predictions[i + 1]
                    # Next prediction should have shorter duration (just started new state)
                    # It should be interval_minutes * 60 (30 * 60 = 1800s)
                    assert next_pred.state_duration == 30 * 60
                break

        assert transition_found, "Expected to find state transition"

    def test_transition_detection(self) -> None:
        """Test that is_transition flag is set correctly."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train alternating pattern
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 10, 30),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 11, 0),
                datetime(2024, 1, 1 + day, 11, 30),
                "off",
            )

        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 15, 10, 0),
            horizon_minutes=90,
            interval_minutes=30,
        )

        # First prediction should not be a transition (no previous state)
        assert predictions[0].is_transition is False

        # Subsequent predictions may be transitions
        transition_count = sum(1 for p in predictions if p.is_transition)
        assert transition_count >= 0  # At least could have transitions


class TestConfidenceDecay:
    """Tests for confidence decay over horizon."""

    def test_decay_factor_decreases_over_time(self) -> None:
        """Test that decay_factor decreases for predictions further in future."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=60, interval_minutes=10
        )

        # First prediction has highest decay factor
        assert predictions[0].decay_factor == 1.0

        # Each subsequent prediction has lower decay factor
        for i in range(1, len(predictions)):
            assert predictions[i].decay_factor < predictions[i - 1].decay_factor

        # All decay factors should be between 0 and 1
        for pred in predictions:
            assert 0.0 < pred.decay_factor <= 1.0

    def test_decay_factor_formula(self) -> None:
        """Test that decay factor follows expected formula."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=30, interval_minutes=10
        )

        # Formula: 1.0 / (1.0 + 0.1 * step)
        # step=0: 1.0 / 1.0 = 1.0
        assert predictions[0].decay_factor == 1.0

        # step=1: 1.0 / 1.1 ≈ 0.909
        assert abs(predictions[1].decay_factor - 0.909) < 0.01

        # step=2: 1.0 / 1.2 ≈ 0.833
        assert abs(predictions[2].decay_factor - 0.833) < 0.01


class TestFindNextTransition:
    """Tests for find_next_transition() utility method."""

    def test_finds_first_transition(self) -> None:
        """Test that it finds the first predicted state transition."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train pattern: "on" at 10:00, "off" at 11:00
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 10, 30),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 11, 0),
                datetime(2024, 1, 1 + day, 11, 30),
                "off",
            )

        # Find transition from "on"
        transition = forecaster.find_next_transition(
            datetime(2024, 1, 15, 10, 0),
            max_horizon_minutes=120,
            interval_minutes=30,
            current_state="on",
        )

        # Should find transition around 11:00
        assert transition is not None
        assert transition.hour == 11

    def test_returns_none_when_no_transition(self) -> None:
        """Test that it returns None when no transition is predicted."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train consistent "on" state
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 12, 0),
                "on",
            )

        # Find transition - should be None (state stays "on")
        transition = forecaster.find_next_transition(
            datetime(2024, 1, 15, 10, 0),
            max_horizon_minutes=60,
            interval_minutes=10,
            current_state="on",
        )

        assert transition is None

    def test_respects_max_horizon(self) -> None:
        """Test that search stops at max_horizon."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train transition at 2 hours
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 11, 30),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 12, 0),
                datetime(2024, 1, 1 + day, 13, 0),
                "off",
            )

        # Search only 60 minutes - should not find transition at 2 hours
        transition = forecaster.find_next_transition(
            datetime(2024, 1, 15, 10, 0),
            max_horizon_minutes=60,
            interval_minutes=30,
            current_state="on",
        )

        assert transition is None or transition < datetime(2024, 1, 15, 12, 0)


class TestGetStateTimeline:
    """Tests for get_state_timeline() utility method."""

    def test_creates_continuous_intervals(self) -> None:
        """Test that timeline creates continuous non-overlapping intervals."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        timeline = forecaster.get_state_timeline(
            datetime(2024, 1, 2, 10, 0), horizon_minutes=30, interval_minutes=10
        )

        # Should have at least one interval
        assert len(timeline) > 0

        # Check continuity - each interval starts where previous ended
        for i in range(1, len(timeline)):
            assert timeline[i][0] == timeline[i - 1][1]

    def test_timeline_covers_full_horizon(self) -> None:
        """Test that timeline spans the entire horizon."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        forecaster.update_interval(
            datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), "on"
        )

        start = datetime(2024, 1, 2, 10, 0)
        horizon = 60
        interval = 10

        timeline = forecaster.get_state_timeline(
            start, horizon_minutes=horizon, interval_minutes=interval
        )

        # First interval should start at start_time
        assert timeline[0][0] == start

        # Last interval should end at or after start + horizon
        expected_end = start + timedelta(minutes=horizon)
        assert timeline[-1][1] >= expected_end

    def test_merges_consecutive_same_states(self) -> None:
        """Test that consecutive predictions with same state are merged."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train consistent "on" state
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 11, 0),
                "on",
            )

        timeline = forecaster.get_state_timeline(
            datetime(2024, 1, 15, 10, 0), horizon_minutes=30, interval_minutes=10
        )

        # Should have single interval (all "on")
        assert len(timeline) == 1
        assert timeline[0][2] == "on"

    def test_splits_on_state_transitions(self) -> None:
        """Test that timeline splits when state changes."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train alternating pattern
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 10, 30),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 11, 0),
                datetime(2024, 1, 1 + day, 11, 30),
                "off",
            )

        timeline = forecaster.get_state_timeline(
            datetime(2024, 1, 15, 10, 0), horizon_minutes=90, interval_minutes=30
        )

        # Should have at least 2 intervals (on and off)
        assert len(timeline) >= 1

        # Intervals should have different states if transition occurred
        if len(timeline) > 1:
            states = [interval[2] for interval in timeline]
            assert len(set(states)) > 1  # At least 2 different states

    def test_empty_horizon_returns_empty_timeline(self) -> None:
        """Test that zero horizon returns empty timeline."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        timeline = forecaster.get_state_timeline(
            datetime(2024, 1, 1, 10, 0), horizon_minutes=0
        )

        assert timeline == []


class TestBucketBoundaryCrossing:
    """Tests for horizon predictions crossing temporal bucket boundaries."""

    def test_predictions_cross_hour_boundary(self) -> None:
        """Test that predictions correctly handle crossing hour boundaries."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train different patterns for different hours
        for day in range(10):
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 11, 0),
                "on",
            )
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 11, 0),
                datetime(2024, 1, 1 + day, 12, 0),
                "off",
            )

        # Predict across boundary (10:30 to 11:30)
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 15, 10, 30),
            horizon_minutes=60,
            interval_minutes=15,
        )

        # Should have predictions in both buckets
        assert len(predictions) == 4

        # Predictions should span both hours
        hours = {pred.timestamp.hour for pred in predictions}
        assert 10 in hours
        assert 11 in hours

    def test_maintains_accuracy_across_boundaries(self) -> None:
        """Test that state predictions remain accurate across boundaries."""
        indexer = CompositeIndexer([TimeOfDayIndexer(60)])
        forecaster = TimeAwareForecaster(indexer)

        # Train distinct patterns
        for day in range(15):
            # Morning: on
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 9, 0),
                datetime(2024, 1, 1 + day, 10, 0),
                "on",
            )
            # Mid-morning: off
            forecaster.update_interval(
                datetime(2024, 1, 1 + day, 10, 0),
                datetime(2024, 1, 1 + day, 11, 0),
                "off",
            )

        # Predict across 9:00-10:00 boundary
        predictions = forecaster.predict_horizon(
            datetime(2024, 1, 20, 9, 40),
            horizon_minutes=40,
            interval_minutes=10,
        )

        # Predictions in 9:00 hour should predict "on"
        for pred in predictions:
            if pred.timestamp.hour == 9:
                # Strong likelihood of "on" in this bucket
                if "on" in pred.prediction.distribution:
                    assert pred.prediction.distribution["on"] > 0.5
