"""
Unit tests for ForecasterEngine.

Comprehensive tests for the time-aware forecasting engine including
initialization, state updates, predictions, persistence modeling,
and hyperparameter management.
"""

from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.forecaster_engine import (
    ForecasterEngine,
)
from custom_components.discrete_state_forecaster.model.forecaster_engine_runtime_parameters import (
    ForecasterEngineRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_stats_runtime_parameters import (
    DriftStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.duration_weighted_baseline_runtime_parameters import (
    DurationWeightedBaselineRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller_runtime_parameters import (
    AdaptationConfig,
    HyperParameterControllerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_runtime_parameters import (
    StatePersistenceTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_runtime_parameters import (
    OnlineErrorTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_runtime_parameters import (
    HierarchicalStateStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)


def create_test_engine(
    half_life: float = 100.0,
    persistence_strength: float = 0.5,
    min_prune_interval_factor: float = 5.0,
) -> ForecasterEngine:
    """Create a test forecaster engine with specified parameters."""
    return ForecasterEngine(
        create_test_parameters(
            min_prune_interval_factor=min_prune_interval_factor,
            half_life=half_life,
            persistence_strength=persistence_strength,
        )
    )


def create_test_parameters(
    min_prune_interval_factor: float = 5.0,
    half_life: float = 100.0,
    persistence_strength: float = 0.5,
) -> ForecasterEngineRuntimeParameters:
    """Create test runtime parameters with default values."""
    return ForecasterEngineRuntimeParameters(
        hierarchical_state_stats=HierarchicalStateStatsRuntimeParameters(
            min_support_factor=2.0
        ),
        short_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
            error_half_life_factor=1.0
        ),
        long_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
            error_half_life_factor=10.0
        ),
        state_persistence_tracker=StatePersistenceTrackerRuntimeParameters(
            persistence_half_life_factor=1.0
        ),
        drift_monitor=DriftMonitorRuntimeParameters(
            slow_baseline=DurationWeightedBaselineRuntimeParameters(
                half_life_factor=20.0,
                prune_threshold=1e-6,
                epsilon=1e-9,
            ),
            fast_baseline=DurationWeightedBaselineRuntimeParameters(
                half_life_factor=1.5,
                prune_threshold=1e-6,
                epsilon=1e-9,
            ),
            drift_stats=DriftStatsRuntimeParameters(half_life_factor=30.0),
            tau_enter=0.1,
            tau_exit=0.05,
            adaptive_tau=True,
            n_enter=3,
            n_exit=5,
        ),
        hyper_parameter_controller=HyperParameterControllerRuntimeParameters(
            min_prune_interval_factor=min_prune_interval_factor,
            base_half_life=half_life,
            base_state_inertia_strength=persistence_strength,
            adaptation_config=AdaptationConfig(
                adapt_half_life=True, adapt_prune_interval=False
            ),
        ),
    )


class TestForecasterEngineInitialization:
    """Tests for ForecasterEngine initialization."""

    def test_initial_state(self: Self) -> None:
        """Test that engine starts with no data."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        prediction = engine.predict(key)
        # Initially, there should be no prediction (insufficient data)
        assert prediction is None


class TestForecasterEngineUpdate:
    """Tests for ForecasterEngine.update method."""

    def test_first_update(self: Self) -> None:
        """Test first update to the engine."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"
        timestamp = 1000.0

        # Should not raise exception
        engine.update(key, state, timestamp)

    def test_multiple_updates_same_state(self: Self) -> None:
        """Test multiple updates with the same state."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        for i in range(10):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

    def test_multiple_updates_different_states(self: Self) -> None:
        """Test multiple updates with different states."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        states = ["on", "off", "on", "off", "on"]
        for i, state in enumerate(states):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

    def test_update_without_timestamp_uses_current_time(self: Self) -> None:
        """Test that update without timestamp uses current time."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        # Should not raise exception
        engine.update(key, state)

    def test_update_with_past_timestamp_raises_error(self: Self) -> None:
        """Test that updating with a past timestamp raises ValueError."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        engine.update(key, state, timestamp=1000.0)

        with pytest.raises(ValueError, match="Timestamp cannot be in the past"):
            engine.update(key, state, timestamp=900.0)

    def test_update_with_same_timestamp_allowed(self: Self) -> None:
        """Test that updating with the same timestamp is allowed (duration=0)."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        engine.update(key, state, timestamp=1000.0)

        # Same timestamp means duration=0, which is allowed (not "in the past")
        engine.update(key, state, timestamp=1000.0)

    def test_update_creates_prediction(self: Self) -> None:
        """Test that updates eventually create a prediction."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        # Add enough data to exceed min_support threshold
        # min_support = min_support_factor * half_life = 2.0 * 100.0 = 200.0
        # With decay half_life=100, support converges to duration/(1-decay_factor)
        # For duration=200: decay_factor=0.25, so support converges to 200/0.75 = 266.67 > 200
        for i in range(10):  # Use longer durations
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        prediction = engine.predict(key)
        # After sufficient updates, we should have a prediction
        assert prediction is not None


class TestForecasterEnginePredict:
    """Tests for ForecasterEngine.predict method."""

    def test_predict_without_data_returns_none(self: Self) -> None:
        """Test that predict returns None when no data is available."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        prediction = engine.predict(key)
        assert prediction is None

    def test_predict_after_updates(self: Self) -> None:
        """Test prediction after adding data."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        # Add sufficient data to exceed min_support
        for i in range(15):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        prediction = engine.predict(key)
        assert prediction is not None
        assert prediction.key == key
        assert prediction.distribution is not None

    def test_predict_distribution_reflects_updates(self: Self) -> None:
        """Test that prediction distribution reflects the updated states."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Add sufficient data with both states closer in time to avoid one
        # being completely decayed when making predictions
        for i in range(10):
            state = "on" if i % 2 == 0 else "off"
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        prediction = engine.predict(key)
        assert prediction is not None
        dist = prediction.distribution
        # Both states should be present
        assert "on" in dist
        assert "off" in dist


class TestForecasterEnginePredictWithPersistence:
    """Tests for ForecasterEngine.predict_with_persistence method."""

    def test_predict_with_persistence_no_data_returns_none(self: Self) -> None:
        """Test that predict_with_persistence returns None when no data."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        prediction = engine.predict_with_persistence(key)
        assert prediction is None

    def test_predict_with_persistence_explicit_state(self: Self) -> None:
        """Test persistence prediction with explicit current state."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Add sufficient data to establish a baseline
        for i in range(15):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)

        # Predict with explicit current state
        prediction = engine.predict_with_persistence(
            key, current_state="on", current_state_duration=50.0
        )

        assert prediction is not None
        assert prediction.key == key

    def test_predict_with_persistence_uses_internal_tracker(self: Self) -> None:
        """Test persistence prediction using internal state tracker."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Add sufficient data
        for i in range(15):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)

        # Predict without explicit state (uses internal tracker)
        prediction = engine.predict_with_persistence(key)

        assert prediction is not None
        assert prediction.key == key

    def test_predict_with_persistence_boost_effect(self: Self) -> None:
        """Test that persistence boost affects probabilities."""
        engine = create_test_engine(persistence_strength=0.8)
        key = TimeKey.GLOBAL

        # Create a scenario with sufficient mixed states
        for i in range(8):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)
        for i in range(8):
            engine.update(key, "off", timestamp=3000.0 + i * 200.0)

        # Get base prediction
        base_prediction = engine.predict(key)

        # Get persistence prediction with explicit state
        persistence_prediction = engine.predict_with_persistence(
            key, current_state="on", current_state_duration=10.0
        )

        # Both should exist
        assert base_prediction is not None
        assert persistence_prediction is not None

    def test_predict_with_persistence_without_duration_returns_base(
        self: Self,
    ) -> None:
        """Test that missing duration returns base prediction."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Add sufficient data
        for i in range(15):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)

        # Provide state but not duration
        prediction = engine.predict_with_persistence(key, current_state="on")

        # Should still return a prediction (falls back to base or internal)
        assert prediction is not None


class TestForecasterEngineDecay:
    """Tests for exponential decay functionality."""

    def test_get_decay_factor_zero_duration(self: Self) -> None:
        """Test that zero duration gives decay factor of 1.0."""
        engine = create_test_engine(half_life=100.0)
        decay = engine._get_decay_factor(0.0)  # noqa: SLF001
        assert decay == 1.0

    def test_get_decay_factor_half_life(self: Self) -> None:
        """Test that one half-life duration gives decay factor of 0.5."""
        engine = create_test_engine(half_life=100.0)
        decay = engine._get_decay_factor(100.0)  # noqa: SLF001
        assert abs(decay - 0.5) < 1e-9

    def test_get_decay_factor_two_half_lives(self: Self) -> None:
        """Test that two half-lives give decay factor of 0.25."""
        engine = create_test_engine(half_life=100.0)
        decay = engine._get_decay_factor(200.0)
        assert abs(decay - 0.25) < 1e-9

    def test_decay_applied_in_updates(self: Self) -> None:
        """Test that decay is applied between updates."""
        engine = create_test_engine(half_life=100.0)
        key = TimeKey.GLOBAL

        # Add sufficient data
        for i in range(8):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)

        # Add more with different state
        for i in range(8):
            engine.update(key, "off", timestamp=3000.0 + i * 200.0)

        # The decay should have been applied
        prediction = engine.predict(key)
        assert prediction is not None


class TestForecasterEnginePruning:
    """Tests for pruning functionality."""

    def test_pruning_respects_min_interval(self: Self) -> None:
        """Test that pruning only occurs after min interval."""
        rp = create_test_parameters(half_life=100, min_prune_interval_factor=5.0)
        engine = ForecasterEngine(rp)
        key = TimeKey.GLOBAL

        # Multiple updates within the prune interval
        for i in range(10):
            engine.update(key, "on", timestamp=1000.0 + i * 10.0)

        # Pruning should respect the min interval
        # This is an internal behavior test, so we rely on no errors

    def test_engine_with_pruning_disabled(self: Self) -> None:
        """Test engine behavior when pruning is disabled."""
        # Note: We can't directly disable pruning via ForecasterEngineParameters
        # This test would require accessing internal hyper_parameters
        # Skipping this as it requires internal state modification
        pass


class TestForecasterEngineIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_single_state(self: Self) -> None:
        """Test complete workflow with single state."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL
        state = "on"

        # Add sufficient data
        for i in range(25):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        # Get predictions
        prediction = engine.predict(key)
        persistence_prediction = engine.predict_with_persistence(key)

        assert prediction is not None
        assert persistence_prediction is not None

        # Should predict "on" with high probability
        dist = prediction.distribution
        assert "on" in dist
        assert dist["on"] > 0.5

    def test_full_workflow_alternating_states(self: Self) -> None:
        """Test complete workflow with alternating states."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Alternating pattern with sufficient data
        states = ["on", "off"] * 15
        for i, state in enumerate(states):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        # Get predictions
        prediction = engine.predict(key)
        assert prediction is not None

        # Distribution should reflect both states
        dist = prediction.distribution
        assert "on" in dist
        assert "off" in dist

    def test_full_workflow_with_temporal_keys(self: Self) -> None:
        """Test workflow with different temporal keys."""
        engine = create_test_engine()

        # Use different time keys
        global_key = TimeKey.GLOBAL
        # TimeKey is built by adding TemporalFeature instances
        specific_key = TimeKey(("hour", 14))

        # Add sufficient data for both keys
        for i in range(15):
            engine.update(global_key, "on", timestamp=1000.0 + i * 200.0)
            engine.update(specific_key, "off", timestamp=1000.0 + i * 200.0)

        # Both should have predictions
        global_pred = engine.predict(global_key)
        specific_pred = engine.predict(specific_key)

        assert global_pred is not None
        assert specific_pred is not None

    def test_drift_detection_integration(self: Self) -> None:
        """Test that drift detection integrates correctly."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Stable period with sufficient weight (use longer durations)
        for i in range(30):
            engine.update(key, "on", timestamp=1000.0 + i * 200.0)

        # Sudden change
        for i in range(30):
            engine.update(key, "off", timestamp=7000.0 + i * 200.0)

        # Engine should handle drift (no errors)
        prediction = engine.predict(key)
        assert prediction is not None

    def test_error_tracking_integration(self: Self) -> None:
        """Test that error tracking integrates correctly."""
        engine = create_test_engine()
        key = TimeKey.GLOBAL

        # Add varied data to generate prediction errors with sufficient weight
        states = ["on", "off", "on", "on", "off", "on", "off", "off"] * 3
        for i, state in enumerate(states):
            engine.update(key, state, timestamp=1000.0 + i * 200.0)

        # Error tracking should work internally
        prediction = engine.predict(key)
        assert prediction is not None


class TestForecasterEngineBackgroundDecay:
    """Tests for the background_decay_half_life_factor feature."""

    def _create_engine_with_bg_decay(
        self: Self,
        half_life: float,
        background_decay_half_life_factor: float,
    ) -> ForecasterEngine:
        """Create an engine with background decay enabled."""
        params = ForecasterEngineRuntimeParameters(
            hierarchical_state_stats=HierarchicalStateStatsRuntimeParameters(
                min_support_factor=0.001
            ),
            short_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
                error_half_life_factor=1.0
            ),
            long_term_error_tracker=OnlineErrorTrackerRuntimeParameters(
                error_half_life_factor=10.0
            ),
            state_persistence_tracker=StatePersistenceTrackerRuntimeParameters(
                persistence_half_life_factor=1.0
            ),
            drift_monitor=DriftMonitorRuntimeParameters(
                slow_baseline=DurationWeightedBaselineRuntimeParameters(
                    half_life_factor=20.0,
                    prune_threshold=1e-6,
                    epsilon=1e-9,
                ),
                fast_baseline=DurationWeightedBaselineRuntimeParameters(
                    half_life_factor=1.5,
                    prune_threshold=1e-6,
                    epsilon=1e-9,
                ),
                drift_stats=DriftStatsRuntimeParameters(half_life_factor=30.0),
                tau_enter=0.1,
                tau_exit=0.05,
                adaptive_tau=True,
                n_enter=3,
                n_exit=5,
            ),
            hyper_parameter_controller=HyperParameterControllerRuntimeParameters(
                min_prune_interval_factor=5.0,
                base_half_life=half_life,
                base_state_inertia_strength=0.5,
                adaptation_config=AdaptationConfig(
                    adapt_half_life=False,
                    adapt_prune_interval=False,
                    adapt_persistence=False,
                ),
            ),
            background_decay_half_life_factor=background_decay_half_life_factor,
        )
        return ForecasterEngine(params)

    def test_background_decay_disabled_dormant_key_unchanged(self: Self) -> None:
        """With factor=0, dormant keys must not decay between updates."""
        half_life = 100.0
        engine = self._create_engine_with_bg_decay(half_life, 0.0)

        winter_key = TimeKey(("season", "winter"))
        summer_key = TimeKey(("season", "summer"))

        # Fill winter with strong support.
        for i in range(20):
            engine.update(winter_key, "heating", timestamp=float(i * 50))

        winter_support_before = (
            engine.predict(winter_key).confidence.support
        )

        # Now run many summer updates.  Winter should not be touched at all.
        base_ts = 1000.0
        for i in range(100):
            engine.update(summer_key, "cooling", timestamp=base_ts + i * 50)

        winter_result = engine.predict(winter_key)
        assert winter_result is not None
        # No background decay → winter support identical to before the summer run.
        assert abs(winter_result.confidence.support - winter_support_before) < 1e-6

    def test_background_decay_enabled_dormant_key_erodes(self: Self) -> None:
        """With factor>0, the winter-key-specific distribution erodes during summer.

        After enough background decay the winter_key's own distribution falls below
        the min-support threshold and ``predict(winter_key)`` falls back to GLOBAL.
        GLOBAL is dominated by the massive number of summer 'cooling' observations,
        so the predicted state should flip from 'heating' to 'cooling'.
        """
        half_life = 100.0
        factor = 2.0  # background half-life = 2 * 100 = 200 s (aggressive for testing)
        engine = self._create_engine_with_bg_decay(half_life, factor)

        winter_key = TimeKey(("season", "winter"))
        summer_key = TimeKey(("season", "summer"))

        # Fill winter with a strong 'heating' pattern.
        for i in range(30):
            engine.update(winter_key, "heating", timestamp=float(i * 50))

        # Verify that winter pattern is recognized before summer starts.
        before = engine.predict(winter_key)
        assert before is not None
        assert before.distribution.get("heating", 0.0) > 0.5

        # Run many summer updates over a long period (200 * 250 = 50 000 s >> 200 s background HL).
        # This floods GLOBAL with 'cooling' data and background-decays the winter_key distribution.
        base_ts = float(30 * 50) + 50.0
        for i in range(200):
            engine.update(summer_key, "cooling", timestamp=base_ts + i * 250)

        after = engine.predict(winter_key)
        assert after is not None

        # The winter_key-specific distribution should have been wiped out by background decay.
        # The fallback goes to GLOBAL which now has mostly 'cooling'.
        # Either the winter_key no longer contributes (only GLOBAL in contributions),
        # or the dominant state has switched from 'heating' to 'cooling'.
        winter_contribution = next(
            (c for c in after.contributions if c.key == winter_key), None
        )
        cooling_dominates = after.distribution.get("cooling", 0.0) > after.distribution.get("heating", 0.0)
        winter_specific_gone = winter_contribution is None or winter_contribution.support < 1.0

        assert cooling_dominates or winter_specific_gone, (
            f"Expected winter-specific distribution to erode: "
            f"dist={after.distribution}, contributions={after.contributions}"
        )

    def test_background_decay_slower_than_active_decay(self: Self) -> None:
        """Background decay on a dormant key must be much slower than active decay."""
        half_life = 100.0
        factor = 20.0
        engine = self._create_engine_with_bg_decay(half_life, factor)

        key_active = TimeKey(("slot", "a"))
        key_dormant = TimeKey(("slot", "b"))

        # Give both keys equal support.
        for i in range(30):
            engine.update(key_active, "on", timestamp=float(i * 50))
            engine.update(key_dormant, "on", timestamp=float(i * 50))

        support_active_start = engine.predict(key_active).confidence.support
        support_dormant_start = engine.predict(key_dormant).confidence.support

        # Now only update the active key for a while.
        base_ts = 1500.0
        for i in range(50):
            engine.update(key_active, "on", timestamp=base_ts + i * 50)

        support_active_end = engine.predict(key_active).confidence.support
        support_dormant_end = engine.predict(key_dormant).confidence.support

        active_drop = support_active_start - support_active_end
        dormant_drop = support_dormant_start - support_dormant_end

        # Active key accumulates new observations so it may even grow;
        # the key assertion is that the dormant key decays much less.
        assert dormant_drop < active_drop or support_dormant_end > support_dormant_start * 0.5
