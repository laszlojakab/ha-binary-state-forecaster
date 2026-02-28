"""
Unit tests for HyperParameterController.

Comprehensive tests for adaptive hyper-parameter control based on drift
and error signals.
"""

import math

from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller import (
    AdaptationMode,
    HyperParameterController,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller_runtime_parameters import (
    AdaptationConfig,
    HyperParameterControllerRuntimeParameters,
)


def create_test_runtime_parameters(  # noqa: PLR0913
    base_half_life: float = 300.0,
    base_state_inertia_strength: float = 0.95,
    min_prune_interval_factor: float = 5.0,
    min_half_life: float = 60.0,
    max_half_life: float = 3600.0 * 48,
    adapt_half_life: bool = True,
    adapt_persistence: bool = True,
    adapt_prune_interval: bool = True,
) -> HyperParameterControllerRuntimeParameters:
    """Create test runtime parameters."""
    return HyperParameterControllerRuntimeParameters(
        base_half_life=base_half_life,
        adaptation_config=AdaptationConfig(
            adapt_half_life=adapt_half_life,
            adapt_persistence=adapt_persistence,
            adapt_prune_interval=adapt_prune_interval,
        ),
        base_state_inertia_strength=base_state_inertia_strength,
        min_prune_interval_factor=min_prune_interval_factor,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
    )


class TestHyperParameterControllerInitialization:
    """Tests for HyperParameterController initialization."""

    def test_initialization_with_default_parameters(self) -> None:
        """Test initialization with default runtime parameters."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        assert controller.hyper_parameters.half_life == 300.0
        assert controller.hyper_parameters.persistence_strength == 0.95
        assert controller.hyper_parameters.prune_enabled is True
        assert controller.mode == AdaptationMode.STABLE

    def test_initialization_with_custom_parameters(self) -> None:
        """Test initialization with custom runtime parameters."""
        rp = create_test_runtime_parameters(
            base_half_life=500.0,
            base_state_inertia_strength=0.8,
            min_prune_interval_factor=10.0,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        assert controller.hyper_parameters.half_life == 500.0
        assert controller.hyper_parameters.persistence_strength == 0.8
        assert controller.hyper_parameters.min_prune_interval_factor == 10.0

    def test_initialization_sets_bounds_correctly(self) -> None:
        """Test that initialization sets min/max bounds correctly."""
        rp = create_test_runtime_parameters(
            min_half_life=100.0,
            max_half_life=1000.0,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        # Access private attributes for testing
        assert controller._min_half_life == 100.0
        assert controller._max_half_life == 1000.0


class TestAdaptationModes:
    """Tests for adaptation mode selection."""

    def test_stable_mode_no_drift_no_errors(self) -> None:
        """Test STABLE mode when no drift and no error increase."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.mode == AdaptationMode.STABLE

    def test_drifting_ok_mode_drift_but_no_errors(self) -> None:
        """Test DRIFTING_OK mode when drift detected but errors stable."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=True,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.mode == AdaptationMode.DRIFTING_OK

    def test_model_degrading_mode_errors_but_no_drift(self) -> None:
        """Test MODEL_DEGRADING mode when errors increase but no drift."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.mode == AdaptationMode.MODEL_DEGRADING

    def test_concept_drift_mode_drift_and_errors(self) -> None:
        """Test CONCEPT_DRIFT mode when both drift and errors increase."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.mode == AdaptationMode.CONCEPT_DRIFT

    def test_low_confidence_triggers_model_degrading(self) -> None:
        """Test that low confidence triggers MODEL_DEGRADING mode."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.3,  # Low confidence
        )

        assert controller.mode == AdaptationMode.MODEL_DEGRADING

    def test_deep_fallback_triggers_drifting_ok(self) -> None:
        """Test that deep fallback triggers DRIFTING_OK mode."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
            fallback_depth=3,  # Deep fallback
        )

        assert controller.mode == AdaptationMode.DRIFTING_OK

    def test_none_error_values_handled_correctly(self) -> None:
        """Test that None error values are handled without errors."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=None,
            long_term_error=None,
            entropy_confidence=0.8,
        )

        assert controller.mode == AdaptationMode.STABLE


class TestHalfLifeAdaptation:
    """Tests for half-life adaptation."""

    def test_concept_drift_decreases_half_life_aggressively(self) -> None:
        """Test that concept drift decreases half-life aggressively."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        initial_half_life = controller.hyper_parameters.half_life

        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.hyper_parameters.half_life < initial_half_life

    def test_stable_mode_increases_half_life(self) -> None:
        """Test that stable mode increases half-life over time."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        # First decrease it
        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        decreased_half_life = controller.hyper_parameters.half_life

        # Then stabilize
        for _ in range(10):
            controller.update(
                is_drifting=False,
                short_term_error=0.1,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        assert controller.hyper_parameters.half_life > decreased_half_life

    def test_half_life_respects_min_bound(self) -> None:
        """Test that half-life doesn't go below minimum bound."""
        rp = create_test_runtime_parameters(
            base_half_life=100.0,
            min_half_life=60.0,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        # Trigger aggressive decrease many times
        for _ in range(100):
            controller.update(
                is_drifting=True,
                short_term_error=0.9,
                long_term_error=0.1,
                entropy_confidence=0.1,
            )

        # Use isclose to account for floating-point precision
        assert controller.hyper_parameters.half_life >= 60.0 - 1e-9

    def test_half_life_respects_max_bound(self) -> None:
        """Test that half-life doesn't go above maximum bound."""
        rp = create_test_runtime_parameters(
            base_half_life=1000.0,
            max_half_life=2000.0,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        # Trigger increase many times
        for _ in range(200):
            controller.update(
                is_drifting=False,
                short_term_error=0.05,
                long_term_error=0.05,
                entropy_confidence=0.9,
            )

        assert controller.hyper_parameters.half_life <= 2000.0


class TestPersistenceAdaptation:
    """Tests for persistence strength adaptation."""

    def test_persistence_adapts_with_half_life(self) -> None:
        """Test that persistence strength adapts based on half-life."""
        rp = create_test_runtime_parameters(
            base_half_life=300.0,
            adapt_persistence=True,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        initial_persistence = controller.hyper_parameters.persistence_strength

        # Trigger concept drift (decreases half-life)
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Lower half-life should result in lower persistence
        assert controller.hyper_parameters.persistence_strength < initial_persistence

    def test_persistence_not_adapted_when_disabled(self) -> None:
        """Test that persistence doesn't adapt when adaptation is disabled."""
        rp = create_test_runtime_parameters(
            base_half_life=300.0,
            base_state_inertia_strength=0.95,
            adapt_persistence=False,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        # Trigger concept drift
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Persistence should remain unchanged
        assert controller.hyper_parameters.persistence_strength == 0.95


class TestPruneIntervalAdaptation:
    """Tests for pruning interval adaptation."""

    def test_concept_drift_disables_pruning(self) -> None:
        """Test that concept drift disables pruning."""
        rp = create_test_runtime_parameters(adapt_prune_interval=True)
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.hyper_parameters.prune_enabled is False

    def test_stable_mode_enables_pruning(self) -> None:
        """Test that stable mode enables pruning."""
        rp = create_test_runtime_parameters(adapt_prune_interval=True)
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.hyper_parameters.prune_enabled is True

    def test_prune_interval_not_adapted_when_disabled(self) -> None:
        """Test that pruning doesn't adapt when adaptation is disabled."""
        rp = create_test_runtime_parameters(adapt_prune_interval=False)
        controller = HyperParameterController(runtime_parameters=rp)

        initial_prune_enabled = controller.hyper_parameters.prune_enabled

        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        assert controller.hyper_parameters.prune_enabled == initial_prune_enabled


class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_contains_all_fields(self) -> None:
        """Test that to_dict includes all necessary fields."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        data = controller.to_dict()

        assert "mode" in data
        assert "hyper_parameters" in data
        assert data["mode"] == "CONCEPT_DRIFT"

    def test_from_dict_restores_state(self) -> None:
        """Test that from_dict correctly restores controller state."""
        rp = create_test_runtime_parameters()
        controller1 = HyperParameterController(runtime_parameters=rp)

        # Modify state
        controller1.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )

        # Serialize
        data = controller1.to_dict()

        # Deserialize
        controller2 = HyperParameterController.from_dict(data, rp)

        assert controller2.mode == controller1.mode
        assert (
            controller2.hyper_parameters.half_life
            == controller1.hyper_parameters.half_life
        )
        assert (
            controller2.hyper_parameters.persistence_strength
            == controller1.hyper_parameters.persistence_strength
        )
        assert (
            controller2.hyper_parameters.prune_enabled
            == controller1.hyper_parameters.prune_enabled
        )

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization roundtrip preserves state."""
        rp = create_test_runtime_parameters()
        controller1 = HyperParameterController(runtime_parameters=rp)

        # Apply multiple updates
        updates = [
            (True, 0.2, 0.1, 0.8, None),
            (False, 0.15, 0.12, 0.7, None),
            (False, 0.1, 0.1, 0.9, None),
        ]

        for is_drift, st_err, lt_err, conf, fb_depth in updates:
            controller1.update(
                is_drifting=is_drift,
                short_term_error=st_err,
                long_term_error=lt_err,
                entropy_confidence=conf,
                fallback_depth=fb_depth,
            )

        # Serialize and deserialize
        data = controller1.to_dict()
        controller2 = HyperParameterController.from_dict(data, rp)

        # Verify state matches
        assert controller2.mode == controller1.mode
        assert math.isclose(
            controller2.hyper_parameters.half_life,
            controller1.hyper_parameters.half_life,
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_updates_same_mode(self) -> None:
        """Test multiple updates in the same adaptation mode."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        initial_half_life = controller.hyper_parameters.half_life

        # Apply stable mode updates
        for _ in range(5):
            controller.update(
                is_drifting=False,
                short_term_error=0.1,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Half-life should increase over stable updates
        assert controller.hyper_parameters.half_life > initial_half_life

    def test_mode_transitions(self) -> None:
        """Test transitions between different modes."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        # Start stable
        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )
        assert controller.mode == AdaptationMode.STABLE

        # Transition to drifting
        controller.update(
            is_drifting=True,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )
        assert controller.mode == AdaptationMode.DRIFTING_OK

        # Transition to concept drift
        controller.update(
            is_drifting=True,
            short_term_error=0.2,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )
        assert controller.mode == AdaptationMode.CONCEPT_DRIFT

        # Back to stable
        controller.update(
            is_drifting=False,
            short_term_error=0.1,
            long_term_error=0.1,
            entropy_confidence=0.8,
        )
        assert controller.mode == AdaptationMode.STABLE

    def test_zero_half_life_bounds(self) -> None:
        """Test behavior with very small half-life bounds."""
        rp = create_test_runtime_parameters(
            base_half_life=10.0,
            min_half_life=1.0,
            max_half_life=20.0,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        # Should not crash
        controller.update(
            is_drifting=True,
            short_term_error=0.9,
            long_term_error=0.1,
            entropy_confidence=0.1,
        )

        assert controller.hyper_parameters.half_life >= 1.0

    def test_all_none_optional_parameters(self) -> None:
        """Test update with all optional parameters as None."""
        rp = create_test_runtime_parameters()
        controller = HyperParameterController(runtime_parameters=rp)

        # Should not crash
        controller.update(
            is_drifting=False,
            short_term_error=None,
            long_term_error=None,
            entropy_confidence=None,
            fallback_depth=None,
        )

        assert controller.mode == AdaptationMode.STABLE


class TestBaselineAdaptation:
    """Tests for baseline half-life adaptation."""

    def test_baseline_initializes_to_current_half_life(self) -> None:
        """Test that baseline starts equal to current half-life."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        # Access private attribute for testing
        assert controller._baseline_half_life == controller._half_life

    def test_baseline_adapts_in_stable_mode(self) -> None:
        """Test that baseline adapts toward current half-life in stable mode."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        # First decrease half-life
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        initial_baseline = controller._baseline_half_life
        decreased_half_life = controller._half_life

        # Now stabilize - baseline should slowly approach current half-life
        for _ in range(100):
            controller.update(
                is_drifting=False,
                short_term_error=0.1,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        final_baseline = controller._baseline_half_life
        final_half_life = controller._half_life

        # Baseline should have moved toward decreased value (but not all the way)
        assert final_baseline < initial_baseline
        # Baseline should be between initial and current
        assert decreased_half_life < final_baseline < initial_baseline
        # Current half-life should have increased from the decreased level
        assert final_half_life > decreased_half_life

    def test_baseline_does_not_adapt_in_drift_mode(self) -> None:
        """Test that baseline doesn't adapt when not in stable mode."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        initial_baseline = controller._baseline_half_life

        # Apply drift updates
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        final_baseline = controller._baseline_half_life

        # Baseline should not have changed
        assert final_baseline == initial_baseline

    def test_baseline_serialization(self) -> None:
        """Test that baseline is included in serialization."""
        rp = create_test_runtime_parameters(base_half_life=300.0)
        controller = HyperParameterController(runtime_parameters=rp)

        # Modify state
        for _ in range(5):
            controller.update(
                is_drifting=False,
                short_term_error=0.1,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Serialize
        data = controller.to_dict()

        # Should include baseline
        assert "baseline_half_life" in data
        assert "half_life" in data

        # Deserialize
        controller2 = HyperParameterController.from_dict(data, rp)

        # Baselines should match
        assert controller2._baseline_half_life == controller._baseline_half_life
        assert controller2._half_life == controller._half_life


class TestAdaptationConfiguration:
    """Tests for adaptation configuration options."""

    def test_disable_half_life_adaptation(self) -> None:
        """Test that half-life doesn't adapt when disabled."""
        rp = create_test_runtime_parameters(
            base_half_life=300.0,
            adapt_half_life=False,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        initial_half_life = controller.hyper_parameters.half_life

        # Trigger concept drift
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Half-life should not have changed
        assert math.isclose(
            controller.hyper_parameters.half_life, initial_half_life, rel_tol=1e-9
        )

    def test_all_adaptations_disabled(self) -> None:
        """Test that all parameters stay constant when adaptation is disabled."""
        rp = create_test_runtime_parameters(
            base_half_life=300.0,
            base_state_inertia_strength=0.95,
            min_prune_interval_factor=5.0,
            adapt_half_life=False,
            adapt_persistence=False,
            adapt_prune_interval=False,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        initial_half_life = controller.hyper_parameters.half_life
        initial_persistence = controller.hyper_parameters.persistence_strength
        initial_prune_enabled = controller.hyper_parameters.prune_enabled

        # Apply various updates
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # All parameters should remain unchanged
        assert math.isclose(
            controller.hyper_parameters.half_life, initial_half_life, rel_tol=1e-9
        )
        assert controller.hyper_parameters.persistence_strength == initial_persistence
        assert controller.hyper_parameters.prune_enabled == initial_prune_enabled

    def test_selective_adaptation(self) -> None:
        """Test that individual adaptations can be enabled/disabled independently."""
        rp = create_test_runtime_parameters(
            base_half_life=300.0,
            adapt_half_life=True,
            adapt_persistence=False,
            adapt_prune_interval=False,
        )
        controller = HyperParameterController(runtime_parameters=rp)

        initial_half_life = controller.hyper_parameters.half_life
        initial_persistence = controller.hyper_parameters.persistence_strength

        # Trigger drift
        for _ in range(10):
            controller.update(
                is_drifting=True,
                short_term_error=0.2,
                long_term_error=0.1,
                entropy_confidence=0.8,
            )

        # Half-life should have changed
        assert controller.hyper_parameters.half_life != initial_half_life
        # But persistence should not
        assert controller.hyper_parameters.persistence_strength == initial_persistence
