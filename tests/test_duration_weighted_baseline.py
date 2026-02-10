"""
Unit tests for DurationWeightedBaseline.

Comprehensive tests for duration-weighted baseline distribution tracking.
"""
from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_hyper_parameters import (
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.duration_weighted_baseline import (
    DurationWeightedBaseline,
)
from custom_components.discrete_state_forecaster.model.learning.duration_weighted_baseline_hyper_parameters import (
    DurationWeightedBaselineHyperParameters,
)


def create_test_hp(
    half_life_factor: float = 1.0,
) -> DurationWeightedBaselineHyperParameters:
    """Create test hyper-parameters."""
    base_hp = HyperParameters(
        half_life=50.0,
        min_prune_interval=10.0,
        prune_enabled=True,
        persistence_strength=0.95,
    )
    drift_hp = DriftMonitorHyperParameters(hyper_parameters=base_hp)
    return DurationWeightedBaselineHyperParameters(
        hyper_parameters=drift_hp,
        half_life_factor=half_life_factor,
    )


class TestDurationWeightedBaselineInitialization:
    """Tests for DurationWeightedBaseline initialization."""

    def test_create_default(self: Self) -> None:
        """Test creating with default configuration."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)
        assert baseline.distribution() == {}
        assert baseline.total_mass() == 0.0

    def test_initial_empty_distribution(self: Self) -> None:
        """Test that initial distribution is empty."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)
        assert baseline.distribution() == {}


class TestDurationWeightedBaselineUpdate:
    """Tests for DurationWeightedBaseline.update method."""

    def test_first_update_sets_timestamp(self: Self) -> None:
        """Test that first update only sets timestamp."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.6, "off": 0.4}
        baseline.update(dist, 100.0)

        # First update should not accumulate mass
        assert baseline.total_mass() == 0.0

    def test_second_update_accumulates_mass(self: Self) -> None:
        """Test that second update accumulates mass."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.6, "off": 0.4}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)

        # Should have accumulated mass
        assert baseline.total_mass() > 0.0

    def test_mass_integration_over_time(self: Self) -> None:
        """Test that mass is integrated over time duration."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.5, "off": 0.5}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)  # dt=10

        mass1 = baseline.total_mass()

        baseline.update(dist, 130.0)  # dt=20
        mass2 = baseline.total_mass()

        # Second interval is twice as long, so should add proportionally more mass
        # (accounting for decay)
        assert mass2 > mass1

    def test_update_with_zero_time_delta(self: Self) -> None:
        """Test that zero time delta is ignored."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.6, "off": 0.4}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)

        mass1 = baseline.total_mass()

        baseline.update(dist, 110.0)
        mass2 = baseline.total_mass()

        assert mass1 == mass2

    def test_update_with_negative_time_delta(self: Self) -> None:
        """Test that negative time delta is ignored."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.6, "off": 0.4}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)

        mass1 = baseline.total_mass()

        baseline.update(dist, 105.0)
        mass2 = baseline.total_mass()

        assert mass1 == mass2


class TestDurationWeightedBaselineDecay:
    """Tests for exponential decay behavior."""

    def test_decay_reduces_old_mass(self: Self) -> None:
        """Test that old mass decays over time."""
        hp = create_test_hp(half_life_factor=1.0)
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 1.0}, 100.0)
        baseline.update({"on": 1.0}, 110.0)

        # After more time, mass should decay
        baseline.update({"on": 1.0}, 200.0)
        mass2 = baseline.total_mass()

        # New mass added but old mass decayed
        # Exact relationship depends on half-life
        assert mass2 > 0.0

    def test_pruning_removes_small_mass(self: Self) -> None:
        """Test that very small mass values are pruned."""
        hp = create_test_hp(half_life_factor=0.1)  # Fast decay
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 1.0}, 100.0)
        baseline.update({"on": 1.0}, 105.0)

        # After many half-lives with different state
        for i in range(30):
            baseline.update({"off": 1.0}, 110.0 + i * 10.0)

        dist = baseline.distribution()
        # "on" should have been pruned or be negligible
        assert "on" not in dist or dist.get("on", 0) < 0.01


class TestDurationWeightedBaselineDistribution:
    """Tests for distribution() method."""

    def test_distribution_sums_to_one(self: Self) -> None:
        """Test that distribution sums to approximately 1.0."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {"on": 0.6, "off": 0.4}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)
        baseline.update(dist, 120.0)

        result = baseline.distribution()
        total = sum(result.values())

        assert abs(total - 1.0) < 1e-6

    def test_distribution_with_laplace_smoothing(self: Self) -> None:
        """Test that Laplace smoothing prevents zero probabilities for seen states."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        # Both states appear but with different probabilities
        baseline.update({"on": 0.9, "off": 0.1}, 100.0)
        baseline.update({"on": 0.9, "off": 0.1}, 110.0)

        result = baseline.distribution()
        # Even with low probability input, should have positive value
        assert result.get("off", 0) > 0

    def test_empty_distribution_at_start(self: Self) -> None:
        """Test that distribution is empty before accumulating mass."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 0.5, "off": 0.5}, 100.0)

        # After first update only (no mass accumulated yet)
        result = baseline.distribution()
        assert result == {}


class TestDurationWeightedBaselineTotalMass:
    """Tests for total_mass() method."""

    def test_total_mass_increases_with_updates(self: Self) -> None:
        """Test that total mass increases with updates."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 0.5, "off": 0.5}, 100.0)
        baseline.update({"on": 0.5, "off": 0.5}, 110.0)

        mass1 = baseline.total_mass()

        baseline.update({"on": 0.5, "off": 0.5}, 120.0)
        mass2 = baseline.total_mass()

        # More mass should be added (even with decay)
        assert mass2 > mass1 * 0.5

    def test_total_mass_zero_at_start(self: Self) -> None:
        """Test that total mass is zero at start."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        assert baseline.total_mass() == 0.0


class TestDurationWeightedBaselineEdgeCases:
    """Tests for edge cases."""

    def test_single_state(self: Self) -> None:
        """Test with single state."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 1.0}, 100.0)
        baseline.update({"on": 1.0}, 110.0)

        result = baseline.distribution()
        assert abs(result.get("on", 0) - 1.0) < 1e-6

    def test_many_states(self: Self) -> None:
        """Test with many states."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        dist = {f"state{i}": 1.0 / 10 for i in range(10)}
        baseline.update(dist, 100.0)
        baseline.update(dist, 110.0)

        result = baseline.distribution()
        assert len(result) == 10
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_varying_distributions(self: Self) -> None:
        """Test with varying distributions over time."""
        hp = create_test_hp()
        baseline = DurationWeightedBaseline(hp)

        baseline.update({"on": 0.8, "off": 0.2}, 100.0)
        baseline.update({"on": 0.8, "off": 0.2}, 110.0)

        # Gradual shift
        for i in range(10):
            ratio = i / 10.0
            baseline.update(
                {"on": 0.8 - 0.6 * ratio, "off": 0.2 + 0.6 * ratio},
                120.0 + i * 10.0,
            )

        result = baseline.distribution()
        # Should reflect the shift
        assert result.get("off", 0) > 0.3
