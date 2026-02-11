"""
Unit tests for OnlineErrorTrackerHyperParameters.

Comprehensive tests for the OnlineErrorTrackerHyperParameters class, covering
initialization and error half-life calculations.
"""

from typing import Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_hyper_parameters import (  # noqa: E501
    OnlineErrorTrackerHyperParameters,
)


class TestOnlineErrorTrackerHyperParametersInitialization:
    """Tests for OnlineErrorTrackerHyperParameters initialization."""

    def test_create_with_default_factor(self: Self) -> None:
        """Test creating with factor 1.0."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=1.0,
        )
        assert hp.error_half_life == 50.0

    def test_create_with_small_factor(self: Self) -> None:
        """Test creating with factor < 1.0 for faster adaptation."""
        base_hp = HyperParameters(
            half_life=100.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=0.5,
        )
        assert hp.error_half_life == 50.0

    def test_create_with_large_factor(self: Self) -> None:
        """Test creating with factor > 1.0 for slower adaptation."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=2.0,
        )
        assert hp.error_half_life == 100.0


class TestOnlineErrorTrackerHyperParametersErrorHalfLife:
    """Tests for error_half_life property."""

    def test_error_half_life_calculation(self: Self) -> None:
        """Test that error half-life is correctly calculated."""
        base_hp = HyperParameters(
            half_life=60.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=0.75,
        )
        assert abs(hp.error_half_life - 45.0) < 1e-9

    def test_error_half_life_multiple_calls(self: Self) -> None:
        """Test that error_half_life returns same value on multiple calls."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=1.5,
        )
        result1 = hp.error_half_life
        result2 = hp.error_half_life
        assert result1 == result2
        assert result1 == 75.0

    def test_error_half_life_with_zero_factor(self: Self) -> None:
        """Test error half-life with zero factor."""
        base_hp = HyperParameters(
            half_life=50.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=0.0,
        )
        assert hp.error_half_life == 0.0

    def test_error_half_life_with_very_large_factor(self: Self) -> None:
        """Test error half-life with very large factor."""
        base_hp = HyperParameters(
            half_life=10.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=100.0,
        )
        assert hp.error_half_life == 1000.0


class TestOnlineErrorTrackerHyperParametersEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_fractional_error_half_life(self: Self) -> None:
        """Test with fractional result for error half-life."""
        base_hp = HyperParameters(
            half_life=7.0,
            min_prune_interval=10.0,
            prune_enabled=True,
            persistence_strength=0.95,
        )
        hp = OnlineErrorTrackerHyperParameters(
            hyper_parameters=base_hp,
            error_half_life_factor=0.3,
        )
        assert abs(hp.error_half_life - 2.1) < 1e-9

    def test_different_base_half_lives(self: Self) -> None:
        """Test with different base half-life values."""
        factors = [0.5, 1.0, 2.0]
        base_half_lives = [10.0, 50.0, 100.0, 1000.0]

        for base_hl in base_half_lives:
            for factor in factors:
                base_hp = HyperParameters(
                    half_life=base_hl,
                    min_prune_interval=10.0,
                    prune_enabled=True,
                    persistence_strength=0.95,
                )
                hp = OnlineErrorTrackerHyperParameters(
                    hyper_parameters=base_hp,
                    error_half_life_factor=factor,
                )
                expected = base_hl * factor
                assert abs(hp.error_half_life - expected) < 1e-9
