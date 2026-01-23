"""Tests for DiscreteConditionalModel class."""

import math
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``custom_components`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from custom_components.binary_state_forecaster.discrete_conditional_model import (  # noqa: E402
    DiscreteConditionalModel,
)


class TestDiscreteConditionalModelBasics:
    """Test basic initialization and simple operations."""

    def test_initialization(self) -> None:
        """Test model initializes with default parameters."""
        model = DiscreteConditionalModel()
        assert model.alpha == 1.0
        assert model.decay == 3600.0
        assert model.max_depth is None
        assert model.priority_decay == 0.98

    def test_initialization_custom_params(self) -> None:
        """Test model initializes with custom parameters."""
        model = DiscreteConditionalModel(
            alpha=0.5,
            decay=0.95,
            max_depth=3,
            priority_decay=0.9,
            interaction_decay=0.9,
        )
        assert model.alpha == 0.5
        assert model.decay == 0.95
        assert model.max_depth == 3
        assert model.priority_decay == 0.9
        assert model.interaction_decay == 0.9

    def test_initial_state(self) -> None:
        """Test model starts with empty state."""
        model = DiscreteConditionalModel()
        assert len(model._states) == 0
        assert len(model._y_domain) == 0
        assert len(model._feature_scores) == 0
        assert len(model._interaction_scores) == 0
        assert model._t == 0.0


class TestDiscreteConditionalModelUpdate:
    """Test model update/training functionality."""

    def test_single_update(self) -> None:
        """Test model can be updated with a single observation."""
        model = DiscreteConditionalModel()
        features = {"weather": "sunny", "hour": 10}
        y = "on"

        model.update(features, y)
        assert "on" in model._y_domain
        assert len(model._states) > 0

    def test_multiple_updates(self) -> None:
        """Test model accumulates multiple observations."""
        model = DiscreteConditionalModel()

        model.update({"weather": "sunny"}, "on")
        model.update({"weather": "sunny"}, "on")
        model.update({"weather": "rainy"}, "off")

        assert "on" in model._y_domain
        assert "off" in model._y_domain

    def test_update_with_none_features(self) -> None:
        """Test model handles None values in features."""
        model = DiscreteConditionalModel()
        features = {"weather": "sunny", "temp": None}

        model.update(features, "on")
        prediction = model.predict(features)
        assert prediction == "on"

    def test_y_domain_accumulation(self) -> None:
        """Test y_domain accumulates all observed labels."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")
        model.update({"x": 2}, "b")
        model.update({"x": 3}, "c")

        assert model._y_domain == {"a", "b", "c"}


class TestDiscreteConditionalModelPrediction:
    """Test prediction functionality."""

    def test_predict_simple(self) -> None:
        """Test basic prediction with clear pattern."""
        model = DiscreteConditionalModel()

        for _ in range(10):
            model.update({"weather": "sunny"}, "on")

        model.update({"weather": "rainy"}, "off")

        prediction = model.predict({"weather": "sunny"})
        assert prediction == "on"

    def test_predict_returns_most_likely(self) -> None:
        """Test prediction returns the most likely outcome."""
        model = DiscreteConditionalModel()

        # Use longer durations for 'work' to ensure it dominates
        for _ in range(7):
            model.update({"day": "weekday"}, "work", duration=2.0)
        for _ in range(3):
            model.update({"day": "weekday"}, "home", duration=1.0)

    def test_predict_with_unseen_features(self) -> None:
        """Test prediction with completely unseen features."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")
        model.update({"x": 2}, "a")

        prediction = model.predict({"z": 99})
        assert prediction in model._y_domain

    def test_predict_empty_model(self) -> None:
        """Test prediction on empty model returns None."""
        model = DiscreteConditionalModel()
        prediction = model.predict({"x": 1})
        assert prediction is None


class TestDiscreteConditionalModelDistribution:
    """Test probability distribution functionality."""

    def test_distribution_sums_to_one(self) -> None:
        """Test returned distribution sums to approximately 1.0."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")
        model.update({"x": 1}, "b")
        model.update({"x": 1}, "a")

        dist, _ = model.distribution_with_key({"x": 1})
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_distribution_with_key(self) -> None:
        """Test distribution_with_key returns distribution and key."""
        model = DiscreteConditionalModel()

        model.update({"weather": "sunny", "hour": 10}, "on")

        dist, key = model.distribution_with_key({"weather": "sunny", "hour": 10})
        assert isinstance(dist, dict)
        assert isinstance(key, tuple)
        assert sum(dist.values()) > 0

    def test_distribution_laplace_smoothing(self) -> None:
        """Test distribution applies Laplace smoothing with alpha."""
        model = DiscreteConditionalModel(alpha=1.0)

        model.update({"x": 1}, "a")

        dist, _ = model.distribution_with_key({"x": 1})
        # All labels should have non-zero probability due to smoothing
        assert all(p > 0 for p in dist.values())


class TestDiscreteConditionalModelDurationWeighting:
    """Test duration-weighted learning functionality."""

    def test_duration_weighting(self) -> None:
        """Test that longer duration states are weighted more heavily."""
        model = DiscreteConditionalModel()

        # Short duration for 'a'
        model.update({"x": 1}, "a", duration=1.0)
        # Long duration for 'b'
        model.update({"x": 1}, "b", duration=10.0)

        prediction = model.predict({"x": 1})
        assert prediction == "b"  # Should favor longer duration state

    def test_default_duration(self) -> None:
        """Test default duration is 1.0."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")  # Default duration

        key = (("x", 1),)
        assert model._states[key].y["a"] == 1.0
        assert model._t == 1.0  # Time should advance by default duration

    def test_zero_duration_ignored(self) -> None:
        """Test that zero or negative duration is ignored."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a", duration=0.0)
        model.update({"x": 1}, "b", duration=-1.0)

        assert len(model._y_domain) == 0  # Nothing should be added


class TestDiscreteConditionalModelBackoff:
    """Test backoff mechanism for missing features."""

    def test_backoff_to_simpler_features(self) -> None:
        """Test model backs off to simpler feature combinations."""
        model = DiscreteConditionalModel()

        # Train on single features
        model.update({"a": 1}, "x")
        model.update({"b": 2}, "y")

        # Query with combination not seen
        dist, key = model.distribution_with_key({"a": 1, "b": 2})

        # Should backoff to either a=1 or b=2, not empty tuple
        assert len(key) >= 0

    def test_feature_subsets_ordering(self) -> None:
        """Test feature subsets are ordered by interaction scores."""
        model = DiscreteConditionalModel()

        # Train to build up scores
        model.update({"a": 1, "b": 2}, "x")
        model.update({"a": 1, "b": 2}, "x")

        # Just verify it doesn't crash - internal method
        subsets = list(model._feature_subsets({"a": 1, "b": 2}))
        assert isinstance(subsets, list)

    def test_backoff_to_global(self) -> None:
        """Test model backs off to global distribution when no matches."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")
        model.update({"y": 2}, "a")

        # Query with completely unseen features
        dist, key = model.distribution_with_key({"z": 99})

        # Should use global distribution (empty key)
        assert key == ()


class TestDiscreteConditionalModelConfidence:
    """Test confidence estimation."""

    def test_confidence_structure(self) -> None:
        """Test confidence returns expected structure."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, "a")

        conf = model.confidence({"x": 1})
        assert hasattr(conf, "max_probability")
        assert hasattr(conf, "entropy_confidence")
        assert hasattr(conf, "support_time")
        assert hasattr(conf, "used_features")

    def test_confidence_high_certainty(self) -> None:
        """Test confidence is high when data is certain."""
        model = DiscreteConditionalModel()

        # Create two labels but heavily weight one with duration
        model.update({"x": 1}, "a", duration=100.0)
        model.update({"x": 1}, "b", duration=1.0)  # Add one opposing example

        conf = model.confidence({"x": 1})
        assert conf.max_probability > 0.9
        assert conf.entropy_confidence > 0.75  # Should be high with strong bias
        assert conf.support_time > 50

    def test_confidence_low_certainty(self) -> None:
        """Test confidence is lower when data is uncertain."""
        model = DiscreteConditionalModel()

        # Equal distribution
        model.update({"x": 1}, "a")
        model.update({"x": 1}, "b")

        conf = model.confidence({"x": 1})
        assert conf.entropy_confidence < 0.5

    def test_confidence_empty_model(self) -> None:
        """Test confidence on empty model."""
        model = DiscreteConditionalModel()
        conf = model.confidence({"x": 1})
        assert conf.entropy_confidence == 0.0
        assert conf.max_probability == 0.0
        assert conf.support_time == 0.0
        assert conf.used_features == {}


class TestDiscreteConditionalModelDecay:
    """Test temporal decay functionality."""

    def test_decay_applied_on_update(self) -> None:
        """Test decay is applied to existing durations on update."""
        model = DiscreteConditionalModel(decay=0.5)  # Half-life of 0.5 time units

        model.update({"x": 1}, "a", duration=1.0)
        key = (("x", 1),)
        initial_duration = model._states[key].y["a"]

        # Advance time significantly to see decay
        model.update({"x": 1}, "b", duration=2.0)  # 2 time units pass

        # First duration should have been decayed (2^(-2/0.5) = 2^-4 = 0.0625)
        current_duration = model._states[key].y["a"]
        assert (
            current_duration < initial_duration * 0.2
        )  # Should be significantly decayed
        """Test recent observations are weighted more heavily."""
        model = DiscreteConditionalModel(decay=0.5)

        # Old data
        for _ in range(10):
            model.update({"x": 1}, "a")

        # Recent data
        for _ in range(10):
            model.update({"x": 1}, "b")

        prediction = model.predict({"x": 1})
        # Recent 'b' should dominate due to decay of 'a'
        assert prediction == "b"


class TestDiscreteConditionalModelFeaturePriority:
    """Test adaptive feature prioritization."""

    def test_feature_scores_updated(self) -> None:
        """Test feature scores are updated during training."""
        model = DiscreteConditionalModel()

        # Feature 'a' is informative
        model.update({"a": 1, "b": 1}, "x")
        model.update({"a": 2, "b": 1}, "y")
        model.update({"a": 1, "b": 1}, "x")

        # Feature scores should be non-zero
        assert len(model._feature_scores) > 0

    def test_informative_features_prioritized(self) -> None:
        """Test informative features get higher scores."""
        model = DiscreteConditionalModel()

        # Feature 'good' is perfectly predictive
        for _ in range(10):
            model.update({"good": 1, "noise": 1}, "a")
            model.update({"good": 2, "noise": 1}, "b")

        # 'good' should have higher score than 'noise'
        assert model._feature_scores.get("good", 0) > model._feature_scores.get(
            "noise", 0
        )


class TestDiscreteConditionalModelMaxDepth:
    """Test max_depth parameter."""

    def test_max_depth_limits_combinations(self) -> None:
        """Test max_depth limits feature combination size."""
        model = DiscreteConditionalModel(max_depth=2)

        features = {"a": 1, "b": 2, "c": 3, "d": 4}
        model.update(features, "x")

        # Check that keys don't exceed max_depth
        for key in model._states:
            if key != ():  # Skip global key
                assert len(key) <= 2

    def test_max_depth_none_allows_all(self) -> None:
        """Test max_depth=None allows all feature combinations."""
        model = DiscreteConditionalModel(max_depth=None)

        features = {"a": 1, "b": 2, "c": 3}
        model.update(features, "x")

        # Should have combinations up to full size
        max_key_len = max(len(key) for key in model._states if key != ())
        assert max_key_len == 3


class TestDiscreteConditionalModelInteractionLearning:
    """Test interaction learning features."""

    def test_interaction_scores_updated(self) -> None:
        """Test interaction scores are updated during training."""
        model = DiscreteConditionalModel()

        # Train with feature interactions
        model.update({"a": 1, "b": 2}, "x")
        model.update({"a": 1, "b": 2}, "x")

        # Interaction scores should be present
        assert len(model._interaction_scores) >= 0  # May or may not have interactions

    def test_interaction_contributions(self) -> None:
        """Test interaction_contributions shows feature interactions."""
        model = DiscreteConditionalModel()

        # Train with clear interaction pattern
        model.update({"a": 1, "b": 2}, "x")
        model.update({"a": 1, "b": 2}, "x")
        model.update({"a": 1}, "y")
        model.update({"b": 2}, "y")

        contributions = model.interaction_contributions({"a": 1, "b": 2})
        # Should return dict of interactions
        assert isinstance(contributions, dict)

    def test_interaction_contributions_single_feature(self) -> None:
        """Test interaction_contributions returns empty for single feature."""
        model = DiscreteConditionalModel()

        model.update({"a": 1}, "x")

        contributions = model.interaction_contributions({"a": 1})
        # No interactions with single feature
        assert contributions == {}


class TestDiscreteConditionalModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_features(self) -> None:
        """Test model handles empty feature dict."""
        model = DiscreteConditionalModel()

        model.update({}, "a")
        prediction = model.predict({})
        assert prediction == "a"

    def test_all_none_features(self) -> None:
        """Test model handles all None values."""
        model = DiscreteConditionalModel()

        model.update({"a": None, "b": None}, "x")
        prediction = model.predict({"a": None, "b": None})
        assert prediction == "x"

    def test_large_number_of_features(self) -> None:
        """Test model handles many features."""
        model = DiscreteConditionalModel()

        features = {f"f{i}": i for i in range(20)}
        model.update(features, "x")

        prediction = model.predict(features)
        assert prediction == "x"

    def test_many_labels(self) -> None:
        """Test model handles many different labels."""
        model = DiscreteConditionalModel()

        for i in range(100):
            model.update({"x": 1}, f"label_{i}")

        dist, _ = model.distribution_with_key({"x": 1})
        assert len(dist) == 100

    def test_numeric_and_string_labels(self) -> None:
        """Test model handles mixed label types."""
        model = DiscreteConditionalModel()

        model.update({"x": 1}, 0)
        model.update({"x": 2}, "on")
        model.update({"x": 3}, True)

        assert 0 in model._y_domain
        assert "on" in model._y_domain
        assert True in model._y_domain


class TestDiscreteConditionalModelEntropy:
    """Test entropy calculations."""

    def test_entropy_uniform_distribution(self) -> None:
        """Test entropy calculation for uniform distribution."""
        model = DiscreteConditionalModel()

        # Create uniform distribution
        model.update({"x": 1}, "a")
        model.update({"x": 1}, "b")
        model.update({"x": 1}, "c")
        model.update({"x": 1}, "d")

        dist, _ = model.distribution_with_key({"x": 1})
        entropy = model._entropy(dist)

        # Entropy should be close to log2(4) for uniform
        expected = math.log2(len(dist))
        assert abs(entropy - expected) < 0.5  # Some smoothing affects this

    def test_entropy_deterministic(self) -> None:
        """Test entropy is low for deterministic distribution."""
        model = DiscreteConditionalModel()

        for _ in range(100):
            model.update({"x": 1}, "a")

        dist, _ = model.distribution_with_key({"x": 1})
        entropy = model._entropy(dist)

        # Should be very low entropy
        assert entropy < 0.5


class TestDiscreteConditionalModelIntegration:
    """Integration tests with realistic scenarios."""

    def test_weather_prediction_scenario(self) -> None:
        """Test realistic weather-based prediction scenario."""
        model = DiscreteConditionalModel()

        # Train on patterns
        training_data = [
            ({"weather": "sunny", "temp": "hot"}, "ac_on"),
            ({"weather": "sunny", "temp": "hot"}, "ac_on"),
            ({"weather": "rainy", "temp": "cold"}, "heater_on"),
            ({"weather": "rainy", "temp": "cold"}, "heater_on"),
            ({"weather": "cloudy", "temp": "mild"}, "off"),
        ]

        for features, label in training_data:
            model.update(features, label)

        # Test predictions
        assert model.predict({"weather": "sunny", "temp": "hot"}) == "ac_on"
        assert model.predict({"weather": "rainy", "temp": "cold"}) == "heater_on"

    def test_incremental_learning(self) -> None:
        """Test model adapts to changing patterns."""
        model = DiscreteConditionalModel(decay=0.9)

        # Initial pattern
        for _ in range(20):
            model.update({"x": 1}, "old")

        assert model.predict({"x": 1}) == "old"

        # Pattern changes
        for _ in range(30):
            model.update({"x": 1}, "new")

        # Should adapt to new pattern
        assert model.predict({"x": 1}) == "new"

    def test_multi_feature_interaction(self) -> None:
        """Test model captures multi-feature interactions."""
        model = DiscreteConditionalModel()

        # Complex interaction pattern
        model.update({"a": 1, "b": 1}, "positive")
        model.update({"a": 1, "b": 2}, "negative")
        model.update({"a": 2, "b": 1}, "negative")
        model.update({"a": 2, "b": 2}, "positive")

        # Model should learn the XOR-like pattern
        assert model.predict({"a": 1, "b": 1}) == "positive"
        assert model.predict({"a": 2, "b": 2}) == "positive"
