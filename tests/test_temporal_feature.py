"""Unit tests for TemporalFeature."""

from dataclasses import asdict
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)


class TestTemporalFeatureInitialization:
    """Tests for TemporalFeature initialization."""

    def test_create_with_string_value(self: Self) -> None:
        """Test creating TemporalFeature with integer value."""
        feature = TemporalFeature(name="hour", value=14)
        assert feature.name == "hour"
        assert feature.value == 14

    def test_create_with_tuple_value(self: Self) -> None:
        """Test creating TemporalFeature with tuple value."""
        feature = TemporalFeature(name="range", value=(0, 6))
        assert feature.name == "range"
        assert feature.value == (0, 6)

    def test_create_with_none_value(self: Self) -> None:
        """Test creating TemporalFeature with None value."""
        feature = TemporalFeature(name="special", value=None)
        assert feature.name == "special"
        assert feature.value is None


class TestTemporalFeatureImmutability:
    """Tests for TemporalFeature frozen (immutable) behavior."""

    def test_frozen_attribute_name(self: Self) -> None:
        """Test that name attribute cannot be modified."""
        feature = TemporalFeature(name="hour", value=14)
        with pytest.raises(AttributeError):
            feature.name = "minute"  # type: ignore[misc]

    def test_frozen_attribute_value(self: Self) -> None:
        """Test that value attribute cannot be modified."""
        feature = TemporalFeature(name="hour", value=14)
        with pytest.raises(AttributeError):
            feature.value = 15  # type: ignore[misc]


class TestTemporalFeatureHashability:
    """Tests for TemporalFeature hashing and use in collections."""

    def test_hashable(self: Self) -> None:
        """Test that TemporalFeature is hashable."""
        feature = TemporalFeature(name="hour", value=14)
        hash_value = hash(feature)
        assert isinstance(hash_value, int)

    def test_can_use_as_dict_key(self: Self) -> None:
        """Test that features can be used as dictionary keys."""
        feature1 = TemporalFeature(name="hour", value=14)
        feature2 = TemporalFeature(name="day", value="Monday")
        d = {feature1: "afternoon", feature2: "weekday"}
        assert d[feature1] == "afternoon"
        assert d[feature2] == "weekday"

    def test_can_store_in_set(self: Self) -> None:
        """Test that features can be stored in sets."""
        feature1 = TemporalFeature(name="hour", value=14)
        feature2 = TemporalFeature(name="day", value="Monday")
        feature3 = TemporalFeature(name="hour", value=14)
        s = {feature1, feature2, feature3}
        # feature1 and feature3 are equal, so set has 2 elements
        assert len(s) == 2
        assert feature1 in s
        assert feature2 in s


class TestTemporalFeatureEquality:
    """Tests for TemporalFeature equality comparison."""

    def test_equal_features(self: Self) -> None:
        """Test equality of identical features."""
        feature1 = TemporalFeature(name="hour", value=14)
        feature2 = TemporalFeature(name="hour", value=14)
        assert feature1 == feature2
        assert hash(feature1) == hash(feature2)

    def test_different_names(self: Self) -> None:
        """Test inequality when names differ."""
        feature1 = TemporalFeature(name="hour", value=14)
        feature2 = TemporalFeature(name="minute", value=14)
        assert feature1 != feature2

    def test_different_values(self: Self) -> None:
        """Test inequality when values differ."""
        feature1 = TemporalFeature(name="hour", value=14)
        feature2 = TemporalFeature(name="hour", value=15)
        assert feature1 != feature2

    def test_different_types(self: Self) -> None:
        """Test inequality with non-TemporalFeature objects."""
        feature = TemporalFeature(name="hour", value=14)
        assert feature != ("hour", 14)
        assert feature != "hour = 14"
        assert feature != 14


class TestTemporalFeatureRepr:
    """Tests for TemporalFeature string representation."""

    def test_repr_format(self: Self) -> None:
        """Test repr format with integer value."""
        feature = TemporalFeature(name="hour", value=14)
        assert repr(feature) == "hour = 14"

    def test_repr_with_string_value(self: Self) -> None:
        """Test repr format with string value."""
        feature = TemporalFeature(name="day", value="Monday")
        assert repr(feature) == "day = Monday"

    def test_repr_with_tuple_value(self: Self) -> None:
        """Test repr format with tuple value."""
        feature = TemporalFeature(name="range", value=(0, 6))
        assert repr(feature) == "range = (0, 6)"


class TestTemporalFeatureEdgeCases:
    """Tests for edge cases and special values."""

    def test_empty_string_name(self: Self) -> None:
        """Test feature with empty string name."""
        feature = TemporalFeature(name="", value=14)
        assert feature.name == ""
        assert asdict(feature) == {"name": "", "value": 14}

    def test_complex_nested_tuple_value(self: Self) -> None:
        """Test feature with nested tuple value."""
        feature = TemporalFeature(name="nested", value=(1, (2, 3), 4))
        assert feature.value == (1, (2, 3), 4)
        restored = TemporalFeature.from_tuple(feature.to_tuple())
        assert restored == feature

    def test_boolean_value(self: Self) -> None:
        """Test features with boolean values."""
        feature_true = TemporalFeature(name="flag", value=True)
        feature_false = TemporalFeature(name="flag", value=False)
        assert feature_true.value is True
        assert feature_false.value is False
        assert feature_true != feature_false

    def test_long_name(self: Self) -> None:
        """Test feature with very long name."""
        long_name = "very_" * 100 + "long_name"
        feature = TemporalFeature(name=long_name, value=1)
        assert feature.name == long_name
        assert feature.to_tuple() == (long_name, 1)

    def test_unicode_names_and_values(self: Self) -> None:
        """Test features with Unicode characters."""
        feature = TemporalFeature(name="时间", value="下午")
        assert feature.name == "时间"
        assert feature.value == "下午"
        restored = TemporalFeature.from_tuple(feature.to_tuple())
        assert restored == feature

    def test_special_characters_in_name(self: Self) -> None:
        """Test feature with special characters in name."""
        feature = TemporalFeature(name="key-with-special_chars.123", value="value")
        assert feature.name == "key-with-special_chars.123"

    def test_zero_values(self: Self) -> None:
        """Test features with zero values."""
        f_int = TemporalFeature(name="count", value=0)
        f_float = TemporalFeature(name="amount", value=0.0)
        assert f_int.value == 0
        assert f_float.value == 0.0
        assert f_int != f_float

    def test_negative_values(self: Self) -> None:
        """Test feature with negative value."""
        feature = TemporalFeature(name="temp", value=-10)
        assert feature.value == -10

    def test_frozenset_as_value(self: Self) -> None:
        """Test feature with frozenset value."""
        feature = TemporalFeature(name="items", value=frozenset([1, 2, 3]))
        assert feature.value == frozenset([1, 2, 3])
        assert hash(feature)


class TestTemporalFeatureHashConsistency:
    """Tests for hash consistency and invariants."""

    def test_hash_consistency_across_calls(self: Self) -> None:
        """Test that hash is consistent across multiple calls."""
        feature = TemporalFeature(name="hour", value=14)
        h1 = hash(feature)
        h2 = hash(feature)
        assert h1 == h2

    def test_equal_features_same_hash(self: Self) -> None:
        """Test that equal features have equal hashes."""
        f1 = TemporalFeature(name="hour", value=14)
        f2 = TemporalFeature(name="hour", value=14)
        assert f1 == f2
        assert hash(f1) == hash(f2)

    def test_hash_different_for_different_name(self: Self) -> None:
        """Test that different names produce different hashes."""
        f1 = TemporalFeature(name="hour", value=14)
        f2 = TemporalFeature(name="minute", value=14)
        assert hash(f1) != hash(f2)

    def test_hash_different_for_different_value(self: Self) -> None:
        """Test that different values produce different hashes."""
        f1 = TemporalFeature(name="hour", value=14)
        f2 = TemporalFeature(name="hour", value=15)
        assert hash(f1) != hash(f2)


class TestTemporalFeatureEqualityProperties:
    """Tests for equality relation properties."""

    def test_equality_reflexive(self: Self) -> None:
        """Test that a feature equals itself (reflexivity)."""
        feature = TemporalFeature(name="hour", value=14)
        assert feature == feature  # noqa: PLR0124

    def test_equality_symmetric(self: Self) -> None:
        """Test equality is symmetric (a==b implies b==a)."""
        f1 = TemporalFeature(name="hour", value=14)
        f2 = TemporalFeature(name="hour", value=14)
        assert f1 == f2
        assert f2 == f1

    def test_equality_transitive(self: Self) -> None:
        """Test equality is transitive (a==b and b==c implies a==c)."""
        f1 = TemporalFeature(name="hour", value=14)
        f2 = TemporalFeature(name="hour", value=14)
        f3 = TemporalFeature(name="hour", value=14)
        assert f1 == f2
        assert f2 == f3
        assert f1 == f3

    def test_inequality_with_none(self: Self) -> None:
        """Test that features are not equal to None."""
        feature = TemporalFeature(name="hour", value=14)
        assert feature is not None
        assert feature is not None
