"""Unit tests for TimeKey."""

from typing import Self

from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestTimeKeyBasic:
    """Tests for basic TimeKey functionality."""

    def test_global_is_empty(self: Self) -> None:
        """Test that GLOBAL has length 0 and empty tuple representation."""
        assert len(TimeKey.GLOBAL) == 0
        assert TimeKey.GLOBAL.to_tuple() == ()
        assert repr(TimeKey.GLOBAL) == "GLOBAL"

    def test_from_tuple_and_to_tuple_roundtrip(self: Self) -> None:
        """Test serialize-deserialize roundtrip preserves data."""
        data = (("weekday", 1), ("hour", 15))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data
        restored = TimeKey.from_tuple(key.to_tuple())
        assert restored == key
        assert hash(restored) == hash(key)

    def test_len_matches_number_of_features(self: Self) -> None:
        """Test that length equals number of features (not counting root)."""
        data = (("a", 1), ("b", 2), ("c", 3))
        key = TimeKey.from_tuple(data)
        assert len(key) == 3


class TestTimeKeyRepresentationAndOrder:
    """Tests for string representation and order sensitivity."""

    def test_repr_shows_root_to_leaf_order(self: Self) -> None:
        """Test repr formats features root-to-leaf order."""
        data = (("hour", 15), ("day", "Mon"))
        key = TimeKey.from_tuple(data)
        assert repr(key) == "hour = 15, day = Mon"

    def test_order_matters_for_equality_and_hash(self: Self) -> None:
        """Test that feature order affects equality and hash."""
        data1 = (("hour", 15), ("weekday", 1))
        data2 = (("weekday", 1), ("hour", 15))
        k1 = TimeKey.from_tuple(data1)
        k2 = TimeKey.from_tuple(data2)
        assert k1 != k2
        assert hash(k1) != hash(k2)


class TestTimeKeyHashability:
    """Tests for hashing and use in collections."""

    def test_usable_as_dict_key_and_set(self: Self) -> None:
        """Test TimeKey can be used as dict key and in sets."""
        a = TimeKey.from_tuple((("x", 1),))
        b = TimeKey.from_tuple((("x", 1),))
        c = TimeKey.from_tuple((("x", 2),))
        d = {a: "one", c: "two"}
        assert d[b] == "one"  # b equals a
        s = {a, b, c}
        assert len(s) == 2  # a and b are equal


class TestTimeKeyAddMethod:
    """Tests for the __add__ operator to build hierarchies."""

    def test_add_feature_to_global(self: Self) -> None:
        """Test adding a feature to GLOBAL creates single-feature key."""
        f = TemporalFeature("hour", 14)
        key = TimeKey.GLOBAL + f
        assert len(key) == 1
        assert key.to_tuple() == (("hour", 14),)
        assert repr(key) == "hour = 14"

    def test_add_multiple_features(self: Self) -> None:
        """Test adding multiple features builds hierarchy."""
        f1 = TemporalFeature("hour", 14)
        f2 = TemporalFeature("day", "Mon")
        key = TimeKey.GLOBAL + f1 + f2
        assert len(key) == 2
        assert key.to_tuple() == (("hour", 14), ("day", "Mon"))
        assert repr(key) == "hour = 14, day = Mon"

    def test_add_builds_correct_chain(self: Self) -> None:
        """Test that add creates proper parent-child chain."""
        f1 = TemporalFeature("a", 1)
        f2 = TemporalFeature("b", 2)
        f3 = TemporalFeature("c", 3)
        key = TimeKey.GLOBAL + f1 + f2 + f3
        assert len(key) == 3
        assert key.to_tuple() == (("a", 1), ("b", 2), ("c", 3))

    def test_add_preserves_order(self: Self) -> None:
        """Test that add preserves feature order."""
        f1 = TemporalFeature("x", 1)
        f2 = TemporalFeature("y", 2)
        key1 = TimeKey.GLOBAL + f1 + f2
        key2 = TimeKey.GLOBAL + f2 + f1
        assert key1 != key2
        assert key1.to_tuple() == (("x", 1), ("y", 2))
        assert key2.to_tuple() == (("y", 2), ("x", 1))

    def test_add_returns_new_timekey(self: Self) -> None:
        """Test that add returns new key without modifying original."""
        base = TimeKey.from_tuple((("hour", 14),))
        f = TemporalFeature("day", "Mon")
        new_key = base + f
        assert base.to_tuple() == (("hour", 14),)  # unchanged
        assert new_key.to_tuple() == (("hour", 14), ("day", "Mon"))
        assert base != new_key


class TestTimeKeyProperties:
    """Tests for is_root property."""

    def test_is_root_global(self: Self) -> None:
        """Test that GLOBAL.is_root is True."""
        assert TimeKey.GLOBAL.is_root is True

    def test_is_root_non_global(self: Self) -> None:
        """Test that non-GLOBAL keys have is_root False."""
        key = TimeKey.from_tuple((("hour", 14),))
        assert key.is_root is False


class TestTimeKeyAncestors:
    """Tests for ancestors() generator method."""

    def test_ancestors_global(self: Self) -> None:
        """Test that GLOBAL has no ancestors."""
        ancestors = list(TimeKey.GLOBAL.ancestors())
        assert ancestors == []

    def test_ancestors_single_feature(self: Self) -> None:
        """Test ancestors for single-feature key returns only GLOBAL."""
        key = TimeKey.from_tuple((("hour", 14),))
        ancestors = list(key.ancestors())
        assert len(ancestors) == 1
        assert ancestors[0] == TimeKey.GLOBAL

    def test_ancestors_multiple_features(self: Self) -> None:
        """Test ancestors yields parent nodes in order (not including self)."""
        key = TimeKey.from_tuple((("hour", 14), ("day", "Mon"), ("month", 3)))
        ancestors = list(key.ancestors())
        assert len(ancestors) == 3
        # First ancestor is parent (2 features)
        assert ancestors[0].to_tuple() == (("hour", 14), ("day", "Mon"))
        # Second ancestor is grandparent (1 feature)
        assert ancestors[1].to_tuple() == (("hour", 14),)
        # Third ancestor is GLOBAL
        assert ancestors[2] == TimeKey.GLOBAL

    def test_ancestors_order_leaf_to_root(self: Self) -> None:
        """Test ancestors yields parent chain from immediate parent to root."""
        key = TimeKey.from_tuple((("a", 1), ("b", 2), ("c", 3)))
        ancestors = list(key.ancestors())
        # First should be immediate parent (not self)
        assert ancestors[0].to_tuple() == (("a", 1), ("b", 2))
        # Last should be GLOBAL root
        assert ancestors[-1] == TimeKey.GLOBAL


class TestTimeKeyEquality:
    """Tests for equality comparison."""

    def test_equality_same_keys(self: Self) -> None:
        """Test equal keys have equal values."""
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14),))
        assert k1 == k2

    def test_equality_global(self: Self) -> None:
        """Test GLOBAL equals empty tuple key."""
        k1 = TimeKey.GLOBAL
        k2 = TimeKey.from_tuple(())
        assert k1 == k2

    def test_inequality_different_values(self: Self) -> None:
        """Test inequality with different values."""
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 15),))
        assert k1 != k2

    def test_inequality_different_lengths(self: Self) -> None:
        """Test inequality with different lengths."""
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        assert k1 != k2

    def test_inequality_with_non_timekey(self: Self) -> None:
        """Test inequality with non-TimeKey objects."""
        key = TimeKey.from_tuple((("hour", 14),))
        assert key != (("hour", 14),)
        assert key != "hour = 14"
        assert key != 14
        assert key != None

    def test_equality_reflexive(self: Self) -> None:
        """Test equality is reflexive (a == a)."""
        key = TimeKey.from_tuple((("hour", 14),))
        assert key == key

    def test_equality_transitive(self: Self) -> None:
        """Test equality is transitive (a==b, b==c implies a==c)."""
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14),))
        k3 = TimeKey.from_tuple((("hour", 14),))
        assert k1 == k2
        assert k2 == k3
        assert k1 == k3


class TestTimeKeyFromTupleValidation:
    """Tests for from_tuple() method."""

    def test_from_tuple_empty_returns_global(self: Self) -> None:
        """Test empty tuple creates GLOBAL node."""
        key = TimeKey.from_tuple(())
        assert key == TimeKey.GLOBAL
        assert len(key) == 0

    def test_from_tuple_preserves_types(self: Self) -> None:
        """Test from_tuple preserves various value types."""
        data = (("int", 42), ("str", "test"), ("float", 3.14), ("tuple", (1, 2)))
        key = TimeKey.from_tuple(data)
        result = key.to_tuple()
        assert result == data


class TestTimeKeyEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_deep_hierarchy(self: Self) -> None:
        """Test very deep hierarchy (20 levels)."""
        data = tuple((f"dim{i}", i) for i in range(20))
        key = TimeKey.from_tuple(data)
        assert len(key) == 20
        assert key.to_tuple() == data

    def test_unicode_names_and_values(self: Self) -> None:
        """Test with Unicode characters in names and values."""
        data = (("时间", "下午"), ("日期", "星期一"))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data
        assert "时间" in repr(key)

    def test_special_characters_in_names(self: Self) -> None:
        """Test with special characters in feature names."""
        data = (("key-with-dash", 1), ("key_with_underscore", 2), ("key.with.dot", 3))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data

    def test_numeric_string_values(self: Self) -> None:
        """Test with numeric strings (distinct from integers)."""
        data = (("hour", "14"), ("minute", "30"))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data

    def test_mixed_value_types_in_same_key(self: Self) -> None:
        """Test key with mixed value types."""
        data = (("a", 1), ("b", "two"), ("c", 3.0), ("d", (4, 5)), ("e", True))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data


class TestTimeKeyHashConsistency:
    """Tests for hash consistency and invertibility."""

    def test_hash_equals_implies_equal_hash(self: Self) -> None:
        """Test that equal keys have equal hashes."""
        k1 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        k2 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_hash_stable_across_calls(self: Self) -> None:
        """Test hash is stable across multiple calls."""
        key = TimeKey.from_tuple((("hour", 14),))
        h1 = hash(key)
        h2 = hash(key)
        assert h1 == h2

    def test_hash_different_for_different_order(self: Self) -> None:
        """Test different order produces different hash."""
        k1 = TimeKey.from_tuple((("a", 1), ("b", 2)))
        k2 = TimeKey.from_tuple((("b", 2), ("a", 1)))
        assert hash(k1) != hash(k2)


class TestTimeKeyHierarchy:
    """Tests for hierarchy() method."""

    def test_hierarchy_includes_self(self: Self) -> None:
        """Test that hierarchy includes self as first element."""
        key = TimeKey.from_tuple((("a", 1), ("b", 2)))
        hierarchy = list(key.hierarchy())
        assert hierarchy[0] == key

    def test_hierarchy_ends_at_global(self: Self) -> None:
        """Test that hierarchy chain ends with GLOBAL."""
        key = TimeKey.from_tuple((("a", 1), ("b", 2)))
        hierarchy = list(key.hierarchy())
        assert hierarchy[-1] == TimeKey.GLOBAL

    def test_hierarchy_count(self: Self) -> None:
        """Test hierarchy has correct number of elements."""
        key = TimeKey.from_tuple((("a", 1), ("b", 2), ("c", 3)))
        hierarchy = list(key.hierarchy())
        # Contains self + 2 parents + GLOBAL = 4 items
        assert len(hierarchy) == 4


class TestTimeKeyFromTemporalFeature:
    """Tests for from_temporal_feature() helper method."""

    def test_from_temporal_feature_creates_single_node(self: Self) -> None:
        """Test that from_temporal_feature creates single-feature key."""
        feature = TemporalFeature("hour", 14)
        key = TimeKey.from_temporal_feature(feature)
        assert len(key) == 1
        assert key.to_tuple() == (("hour", 14),)

    def test_from_temporal_feature_with_various_types(self: Self) -> None:
        """Test from_temporal_feature with various value types."""
        test_cases = [
            TemporalFeature("int_val", 42),
            TemporalFeature("str_val", "test"),
            TemporalFeature("bool_val", True),
            TemporalFeature("tuple_val", (1, 2, 3)),
        ]
        for feature in test_cases:
            key = TimeKey.from_temporal_feature(feature)
            assert len(key) == 1
            assert key.to_tuple() == (feature.to_tuple(),)
