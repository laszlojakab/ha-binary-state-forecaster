"""
Comprehensive tests for the TimeKey class.

Tests cover initialization, hashing, equality, string representation,
parent navigation, hierarchical iteration, and edge cases.
"""

import json
from typing import Self

import pytest

from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)


class TestTimeKeyInitialization:
    """Tests for TimeKey initialization and basic properties."""

    def test_empty_initialization(self: Self) -> None:
        """Test creating an empty TimeKey."""
        key = TimeKey()
        assert len(key) == 0
        assert key.items == ()

    def test_initialization_with_none(self: Self) -> None:
        """Test creating a TimeKey with explicit None."""
        key = TimeKey(None)
        assert len(key) == 0
        assert key.items == ()

    def test_single_dimension_initialization(self: Self) -> None:
        """Test creating a TimeKey with a single dimension."""
        key = TimeKey((("hour", 15),))
        assert len(key) == 1
        assert key.items == (("hour", 15),)

    def test_multiple_dimensions_initialization(self: Self) -> None:
        """Test creating a TimeKey with multiple dimensions."""
        key = TimeKey((("weekday", 1), ("hour", 15)))
        assert len(key) == 2
        assert key.items == (("weekday", 1), ("hour", 15))

    def test_complex_initialization(self: Self) -> None:
        """Test creating a TimeKey with many dimensions."""
        items = (("month", 3), ("weekday", 2), ("hour", 14), ("minute_bucket", 1))
        key = TimeKey(items)
        assert len(key) == 4
        assert key.items == items


class TestTimeKeyGlobal:
    """Tests for the TimeKey.GLOBAL constant."""

    def test_global_exists(self: Self) -> None:
        """Test that TimeKey.GLOBAL is initialized."""
        assert hasattr(TimeKey, "GLOBAL")
        assert isinstance(TimeKey.GLOBAL, TimeKey)

    def test_global_is_empty(self: Self) -> None:
        """Test that TimeKey.GLOBAL is an empty key."""
        assert len(TimeKey.GLOBAL) == 0
        assert TimeKey.GLOBAL.items == ()

    def test_global_equals_empty_key(self: Self) -> None:
        """Test that TimeKey.GLOBAL equals a newly created empty key."""
        empty_key = TimeKey()
        assert empty_key == TimeKey.GLOBAL

    def test_global_repr(self: Self) -> None:
        """Test that TimeKey.GLOBAL has the correct string representation."""
        assert repr(TimeKey.GLOBAL) == "TimeKey.GLOBAL"


class TestTimeKeyHashing:
    """Tests for TimeKey hashing behavior."""

    def test_hash_consistency(self: Self) -> None:
        """Test that the same key always produces the same hash."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 15),))
        assert hash(key1) == hash(key2)

    def test_hash_different_for_different_keys(self: Self) -> None:
        """Test that different keys generally produce different hashes."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 16),))
        # Note: Hash collisions are possible but unlikely for different data
        assert hash(key1) != hash(key2)

    def test_hash_different_for_different_dimensions(self: Self) -> None:
        """Test that keys with different dimension names have different hashes."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("minute", 15),))
        assert hash(key1) != hash(key2)

    def test_hash_order_matters(self: Self) -> None:
        """Test that dimension order affects the hash."""
        key1 = TimeKey((("hour", 15), ("weekday", 1)))
        key2 = TimeKey((("weekday", 1), ("hour", 15)))
        assert hash(key1) != hash(key2)

    def test_hash_usable_in_dict(self: Self) -> None:
        """Test that TimeKey can be used as a dictionary key."""
        key = TimeKey((("hour", 15),))
        d = {key: "value"}
        assert d[key] == "value"

    def test_hash_usable_in_set(self: Self) -> None:
        """Test that TimeKey can be used in a set."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 15),))
        key3 = TimeKey((("hour", 16),))
        s = {key1, key2, key3}
        assert len(s) == 2  # key1 and key2 are equal


class TestTimeKeyEquality:
    """Tests for TimeKey equality comparison."""

    def test_equality_same_single_dimension(self: Self) -> None:
        """Test equality for keys with the same single dimension."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 15),))
        assert key1 == key2

    def test_equality_same_multiple_dimensions(self: Self) -> None:
        """Test equality for keys with the same multiple dimensions."""
        key1 = TimeKey((("weekday", 1), ("hour", 15)))
        key2 = TimeKey((("weekday", 1), ("hour", 15)))
        assert key1 == key2

    def test_inequality_different_values(self: Self) -> None:
        """Test inequality for keys with different values."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 16),))
        assert key1 != key2

    def test_inequality_different_dimensions(self: Self) -> None:
        """Test inequality for keys with different dimension names."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("minute", 15),))
        assert key1 != key2

    def test_inequality_different_lengths(self: Self) -> None:
        """Test inequality for keys with different numbers of dimensions."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 15), ("weekday", 1)))
        assert key1 != key2

    def test_inequality_different_order(self: Self) -> None:
        """Test that dimension order matters for equality."""
        key1 = TimeKey((("hour", 15), ("weekday", 1)))
        key2 = TimeKey((("weekday", 1), ("hour", 15)))
        assert key1 != key2

    def test_equality_with_non_timekey(self: Self) -> None:
        """Test that TimeKey is not equal to non-TimeKey objects."""
        key = TimeKey((("hour", 15),))
        assert key != "not a timekey"
        assert key != 15
        assert key != (("hour", 15),)
        assert key is not None

    def test_empty_keys_equal(self: Self) -> None:
        """Test that empty keys are equal."""
        key1 = TimeKey()
        key2 = TimeKey()
        assert key1 == key2


class TestTimeKeyStringRepresentation:
    """Tests for TimeKey string representation."""

    def test_repr_empty_key(self: Self) -> None:
        """Test repr for an empty key."""
        key = TimeKey()
        assert repr(key) == "TimeKey.GLOBAL"

    def test_repr_single_dimension(self: Self) -> None:
        """Test repr for a single dimension key."""
        key = TimeKey((("hour", 15),))
        assert repr(key) == "TimeKey((('hour', 15),))"

    def test_repr_multiple_dimensions(self: Self) -> None:
        """Test repr for a multiple dimension key."""
        key = TimeKey((("weekday", 1), ("hour", 15)))
        assert repr(key) == "TimeKey((('weekday', 1), ('hour', 15)))"

    def test_repr_with_string_values(self: Self) -> None:
        """Test repr with string dimension values."""
        key = TimeKey((("period", "morning"),))
        assert repr(key) == "TimeKey((('period', 'morning'),))"


class TestTimeKeyLength:
    """Tests for TimeKey length."""

    def test_len_empty(self: Self) -> None:
        """Test length of empty key."""
        assert len(TimeKey()) == 0

    def test_len_single(self: Self) -> None:
        """Test length of single dimension key."""
        assert len(TimeKey((("hour", 15),))) == 1

    def test_len_multiple(self: Self) -> None:
        """Test length of multiple dimension key."""
        assert len(TimeKey((("weekday", 1), ("hour", 15)))) == 2

    def test_len_complex(self: Self) -> None:
        """Test length of complex key."""
        items = (("month", 3), ("weekday", 2), ("hour", 14), ("minute", 30))
        assert len(TimeKey(items)) == 4


class TestTimeKeyParent:
    """Tests for the parent() method."""

    def test_parent_of_single_dimension(self: Self) -> None:
        """Test parent of a single dimension key returns GLOBAL."""
        key = TimeKey((("hour", 15),))
        parent = key.parent()
        assert parent == TimeKey.GLOBAL

    def test_parent_of_multiple_dimensions(self: Self) -> None:
        """Test parent removes the last dimension."""
        key = TimeKey((("weekday", 1), ("hour", 15)))
        parent = key.parent()
        assert parent == TimeKey((("weekday", 1),))

    def test_parent_of_global_is_none(self: Self) -> None:
        """Test that parent of GLOBAL is None."""
        parent = TimeKey.GLOBAL.parent()
        assert parent is None

    def test_parent_chain(self: Self) -> None:
        """Test navigating up a parent chain."""
        key = TimeKey((("month", 3), ("weekday", 1), ("hour", 15)))

        parent1 = key.parent()
        assert parent1 == TimeKey((("month", 3), ("weekday", 1)))

        parent2 = parent1.parent()
        assert parent2 == TimeKey((("month", 3),))

        parent3 = parent2.parent()
        assert parent3 == TimeKey.GLOBAL

        parent4 = parent3.parent()
        assert parent4 is None

    def test_parent_preserves_original(self: Self) -> None:
        """Test that calling parent() doesn't modify the original key."""
        original = TimeKey((("weekday", 1), ("hour", 15)))
        original_items = original.items
        parent = original.parent()

        assert original.items == original_items  # Original unchanged
        assert len(parent) == len(original) - 1


class TestTimeKeyParents:
    """Tests for the parents() iterator method."""

    def test_parents_single_dimension(self: Self) -> None:
        """Test parents iterator for a single dimension key."""
        key = TimeKey((("hour", 15),))
        parents_list = list(key.parents())

        assert len(parents_list) == 2
        assert parents_list[0] == key
        assert parents_list[1] == TimeKey.GLOBAL

    def test_parents_multiple_dimensions(self: Self) -> None:
        """Test parents iterator for multiple dimensions."""
        key = TimeKey((("weekday", 1), ("hour", 15)))
        parents_list = list(key.parents())

        assert len(parents_list) == 3
        assert parents_list[0] == TimeKey((("weekday", 1), ("hour", 15)))
        assert parents_list[1] == TimeKey((("weekday", 1),))
        assert parents_list[2] == TimeKey.GLOBAL

    def test_parents_global(self: Self) -> None:
        """Test that parents of GLOBAL only yields GLOBAL itself."""
        parents_list = list(TimeKey.GLOBAL.parents())

        assert len(parents_list) == 1
        assert parents_list[0] == TimeKey.GLOBAL

    def test_parents_complex_hierarchy(self: Self) -> None:
        """Test parents iterator for a complex hierarchy."""
        key = TimeKey((("year", 2024), ("month", 3), ("weekday", 1), ("hour", 15)))
        parents_list = list(key.parents())

        assert len(parents_list) == 5
        assert parents_list[0] == key
        assert parents_list[1] == TimeKey(
            (("year", 2024), ("month", 3), ("weekday", 1))
        )
        assert parents_list[2] == TimeKey((("year", 2024), ("month", 3)))
        assert parents_list[3] == TimeKey((("year", 2024),))
        assert parents_list[4] == TimeKey.GLOBAL

    def test_parents_is_iterator(self: Self) -> None:
        """Test that parents() returns an iterator (not a list)."""
        key = TimeKey((("hour", 15),))
        parents_iter = key.parents()

        # Should be able to iterate one at a time
        first = next(parents_iter)
        assert first == key

        second = next(parents_iter)
        assert second == TimeKey.GLOBAL

        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(parents_iter)

    def test_parents_can_iterate_multiple_times(self: Self) -> None:
        """Test that we can call parents() multiple times."""
        key = TimeKey((("weekday", 1), ("hour", 15)))

        first_iteration = list(key.parents())
        second_iteration = list(key.parents())

        assert first_iteration == second_iteration


class TestTimeKeyEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_dimension_value_types(self: Self) -> None:
        """Test that various hashable types can be used as dimension values."""
        # Integer values
        key1 = TimeKey((("hour", 15),))
        assert key1.items[0][1] == 15

        # String values
        key2 = TimeKey((("period", "morning"),))
        assert key2.items[0][1] == "morning"

        # Tuple values (hashable)
        key3 = TimeKey((("range", (10, 20)),))
        assert key3.items[0][1] == (10, 20)

    def test_same_dimension_different_values(self: Self) -> None:
        """Test keys with the same dimension but different values."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 16),))

        assert key1 != key2
        assert hash(key1) != hash(key2)

    def test_immutability(self: Self) -> None:
        """Test that TimeKey items tuple is immutable."""
        key = TimeKey((("hour", 15),))
        original_items = key.items

        # items is a tuple, so individual elements cannot be modified
        with pytest.raises(TypeError):
            key.items[0] = ("hour", 16)

        # Original should be unchanged
        assert key.items == original_items

    def test_deep_hierarchy(self: Self) -> None:
        """Test a very deep hierarchy."""
        dimensions = tuple((f"dim{i}", i) for i in range(10))
        key = TimeKey(dimensions)

        assert len(key) == 10

        parents_list = list(key.parents())
        assert len(parents_list) == 11  # 10 levels + GLOBAL

    def test_unicode_dimension_names(self: Self) -> None:
        """Test that unicode characters work in dimension names."""
        key = TimeKey((("时间", 15), ("日期", 1)))
        assert len(key) == 2
        assert key.items[0][0] == "时间"


class TestTimeKeyUsagePatterns:
    """Tests for common usage patterns."""

    def test_as_dict_key_with_values(self: Self) -> None:
        """Test using TimeKey as dictionary key with associated values."""
        key1 = TimeKey((("hour", 15),))
        key2 = TimeKey((("hour", 16),))
        key3 = TimeKey((("hour", 15),))  # Same as key1

        data = {}
        data[key1] = "value1"
        data[key2] = "value2"

        assert data[key1] == "value1"
        assert data[key3] == "value1"  # key3 equals key1
        assert data[key2] == "value2"
        assert len(data) == 2

    def test_hierarchical_lookup_pattern(self: Self) -> None:
        """Test a hierarchical lookup pattern."""
        # Create a hierarchical data structure
        data = {
            TimeKey.GLOBAL: "global_default",
            TimeKey((("hour", 15),)): "hour_specific",
            TimeKey((("hour", 15), ("weekday", 1))): "very_specific",
        }

        # Look up specific key
        specific_key = TimeKey((("hour", 15), ("weekday", 1)))
        assert data[specific_key] == "very_specific"

        # Look up parent
        parent_key = specific_key.parent()
        assert data[parent_key] == "hour_specific"

        # Look up global
        assert data[TimeKey.GLOBAL] == "global_default"

    def test_fallback_lookup_pattern(self: Self) -> None:
        """Test a fallback lookup pattern using parents()."""
        # Create partial data
        data = {TimeKey.GLOBAL: "default", TimeKey((("hour", 15),)): "hour_value"}

        # Try to find value for a specific key, falling back to parents
        search_key = TimeKey((("hour", 15), ("weekday", 1)))

        result = None
        for parent in search_key.parents():
            if parent in data:
                result = data[parent]
                break

        assert result == "hour_value"  # Found in parent

    def test_collect_all_parent_values(self: Self) -> None:
        """Test collecting values from all parents in hierarchy."""
        data = {
            TimeKey.GLOBAL: 1,
            TimeKey((("hour", 15),)): 10,
            TimeKey((("hour", 15), ("weekday", 1))): 100,
        }

        key = TimeKey((("hour", 15), ("weekday", 1)))
        values = [data.get(parent, 0) for parent in key.parents()]

        assert values == [100, 10, 1]


class TestTimeKeySerialization:
    """Tests for TimeKey serialization (to_dict/from_dict)."""

    def test_to_dict_empty_key(self: Self) -> None:
        """Test serializing an empty TimeKey (GLOBAL)."""
        key = TimeKey.GLOBAL
        data = key.to_dict()
        assert data == []
        assert isinstance(data, list)

    def test_to_dict_single_dimension(self: Self) -> None:
        """Test serializing a TimeKey with one dimension."""
        key = TimeKey((("hour", 15),))
        data = key.to_dict()
        assert data == [["hour", 15]]
        assert isinstance(data, list)
        assert len(data) == 1

    def test_to_dict_multiple_dimensions(self: Self) -> None:
        """Test serializing a TimeKey with multiple dimensions."""
        key = TimeKey((("weekday", 1), ("hour", 15)))
        data = key.to_dict()
        assert data == [["weekday", 1], ["hour", 15]]
        assert len(data) == 2

    def test_to_dict_complex_key(self: Self) -> None:
        """Test serializing a complex TimeKey."""
        key = TimeKey(
            (("month", 3), ("weekday", 2), ("hour", 14), ("minute_bucket", 1))
        )
        data = key.to_dict()
        assert data == [
            ["month", 3],
            ["weekday", 2],
            ["hour", 14],
            ["minute_bucket", 1],
        ]
        assert len(data) == 4

    def test_to_dict_with_string_values(self: Self) -> None:
        """Test serializing a TimeKey with string values."""
        key = TimeKey((("category", "work"), ("priority", "high")))
        data = key.to_dict()
        assert data == [["category", "work"], ["priority", "high"]]

    def test_from_dict_empty_list(self: Self) -> None:
        """Test deserializing an empty list to GLOBAL."""
        data = []
        key = TimeKey.from_dict(data)
        assert key == TimeKey.GLOBAL
        assert len(key) == 0

    def test_from_dict_single_dimension(self: Self) -> None:
        """Test deserializing a single dimension."""
        data = [["hour", 15]]
        key = TimeKey.from_dict(data)
        assert key.items == (("hour", 15),)
        assert len(key) == 1

    def test_from_dict_multiple_dimensions(self: Self) -> None:
        """Test deserializing multiple dimensions."""
        data = [["weekday", 1], ["hour", 15]]
        key = TimeKey.from_dict(data)
        assert key.items == (("weekday", 1), ("hour", 15))
        assert len(key) == 2

    def test_from_dict_complex_key(self: Self) -> None:
        """Test deserializing a complex key."""
        data = [["month", 3], ["weekday", 2], ["hour", 14], ["minute_bucket", 1]]
        key = TimeKey.from_dict(data)
        assert key.items == (
            ("month", 3),
            ("weekday", 2),
            ("hour", 14),
            ("minute_bucket", 1),
        )
        assert len(key) == 4

    def test_from_dict_with_string_values(self: Self) -> None:
        """Test deserializing with string values."""
        data = [["category", "work"], ["priority", "high"]]
        key = TimeKey.from_dict(data)
        assert key.items == (("category", "work"), ("priority", "high"))

    def test_roundtrip_empty_key(self: Self) -> None:
        """Test round-trip serialization of empty key."""
        original = TimeKey.GLOBAL
        data = original.to_dict()
        restored = TimeKey.from_dict(data)
        assert restored == original
        assert hash(restored) == hash(original)

    def test_roundtrip_single_dimension(self: Self) -> None:
        """Test round-trip serialization of single dimension."""
        original = TimeKey((("hour", 15),))
        data = original.to_dict()
        restored = TimeKey.from_dict(data)
        assert restored == original
        assert restored.items == original.items
        assert hash(restored) == hash(original)

    def test_roundtrip_multiple_dimensions(self: Self) -> None:
        """Test round-trip serialization of multiple dimensions."""
        original = TimeKey((("weekday", 1), ("hour", 15), ("minute_bucket", 2)))
        data = original.to_dict()
        restored = TimeKey.from_dict(data)
        assert restored == original
        assert restored.items == original.items
        assert hash(restored) == hash(original)

    def test_roundtrip_preserves_order(self: Self) -> None:
        """Test that round-trip preserves dimension order."""
        original = TimeKey((("hour", 15), ("weekday", 1)))
        data = original.to_dict()
        restored = TimeKey.from_dict(data)
        # Order matters for equality
        assert restored.items == original.items
        assert restored == original

    def test_roundtrip_various_value_types(self: Self) -> None:
        """Test round-trip with various hashable value types."""
        # int, str, float, tuple
        original = TimeKey((("int_val", 42), ("str_val", "test"), ("float_val", 3.14)))
        data = original.to_dict()
        restored = TimeKey.from_dict(data)
        assert restored == original
        assert restored.items == original.items

    def test_serialized_data_is_json_compatible(self: Self) -> None:
        """Test that serialized data can be JSON-encoded."""
        key = TimeKey((("hour", 15), ("weekday", 1)))
        data = key.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Verify we can restore from parsed JSON
        restored = TimeKey.from_dict(parsed)
        assert restored == key

    def test_multiple_keys_serialization(self: Self) -> None:
        """Test serializing multiple different keys."""
        keys = [
            TimeKey.GLOBAL,
            TimeKey((("hour", 10),)),
            TimeKey((("hour", 10), ("weekday", 2))),
            TimeKey((("month", 5), ("weekday", 3), ("hour", 14))),
        ]

        # Serialize all
        serialized = [k.to_dict() for k in keys]

        # Deserialize all
        restored = [TimeKey.from_dict(data) for data in serialized]

        # Verify all match
        for original, restored_key in zip(keys, restored, strict=True):
            assert restored_key == original
            assert restored_key.items == original.items
