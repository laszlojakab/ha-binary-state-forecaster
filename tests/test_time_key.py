from typing import Self

from custom_components.discrete_state_forecaster.model.temporal.temporal_feature import (
    TemporalFeature,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import TimeKey


class TestTimeKeyBasic:
    def test_global_is_empty(self: Self) -> None:
        assert len(TimeKey.GLOBAL) == 0
        assert TimeKey.GLOBAL.to_tuple() == ()
        assert repr(TimeKey.GLOBAL) == "GLOBAL"

    def test_from_tuple_and_to_tuple_roundtrip(self: Self) -> None:
        data = (("weekday", 1), ("hour", 15))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data
        restored = TimeKey.from_tuple(key.to_tuple())
        assert restored == key
        assert hash(restored) == hash(key)

    def test_len_matches_number_of_features(self: Self) -> None:
        data = (("a", 1), ("b", 2), ("c", 3))
        key = TimeKey.from_tuple(data)
        assert len(key) == 3


class TestTimeKeyRepresentationAndOrder:
    def test_repr_shows_root_to_leaf_order(self: Self) -> None:
        data = (("hour", 15), ("day", "Mon"))
        key = TimeKey.from_tuple(data)
        assert repr(key) == "hour = 15, day = Mon"

    def test_order_matters_for_equality_and_hash(self: Self) -> None:
        data1 = (("hour", 15), ("weekday", 1))
        data2 = (("weekday", 1), ("hour", 15))
        k1 = TimeKey.from_tuple(data1)
        k2 = TimeKey.from_tuple(data2)
        assert k1 != k2
        assert hash(k1) != hash(k2)


class TestTimeKeyHashability:
    def test_usable_as_dict_key_and_set(self: Self) -> None:
        a = TimeKey.from_tuple((("x", 1),))
        b = TimeKey.from_tuple((("x", 1),))
        c = TimeKey.from_tuple((("x", 2),))
        d = {a: "one", c: "two"}
        assert d[b] == "one"
        s = {a, b, c}
        assert len(s) == 2


class TestTimeKeyAddMethod:
    def test_add_feature_to_global(self: Self) -> None:
        f = TemporalFeature("hour", 14)
        key = TimeKey.GLOBAL + f
        assert len(key) == 1
        assert key.to_tuple() == (("hour", 14),)
        assert repr(key) == "hour = 14"

    def test_add_multiple_features(self: Self) -> None:
        f1 = TemporalFeature("hour", 14)
        f2 = TemporalFeature("day", "Mon")
        key = TimeKey.GLOBAL + f1 + f2
        assert len(key) == 2
        assert key.to_tuple() == (("hour", 14), ("day", "Mon"))
        assert repr(key) == "hour = 14, day = Mon"

    def test_add_builds_correct_chain(self: Self) -> None:
        f1 = TemporalFeature("a", 1)
        f2 = TemporalFeature("b", 2)
        f3 = TemporalFeature("c", 3)
        key = TimeKey.GLOBAL + f1 + f2 + f3
        assert len(key) == 3
        assert key.to_tuple() == (("a", 1), ("b", 2), ("c", 3))

    def test_add_preserves_order(self: Self) -> None:
        f1 = TemporalFeature("x", 1)
        f2 = TemporalFeature("y", 2)
        key1 = TimeKey.GLOBAL + f1 + f2
        key2 = TimeKey.GLOBAL + f2 + f1
        assert key1 != key2
        assert key1.to_tuple() == (("x", 1), ("y", 2))
        assert key2.to_tuple() == (("y", 2), ("x", 1))

    def test_add_returns_new_timekey(self: Self) -> None:
        base = TimeKey.from_tuple((("hour", 14),))
        f = TemporalFeature("day", "Mon")
        new_key = base + f
        assert base.to_tuple() == (("hour", 14),)
        assert new_key.to_tuple() == (("hour", 14), ("day", "Mon"))
        assert base != new_key


class TestTimeKeyProperties:
    def test_is_root_global(self: Self) -> None:
        assert TimeKey.GLOBAL.is_root is True

    def test_is_root_non_global(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14),))
        assert key.is_root is False


class TestTimeKeyAncestors:
    def test_ancestors_global(self: Self) -> None:
        ancestors = list(TimeKey.GLOBAL.ancestors())
        assert ancestors == []

    def test_ancestors_single_feature(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14),))
        ancestors = list(key.ancestors())
        assert len(ancestors) == 1
        assert ancestors[0] == key

    def test_ancestors_multiple_features(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14), ("day", "Mon"), ("month", 3)))
        ancestors = list(key.ancestors())
        assert len(ancestors) == 3
        assert ancestors[0].to_tuple() == (("hour", 14), ("day", "Mon"), ("month", 3))
        assert ancestors[1].to_tuple() == (("hour", 14), ("day", "Mon"))
        assert ancestors[2].to_tuple() == (("hour", 14),)

    def test_ancestors_order_leaf_to_root(self: Self) -> None:
        key = TimeKey.from_tuple((("a", 1), ("b", 2), ("c", 3)))
        ancestors = list(key.ancestors())
        assert ancestors[0] == key
        assert len(ancestors[-1]) == 1


class TestTimeKeyEquality:
    def test_equality_same_keys(self: Self) -> None:
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14),))
        assert k1 == k2

    def test_equality_global(self: Self) -> None:
        k1 = TimeKey.GLOBAL
        k2 = TimeKey.from_tuple(())
        assert k1 == k2

    def test_inequality_different_values(self: Self) -> None:
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 15),))
        assert k1 != k2

    def test_inequality_different_lengths(self: Self) -> None:
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        assert k1 != k2

    def test_inequality_with_non_timekey(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14),))
        assert key != (("hour", 14),)
        assert key != "hour = 14"
        assert key != 14
        assert key != None

    def test_equality_reflexive(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14),))
        assert key == key

    def test_equality_transitive(self: Self) -> None:
        k1 = TimeKey.from_tuple((("hour", 14),))
        k2 = TimeKey.from_tuple((("hour", 14),))
        k3 = TimeKey.from_tuple((("hour", 14),))
        assert k1 == k2
        assert k2 == k3
        assert k1 == k3


class TestTimeKeyFromTupleValidation:
    def test_from_tuple_empty_returns_global(self: Self) -> None:
        key = TimeKey.from_tuple(())
        assert key == TimeKey.GLOBAL
        assert len(key) == 0

    def test_from_tuple_preserves_types(self: Self) -> None:
        data = (("int", 42), ("str", "test"), ("float", 3.14), ("tuple", (1, 2)))
        key = TimeKey.from_tuple(data)
        result = key.to_tuple()
        assert result == data


class TestTimeKeyEdgeCases:
    def test_deep_hierarchy(self: Self) -> None:
        data = tuple((f"dim{i}", i) for i in range(20))
        key = TimeKey.from_tuple(data)
        assert len(key) == 20
        assert key.to_tuple() == data

    def test_unicode_names_and_values(self: Self) -> None:
        data = (("时间", "下午"), ("日期", "星期一"))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data
        assert "时间" in repr(key)

    def test_special_characters_in_names(self: Self) -> None:
        data = (("key-with-dash", 1), ("key_with_underscore", 2), ("key.with.dot", 3))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data

    def test_numeric_string_values(self: Self) -> None:
        data = (("hour", "14"), ("minute", "30"))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data

    def test_mixed_value_types_in_same_key(self: Self) -> None:
        data = (("a", 1), ("b", "two"), ("c", 3.0), ("d", (4, 5)), ("e", True))
        key = TimeKey.from_tuple(data)
        assert key.to_tuple() == data


class TestTimeKeyHashConsistency:
    def test_hash_equals_implies_equal_hash(self: Self) -> None:
        k1 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        k2 = TimeKey.from_tuple((("hour", 14), ("day", "Mon")))
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_hash_stable_across_calls(self: Self) -> None:
        key = TimeKey.from_tuple((("hour", 14),))
        h1 = hash(key)
        h2 = hash(key)
        assert h1 == h2

    def test_hash_different_for_different_order(self: Self) -> None:
        k1 = TimeKey.from_tuple((("a", 1), ("b", 2)))
        k2 = TimeKey.from_tuple((("b", 2), ("a", 1)))
        assert hash(k1) != hash(k2)
