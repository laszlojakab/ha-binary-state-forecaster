"""
Utility helpers for serializing and deserializing hashable states.

These helpers produce JSON-serializable dictionaries for common hashable
Python types and provide a best-effort deserialization. For unknown types the
serializer falls back to `repr` and the deserializer returns that repr string.
"""
from __future__ import annotations

from typing import Any


def serialize_state(state: Any) -> dict:
    """
    Serializes a hashable state into a JSON-able dict.

    Supported types: None, bool, int, float, str, tuple (recursively).
    Unknown types are represented using their `repr` string.
    """
    if state is None:
        return {"type": "none", "value": None}
    if isinstance(state, bool):
        return {"type": "bool", "value": state}
    if isinstance(state, int) and not isinstance(state, bool):
        return {"type": "int", "value": state}
    if isinstance(state, float):
        return {"type": "float", "value": state}
    if isinstance(state, str):
        return {"type": "str", "value": state}
    if isinstance(state, tuple):
        return {"type": "tuple", "value": [serialize_state(s) for s in state]}

    return {"type": "repr", "value": repr(state)}


def deserialize_state(obj: dict) -> Any:
    """
    Deserializes a state previously serialized with `serialize_state`.

    If the serializer used the `repr` fallback, the original object cannot be
    reconstructed; the `value` (repr string) is returned instead.
    """
    t = obj.get("type")
    v = obj.get("value")
    if t == "none":
        return None
    if t == "bool":
        return bool(v)
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "str":
        return str(v)
    if t == "tuple":
        return tuple(deserialize_state(e) for e in v)

    return v
