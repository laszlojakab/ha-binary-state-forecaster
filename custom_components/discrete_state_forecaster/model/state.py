"""
Type definition for state values in the prediction model.

This module defines the State type alias used throughout the discrete state
forecaster. A State represents any hashable value that can be predicted by
the model, such as device states, operational modes, or any categorical value
that can be learned from historical patterns.

The State type is intentionally flexible to support various use cases:
- Binary states: "on"/"off", True/False, 0/1
- Multi-state systems: "heating"/"cooling"/"idle", "home"/"away"/"vacation"
- Numeric categories: temperature zones, speed levels, etc.
- Complex states: tuples or other hashable types for composite states

The only requirement is that states must be hashable so they can be used
as dictionary keys for probability distributions and statistical tracking.

Example:
    ```
    # String states (most common)
    state: State = "on"

    # Numeric states
    state: State = 1

    # Boolean states
    state: State = True

    # Tuple states for composite conditions
    state: State = ("heating", "high_speed")
    ```
"""

from collections.abc import Hashable

State = Hashable
"""
Type alias for state values in the prediction model.
Any hashable value can represent a state - strings, numbers, booleans, tuples, etc.
Common examples: "on", "off", "heating", "cooling", "idle", 0, 1, True, False
"""
