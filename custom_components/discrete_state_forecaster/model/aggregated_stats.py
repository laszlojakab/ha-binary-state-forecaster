"""
Aggregated statistics from hierarchical state data.

This module defines the AggregatedStats dataclass used to represent
state distribution statistics that have been aggregated across one or
more levels of a hierarchical time-based model.
"""

from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.state import State


@dataclass
class AggregatedStats:
    """
    Aggregated state distribution statistics across hierarchical levels.

    This dataclass represents the result of aggregating state statistics
    across one or more levels of a hierarchical temporal model. It includes
    the probability distribution of states, the total support time used for
    the aggregation, and the depth indicating how many hierarchical levels
    were combined.

    Attributes:
        distribution: Probability distribution mapping states to their
            probabilities (values sum to 1.0). Each state's probability
            represents the proportion of time spent in that state based
            on the aggregated historical data.
        support_time: Total support time in seconds that contributed to
            this distribution. This represents the cumulative duration of
            observations used to calculate the probabilities. Higher values
            indicate more data and typically more reliable predictions.
        depth: Number of hierarchical levels that were aggregated to produce
            this distribution. A depth of 1 means only a single specific
            level was used, while higher values indicate blending across
            multiple levels (e.g., specific time + parent time + global).

    Example:
        ```
        >>> stats = AggregatedStats(
        ...     distribution={"on": 0.7, "off": 0.3},
        ...     support_time=3600.0,  # 1 hour of observations
        ...     depth=2  # Specific level + parent level
        ... )
        >>> print(stats.distribution["on"])
        0.7
        >>> print(f"Based on {stats.support_time}s of data")
        Based on 3600.0s of data
        ```
    """

    distribution: dict[State, float]
    support_time: float
    depth: int
