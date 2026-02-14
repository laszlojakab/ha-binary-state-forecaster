from dataclasses import dataclass
from typing import Final

from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_runtime_parameters import (
    StatePersistenceTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_runtime_parameters import (
    OnlineErrorTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_runtime_parameters import (
    HierarchicalStateStatsRuntimeParameters,
)


@dataclass()
class ForecasterEngineRuntimeParameters:
    """Runtime parameters for ForecasterEngine."""

    hierarchical_state_stats: Final[HierarchicalStateStatsRuntimeParameters]
    """Runtime parameters for hierarchical state statistics."""

    drift_monitor: Final[DriftMonitorRuntimeParameters]
    """Runtime parameters for drift monitoring."""

    long_term_error_tracker: Final[OnlineErrorTrackerRuntimeParameters]
    """Runtime parameters for long-term error tracking."""

    short_term_error_tracker: Final[OnlineErrorTrackerRuntimeParameters]
    """Runtime parameters for short-term error tracking."""

    state_persistence_tracker: Final[StatePersistenceTrackerRuntimeParameters]
    """Runtime parameters for state persistence tracking."""

    half_life: float
    """Base half-life value for decay calculations (in seconds)."""

    min_prune_interval_factor: float
    """Multiplier for minimum interval between prune operations"""

    persistence_strength: float
    """Strength of persistence modeling (0.0 = no persistence, 1.0 = full persistence)."""
