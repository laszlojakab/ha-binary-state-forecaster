from dataclasses import dataclass

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


@dataclass(frozen=True)
class ForecasterEngineRuntimeParameters:
    """Runtime parameters for ForecasterEngine."""

    hierarchical_state_stats: HierarchicalStateStatsRuntimeParameters
    """Runtime parameters for hierarchical state statistics."""

    drift_monitor: DriftMonitorRuntimeParameters
    """Runtime parameters for drift monitoring."""

    long_term_error_tracker: OnlineErrorTrackerRuntimeParameters
    """Runtime parameters for long-term error tracking."""

    short_term_error_tracker: OnlineErrorTrackerRuntimeParameters
    """Runtime parameters for short-term error tracking."""

    state_persistence_tracker: StatePersistenceTrackerRuntimeParameters
    """Runtime parameters for state persistence tracking."""
