from dataclasses import dataclass
from typing import Final

from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller_runtime_parameters import (
    HyperParameterControllerRuntimeParameters,
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

    hyper_parameter_controller: Final[HyperParameterControllerRuntimeParameters]
    """Runtime parameters for hyperparameter control."""

    background_decay_half_life_factor: float = 0.0
    """
    Multiplier for background (dormant-key) decay.

    ``0.0`` disables background decay completely (pure per-key
    observation-weighted decay, the default).  A positive value ``f`` causes
    *all* keys – including dormant ones – to receive a slow exponential decay
    whose effective half-life is ``f * base_half_life``.
    """
