"""Runtime parameters for the model."""

from dataclasses import dataclass

from .forecaster_engine_runtime_parameters import ForecasterEngineRuntimeParameters


@dataclass()
class RuntimeParameters:
    """
    Parameters for runtime configuring the model.

    This dataclass contains all parameters used to configure the behavior
    of the model, including decay rates, drift detection thresholds,
    and persistence modeling.
    """

    engine: ForecasterEngineRuntimeParameters
    """Runtime parameters for the ForecasterEngine, including decay rates and drift detection."""

    min_prune_interval_factor: float = 5.0
    """
    Multiplier for minimum interval between prune operations
    (typically 5-10 x base_half_life).
    """
