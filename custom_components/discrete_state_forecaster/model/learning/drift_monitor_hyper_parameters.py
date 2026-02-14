"""
Hyper-parameters for drift monitoring.

This module provides configuration for the DriftMonitor, controlling how concept
drift is detected through comparison of fast and slow baseline distributions.
"""

from typing import Final, Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)


class DriftMonitorHyperParameters:
    """
    Hyper-parameters for concept drift detection.

    Controls the behavior of drift detection including baseline half-lives,
    drift thresholds, and adaptive threshold adjustment.
    """

    _hyper_parameters: Final[ForecasterEngineHyperParameters]
    """Base hyper-parameters providing the half-life value used for baselines."""

    def __init__(
        self: Self,
        hyper_parameters: ForecasterEngineHyperParameters,
    ):
        """
        Initialize drift monitor hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters.
        """
        self._hyper_parameters: Final = hyper_parameters

    @property
    def half_life(self: Self) -> float:
        """Get base half-life from underlying hyper-parameters."""
        return self._hyper_parameters.half_life
