"""Configuration parameters for hierarchical state statistics.

This module provides `HierarchicalStateStatsHyperParameters`, a configuration
class that encapsulates parameters controlling the behavior of the hierarchical
prediction engine, particularly the minimum support thresholds that determine
prediction confidence.
"""
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)


class HierarchicalStateStatsHyperParameters:
    """Configuration for hierarchical state statistics prediction engine.

    Manages parameters that control how predictions are made, particularly the
    minimum support thresholds. The actual threshold is computed from a base
    value (half_life) and a scaling factor to allow fine-tuning of prediction
    confidence requirements.

    The minimum support threshold determines whether a distribution has enough
    data to be used for making predictions. Higher thresholds make predictions
    more conservative (requiring more data), while lower thresholds make them
    more permissive.

    Attributes:
        _hyper_parameters: The base hyper parameters containing global
            configuration like half_life for decay calculations.
        _min_support_factor: Scaling factor applied to half_life to compute
            the actual minimum support threshold. Defaults to 1.0 (use half_life
            directly).

    Example:
        >>> from custom_components.discrete_state_forecaster.model.hyper_parameters import HyperParameters
        >>> base_hp = HyperParameters(half_life=50.0)
        >>> hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.5)
        >>> hp.min_support
        25.0
    """
    def __init__(
        self: Self, hyper_parameters: HyperParameters, min_support_factor: float = 1.0
    ):
        """Initialize hierarchical state statistics configuration.

        Args:
            hyper_parameters: Base hyper parameters containing global settings
                like half_life.
            min_support_factor: Scaling factor for minimum support threshold
                (default 1.0). The actual threshold is computed as:
                min_support = half_life * min_support_factor.
                Values < 1.0 make predictions more permissive, while values > 1.0
                make them more conservative.
        """
        self._hyper_parameters: Final = hyper_parameters
        self._min_support_factor: Final = min_support_factor

    @property
    def min_support(self: Self) -> float:
        """Compute the minimum support threshold for predictions.

        Multiplies the base half_life parameter by the adjustment factor to
        allow flexible control of prediction confidence requirements.

        Returns:
            The minimum support threshold. Distributions must have total support
                >= this value to be considered confident for prediction.
        """
        return self._hyper_parameters.half_life * self._min_support_factor
