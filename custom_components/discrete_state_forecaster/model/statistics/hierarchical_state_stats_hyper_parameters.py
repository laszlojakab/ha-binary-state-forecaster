"""Configuration parameters for hierarchical state statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Self

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.hyper_parameters import (
        HyperParameters,
    )


class HierarchicalStateStatsHyperParameters:
    """
    Configuration for hierarchical state statistics prediction engine.

    Manages parameters that control how predictions are made, particularly the
    minimum support thresholds. The actual threshold is computed from a base
    value (half_life) and a scaling factor to allow fine-tuning of prediction
    confidence requirements.

    The minimum support threshold determines whether a distribution has enough
    data to be used for making predictions. Higher thresholds make predictions
    more conservative (requiring more data), while lower thresholds make them
    more permissive.

    Example:
        >>> base_hp = HyperParameters(half_life=50.0)
        >>> hp = HierarchicalStateStatsHyperParameters(base_hp, min_support_factor=0.5)
        >>> hp.min_support
        25.0

    """

    _hyper_parameters: Final[HyperParameters]
    """Base hyper parameters containing global configuration values."""

    _min_support_factor: Final[float]
    """Scaling factor for minimum support threshold calculation."""

    def __init__(
        self: Self, hyper_parameters: HyperParameters, min_support_factor: float = 1.0
    ):
        """
        Initializes hierarchical state statistics configuration.

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
        """
        Computes the minimum support threshold for predictions.

        Multiplies the base half_life parameter by the adjustment factor to
        allow flexible control of prediction confidence requirements.

        Returns:
            The minimum support threshold. Distributions must have total support
                >= this value to be considered confident for prediction.

        """
        return self._hyper_parameters.half_life * self._min_support_factor

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Returns a JSON-serializable representation of the instance.

        Returns:
          A dictionary containing the minimum support threshold.
        """
        return {"min_support": self.min_support}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> HierarchicalStateStatsHyperParameters:
        """
        Creates an instance of HierarchicalStateStatsHyperParameters from a dictionary.

        Args:
          data: A dictionary containing the minimum support threshold.
          hyper_parameters: The base HyperParameters instance to use for calculations.

        Returns:
          An instance of HierarchicalStateStatsHyperParameters initialized with the provided
          minimum support threshold and base hyper parameters.
        """
        return cls(
            hyper_parameters=hyper_parameters,
            min_support_factor=data["min_support"] / hyper_parameters.half_life,
        )
