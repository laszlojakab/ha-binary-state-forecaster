"""Configuration parameters for hierarchical state statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .hierarchical_state_stats_hyper_parameters import (
        HierarchicalStateStatsHyperParameters,
    )
    from .hierarchical_state_stats_runtime_parameters import (
        HierarchicalStateStatsRuntimeParameters,
    )


@dataclass(frozen=True)
class HierarchicalStateStatsParameters:
    """Parameters for hierarchical state statistics."""

    hyper_parameters: Final[HierarchicalStateStatsHyperParameters]
    """
    Hierarchical state statistics hyper parameters containing configuration values which are updated
    by hyper parameter optimization processes.
    """

    runtime_parameters: Final[HierarchicalStateStatsRuntimeParameters]
    """Hierarchical state statistics runtime parameters containing dynamic configuration values."""

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
        return (
            self.hyper_parameters.half_life * self.runtime_parameters.min_support_factor
        )
