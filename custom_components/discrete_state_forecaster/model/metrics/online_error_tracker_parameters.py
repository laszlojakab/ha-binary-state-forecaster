"""Hyper-parameters for online error tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .online_error_tracker_hyper_parameters import OnlineErrorTrackerHyperParameters
    from .online_error_tracker_runtime_parameters import (
        OnlineErrorTrackerRuntimeParameters,
    )


@dataclass(frozen=True)
class OnlineErrorTrackerParameters:
    """Configuration for online prediction error tracking."""

    hyper_parameters: Final[OnlineErrorTrackerHyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    runtime_parameters: Final[OnlineErrorTrackerRuntimeParameters]
    """Runtime parameters containing dynamic configuration values."""

    @property
    def error_half_life(self: Self) -> float:
        """
        Gets the half-life for error statistics.

        Returns:
            The half-life (in the same units as base half-life) for tracking
                prediction errors. After this time, the weight of old errors
                decays to 50% of their original value.

        """
        return (
            self.hyper_parameters.half_life
            * self.runtime_parameters.error_half_life_factor
        )
