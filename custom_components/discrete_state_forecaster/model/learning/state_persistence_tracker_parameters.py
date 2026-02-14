"""Parameters for state persistence tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from .state_persistence_tracker_hyper_parameters import (
        StatePersistenceTrackerHyperParameters,
    )
    from .state_persistence_tracker_runtime_parameters import (
        StatePersistenceTrackerRuntimeParameters,
    )


class StatePersistenceTrackerParameters:
    """Configuration for state persistence tracking behavior."""

    _hyper_parameters: Final[StatePersistenceTrackerHyperParameters]
    """Base hyper-parameters containing the reference half-life."""

    _runtime_parameters: Final[StatePersistenceTrackerRuntimeParameters]

    def __init__(
        self: Self,
        hyper_parameters: StatePersistenceTrackerHyperParameters,
        runtime_parameters: StatePersistenceTrackerRuntimeParameters,
    ) -> None:
        """
        Initialize state persistence tracker hyper-parameters.

        Args:
            hyper_parameters: Base hyper-parameters providing reference half-life.
            runtime_parameters: Runtime parameters containing dynamic configuration values.
        """
        self._hyper_parameters: Final = hyper_parameters
        self._runtime_parameters: Final = runtime_parameters

    @property
    def persistence_half_life(self: Self) -> float:
        """
        Get the half-life for persistence duration statistics.

        Returns:
            The half-life (in same units as base) for duration statistics decay.

        """
        return (
            self._hyper_parameters.half_life
            * self._runtime_parameters.persistence_half_life_factor
        )
