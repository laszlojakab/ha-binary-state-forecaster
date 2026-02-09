import math
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.state import (
    State,
)
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)

from .online_error_tracker_hyper_parameters import OnlineErrorTrackerHyperParameters


class OnlineErrorTracker:
    """
    Tracks prediction error online using exponentially decayed statistics.
    """

    def __init__(
        self: Self,
        hyper_parameters: OnlineErrorTrackerHyperParameters,
    ):
        self._hyper_parameters: Final = hyper_parameters
        self._mean_error: float = 0.0
        self._var_error: float = 0.0
        self._last_ts: float | None = None

    def update(
        self: Self,
        prediction: DistributionStats,
        y_true: State,
        timestamp: float,
    ) -> None:
        """
        Update error statistics using negative log-likelihood.
        """
        p = prediction.distribution().get(y_true, 0.0)

        # Clamp to avoid log(0)
        error = -math.log(max(p, 1e-12))

        if self._last_ts is None:
            self._mean_error = error
            self._var_error = 0.0
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        _lambda = math.log(2.0) / (self._hyper_parameters.error_half_life)
        decay = math.exp(-_lambda * dt)

        diff = error - self._mean_error

        self._mean_error = decay * self._mean_error + (1.0 - decay) * error
        self._var_error = decay * self._var_error + (1.0 - decay) * diff * diff

        self._last_ts = timestamp

    @property
    def mean(self: Self) -> float:
        return self._mean_error

    @property
    def std(self: Self) -> float:
        return math.sqrt(max(self._var_error, 1e-12))
