"""
Online prediction error tracking using exponential decay.

This module provides OnlineErrorTracker, which maintains running statistics of
prediction errors using negative log-likelihood as the error metric. Statistics
are updated with exponential decay to give more weight to recent errors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Self

from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)

from .online_error_tracker_hyper_parameters import OnlineErrorTrackerHyperParameters

if TYPE_CHECKING:
    from custom_components.discrete_state_forecaster.model.state import (
        State,
    )
    from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
        DistributionStats,
    )


class OnlineErrorTracker:
    """
    Tracks prediction error online using exponentially weighted statistics.

    Maintains mean and variance of prediction errors using negative log-likelihood
    (NLL) as the error metric. More confident correct predictions have lower NLL,
    while incorrect or uncertain predictions have higher NLL.

    The tracker uses exponential weighting controlled by a half-life parameter,
    allowing recent errors to have more influence than older errors.

    Attributes:
        _hyper_parameters: Configuration controlling error decay rate.
        _mean_error: Current mean of negative log-likelihood errors.
        _var_error: Current variance of negative log-likelihood errors.
        _last_ts: Timestamp of last update, or None if never updated.

    Example:
        >>> from custom_components.discrete_state_forecaster.model.hyper_parameters import (  # noqa: E501
        ...     HyperParameters,
        ... )
        >>> base_hp = HyperParameters(
        ...     half_life=50.0,
        ...     min_prune_interval=10.0,
        ...     prune_enabled=True,
        ...     persistence_strength=0.95,
        ... )
        >>> hp = OnlineErrorTrackerHyperParameters(
        ...     hyper_parameters=base_hp,
        ...     error_half_life_factor=1.0,
        ... )
        >>> tracker = OnlineErrorTracker(hp)
        >>> dist = DistributionStats()
        >>> dist.update("on", 2.0)
        >>> dist.update("off", 1.0)
        >>> tracker.update(dist, "on", 100.0)
        >>> tracker.mean > 0  # Some error because prediction wasn't 100% certain
        True

    """

    def __init__(
        self: Self,
        hyper_parameters: OnlineErrorTrackerHyperParameters,
    ):
        """
        Initialize the online error tracker.

        Args:
            hyper_parameters: Configuration controlling error decay and tracking.

        """
        self._hyper_parameters: Final = hyper_parameters
        self._mean: float = 0.0
        self._var: float = 0.0
        self._last_ts: float | None = None

    def update(
        self: Self,
        prediction: DistributionStats,
        y_true: State,
        timestamp: float,
    ) -> None:
        """
        Update error statistics with a new prediction outcome.

        Computes negative log-likelihood of the true outcome given the prediction,
        then updates mean and variance using exponential weighting based on the
        time elapsed since the last update.

        Args:
            prediction: The predicted probability distribution over states.
            y_true: The true state that was observed.
            timestamp: Time of this observation (monotonically increasing).

        """
        p = prediction.distribution().get(y_true, 0.0)

        # Clamp to avoid log(0)
        error = -math.log(max(p, 1e-12))

        if self._last_ts is None:
            self._mean = error
            self._var = 0.0
            self._last_ts = timestamp
            return

        dt = timestamp - self._last_ts
        if dt <= 0:
            return

        _lambda = math.log(2.0) / (self._hyper_parameters.error_half_life)
        decay = math.exp(-_lambda * dt)

        diff = error - self._mean

        self._mean = decay * self._mean + (1.0 - decay) * error
        self._var = decay * self._var + (1.0 - decay) * diff * diff

        self._last_ts = timestamp

    @property
    def mean(self: Self) -> float:
        """
        Get the current mean prediction error.

        Returns:
            Mean negative log-likelihood of predictions. Lower values indicate
                better prediction performance on average.

        """
        return self._mean

    @property
    def std(self: Self) -> float:
        """
        Get the current standard deviation of prediction errors.

        Returns:
            Standard deviation of negative log-likelihood errors. Lower values
                indicate more consistent prediction performance.

        """
        return math.sqrt(max(self._var, 1e-12))

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "mean": self._mean,
            "var": self._var,
            "last_ts": self._last_ts,
            "hyper_parameters": self._hyper_parameters.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], hyper_parameters: HyperParameters
    ) -> OnlineErrorTracker:
        tracker = cls(
            hyper_parameters=OnlineErrorTrackerHyperParameters.from_dict(
                data["hyper_parameters"], hyper_parameters
            )
        )
        tracker._mean = data.get("mean", 0.0)
        tracker._var = data.get("var", 0.0)
        tracker._last_ts = data.get("last_ts")
        return tracker
