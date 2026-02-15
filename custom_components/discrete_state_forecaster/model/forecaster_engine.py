from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Final, Self

from custom_components.discrete_state_forecaster.model.forecaster_engine_hyper_parameters import (
    ForecasterEngineHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor import (
    DriftMonitor,
    DriftMonitorHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.drift_monitor_runtime_parameters import (
    DriftMonitorRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.learning.hyper_parameter_controller import (
    HyperParameterController,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker import (
    StatePersistenceTracker,
    StatePersistenceTrackerHyperParameters,
)
from custom_components.discrete_state_forecaster.model.learning.state_persistence_tracker_runtime_parameters import (
    StatePersistenceTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker import (
    OnlineErrorTracker,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_hyper_parameters import (
    OnlineErrorTrackerHyperParameters,
)
from custom_components.discrete_state_forecaster.model.metrics.online_error_tracker_runtime_parameters import (
    OnlineErrorTrackerRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.distribution_stats import (
    DistributionStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats import (
    HierarchicalStateStats,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (
    HierarchicalStateStatsHyperParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_runtime_parameters import (
    HierarchicalStateStatsRuntimeParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.prediction_result import (
    PredictionResult,
)
from custom_components.discrete_state_forecaster.model.temporal.time_key import (
    TimeKey,
)

from .forecaster_engine_runtime_parameters import (
    ForecasterEngineRuntimeParameters,
)
from .state import State

# @dataclass(frozen=True)
# class ForecasterEngineParameters:
#     """
#     Parameters for configuring the ForecasterEngine.

#     This dataclass contains all hyperparameters used to configure the behavior
#     of the forecasting engine, including decay rates, drift detection thresholds,
#     and persistence modeling.

#     Typical relationships:
#         half_life        ≈ typical_change
#         fast_half_life   ≈ 1-2  typical_change
#         slow_half_life   ≈ 10-30 x typical_change
#         drift_half_life  ≈ 1.5 x slow_half_life
#         min_support      ≈ 5-10 x typical_change

#     Attributes:
#         half_life: Base half-life for exponential decay (in seconds). Controls how
#             quickly historical observations lose influence.
#         slow_half_life_factor: Multiplier for slow baseline tracking (typically 20).
#         slow_epsilon: Small value to prevent numerical issues in slow tracker.
#         slow_prune_threshold: Threshold below which slow tracker entries are pruned.
#         fast_half_life_factor: Multiplier for fast baseline tracking (typically 1.5).
#         fast_epsilon: Small value to prevent numerical issues in fast tracker.
#         fast_prune_threshold: Threshold below which fast tracker entries are pruned.
#         drift_half_life_factor: Multiplier for drift detection baseline (typically 30).
#         tau_enter: Threshold for entering drift state.
#         tau_exit: Threshold for exiting drift state.
#         adaptive_tau: Whether to use adaptive thresholds for drift detection.
#         n_enter: Number of consecutive detections needed to enter drift state.
#         n_exit: Number of consecutive stable readings needed to exit drift state.
#         short_term_error_half_life_factor: Multiplier for short-term error tracking
#             (typically 4, for 2-4 x base_half_life, enables quick reaction).
#         long_term_error_half_life_factor: Multiplier for long-term error tracking
#             (typically 40, for 20-50 x base_half_life, provides stable reference).
#         persistence_half_life_factor: Multiplier for state persistence tracking.
#         min_prune_interval_factor: Multiplier for minimum interval between prune
#             operations (typically 5-10 x base_half_life).
#         persistence_strength: Weight for persistence boost (0.0 to 1.0).
#         min_support_factor: Multiplier for minimum support threshold.
#     """

#     half_life: float = 3600.0
#     # # T = tipikus változási idő
#     # # | Tracker          | Half-life       | Miért             |
#     # # | ---------------- | --------------- | ----------------- |
#     # # | short_term_error | **~ 2-4 x T**   | gyors reakció     |
#     # # | long_term_error  | **~ 20-50 x T** | stabil referencia |
#     # # 5-10 x base_half_life
#     # #######
#     # # half_life        ≈ typical_change
#     # # fast_half_life   ≈ 1-2 x typical_change
#     # # slow_half_life   ≈ 10-30 x typical_change
#     # # drift_half_life  ≈ 1.5 x slow_half_life
#     # slow_half_life_factor: float = 20
#     # slow_epsilon: float = 1e-9
#     # slow_prune_threshold: float = 1e-6
#     # fast_half_life_factor: float = 1.5
#     # fast_epsilon: float = 1e-9
#     # fast_prune_threshold: float = 1e-6
#     # drift_half_life_factor: float = 30
#     # tau_enter: float = 0.1
#     # tau_exit: float = 0.05
#     # adaptive_tau: bool = True
#     # n_enter: int = 3
#     # n_exit: int = 5
#     # short_term_error_half_life_factor: float = 4
#     # long_term_error_half_life_factor: float = 40
#     # persistence_half_life_factor: float = 5.0
#     # # min_prune_interval ≈ 5-10 x base_half_life
#     # min_prune_interval_factor: float = 5.0
#     persistence_strength: float = 0.5
#     # # min_support ≈ 5-10 x typical_change
#     # min_support_factor: float = 7.5


class ForecasterEngine:
    """
    Time-aware forecasting engine for discrete state prediction.

    This engine combines hierarchical temporal statistics, drift detection,
    error tracking, and state persistence modeling to provide robust
    state predictions over time. It uses exponential decay to weight
    recent observations more heavily than older ones.

    The engine maintains:
        - Hierarchical state statistics with temporal context
        - Global drift monitoring for concept drift detection
        - Short-term and long-term error tracking
        - State persistence modeling
        - Adaptive hyperparameter control

    Example:
        >>> params = ForecasterEngineParameters(half_life=3600.0)
        >>> engine = ForecasterEngine(params)
        >>> engine.update(time_key, State.ON)
        >>> prediction = engine.predict(time_key)
    """

    def __init__(
        self: Self,
        parameters: ForecasterEngineRuntimeParameters,
    ) -> None:
        """
        Initializes the forecaster engine.

        Args:
            parameters: Runtime parameters for configuring the engine's behavior.
        """
        self._hyper_parameter_controller: Final = HyperParameterController(
            runtime_parameters=parameters.hyper_parameter_controller,
        )

        self._stats: Final = HierarchicalStateStats(
            HierarchicalStateStatsHyperParameters(
                hyper_parameters=self._hyper_parameter_controller.hyper_parameters,
            ),
            parameters.hierarchical_state_stats,
        )

        # Drift monitor detects concept drifts in GLOBAL distribution.
        self._drift_monitor: Final = DriftMonitor(
            DriftMonitorHyperParameters(
                hyper_parameters=self._hyper_parameter_controller.hyper_parameters,
            ),
            parameters.drift_monitor,
        )
        self._short_term_error_tracker: Final = OnlineErrorTracker(
            OnlineErrorTrackerHyperParameters(
                hyper_parameters=self._hyper_parameter_controller.hyper_parameters,
            ),
            parameters.short_term_error_tracker,
        )
        self._long_term_error_tracker: Final = OnlineErrorTracker(
            OnlineErrorTrackerHyperParameters(
                hyper_parameters=self._hyper_parameter_controller.hyper_parameters,
            ),
            parameters.long_term_error_tracker,
        )

        self._state_persistence_tracker: Final = StatePersistenceTracker(
            StatePersistenceTrackerHyperParameters(
                hyper_parameters=self._hyper_parameter_controller.hyper_parameters,
            ),
            parameters.state_persistence_tracker,
        )

        self._last_update_timestamp: float | None = None
        self._last_prune_timestamp: float | None = None

    def update(
        self: Self,
        key: TimeKey,
        state: State,
        timestamp: float | None = None,
    ) -> None:
        """
        Updates the forecaster with a new state observation.

        This method applies exponential decay to existing statistics, incorporates
        the new observation, updates drift detection, tracks prediction errors,
        and adjusts hyperparameters based on current conditions.

        Args:
            key: Temporal context key for the observation.
            state: The observed state value.
            timestamp: Unix timestamp of the observation. If None, uses current time.

        Raises:
            ValueError: If timestamp is earlier than the last update timestamp.
        """
        timestamp = (
            timestamp if timestamp is not None else datetime.now(tz=UTC).timestamp()
        )

        if self._last_update_timestamp is not None:
            duration = timestamp - self._last_update_timestamp

            if duration < 0:
                raise ValueError(
                    "Timestamp cannot be in the past, "
                    f"previous update was at {self._last_update_timestamp}, "
                    f"current timestamp is {timestamp}"
                )

            self._stats.apply_decay(self._get_decay_factor(duration))
        else:
            duration = 0.0

        self._stats.update(key, state, weight=duration)

        global_prediction = self._stats.predict(TimeKey.GLOBAL)

        if global_prediction:
            # After the first update we may not be able to get a global prediction.
            self._drift_monitor.update(global_prediction.distribution, timestamp)

        self._prune(timestamp)

        prediction = self._stats.predict(key)
        if prediction is not None:
            self._short_term_error_tracker.update(
                prediction.distribution, state, timestamp
            )
            self._long_term_error_tracker.update(
                prediction.distribution, state, timestamp
            )

        # Update state persistence tracker
        self._state_persistence_tracker.update(state, timestamp)

        self._last_update_timestamp = timestamp

        # TODO: ezt ki lehetne egy felsobb retegbe vinni? Igy akkor a hiperparameterek bentrol
        # nezve statikusak. Kérdés: itt van-e a helye a frift monitornak és az error trackernek?
        # lehet nem... ForecasterOrchestrator? (nem sokat tesz hozza, mert majdnem minden kikerül innen akkor...)
        self._hyper_parameter_controller.update(
            is_drifting=self._drift_monitor.is_drifting,
            short_term_error=self._short_term_error_tracker.mean,
            long_term_error=self._long_term_error_tracker.mean,
            entropy_confidence=(
                global_prediction.confidence.entropy_confidence
                if global_prediction
                else None
            ),
            fallback_depth=(
                global_prediction.confidence.depth if global_prediction else None
            ),
        )

    def predict(self: Self, key: TimeKey) -> PredictionResult | None:
        """
        Generates a state prediction for the given temporal context.

        Returns the base prediction without persistence adjustments.

        Args:
            key: Temporal context key for the prediction.

        Returns:
            PredictionResult containing the predicted distribution and contributions,
            or None if insufficient data is available.
        """
        return self._stats.predict(key)

    def predict_with_persistence(
        self,
        key: TimeKey,
        current_state: State | None = None,
        current_state_duration: float | None = None,
    ) -> PredictionResult | None:
        """
        Generates a state prediction with persistence adjustments.

        This method enhances the base prediction by applying a persistence boost
        to the current state based on how long it has persisted relative to its
        expected duration. The persistence effect follows a hazard-style decay.

        Three cases are handled:
        1. Explicit current state and duration provided → use those values
        2. No explicit state → use internal tracker's current state
        3. No persistence info available → return base prediction

        Args:
            key: Temporal context key for the prediction.
            current_state: Optional current state to apply persistence to.
            current_state_duration: Optional duration (seconds) the current state
                has been active.

        Returns:
            PredictionResult with persistence-adjusted distribution and contributions,
            or None if insufficient data is available.
        """
        prediction = self._stats.predict(key)
        if prediction is None:
            return None

        base_probs = prediction.distribution
        total_support = prediction.confidence.support

        adjusted = DistributionStats()

        # ------------------------------------------------------------------
        # CASE 1: explicit current state + duration provided
        # ------------------------------------------------------------------
        if current_state is not None and current_state_duration is not None:
            expected = self._state_persistence_tracker.expected_duration(current_state)

            # hazard-style decay
            ratio = current_state_duration / max(expected, 1e-6)
            persistence_boost = math.exp(-ratio)

            for state, prob in base_probs.items():
                weight = prob * total_support

                if state == current_state:
                    weight *= (
                        1.0
                        + self._hyper_parameter_controller.hyper_parameters.persistence_strength
                        * persistence_boost
                    )

                adjusted.update(state, weight)

            return PredictionResult(
                key=prediction.key,
                distribution_stats=adjusted,
                contributions=prediction.contributions,
            )

        # ------------------------------------------------------------------
        # CASE 2: no explicit current state → use internal tracker
        # ------------------------------------------------------------------
        internal_current = self._state_persistence_tracker.current_state

        if internal_current is not None:
            timestamp = self._last_update_timestamp
            if timestamp is not None:
                expected = self._state_persistence_tracker.expected_duration(
                    internal_current
                )
                current_duration = self._state_persistence_tracker.current_duration(
                    timestamp
                )

                ratio = current_duration / max(expected, 1e-6)
                persistence_boost = math.exp(-ratio)

                for state, prob in base_probs.items():
                    weight = prob * total_support

                    if state == internal_current:
                        weight *= (
                            1.0
                            + self._hyper_parameter_controller.hyper_parameters.persistence_strength
                            * persistence_boost
                        )

                    adjusted.update(state, weight)

                return PredictionResult(
                    key=prediction.key,
                    distribution_stats=adjusted,
                    contributions=prediction.contributions,
                )

        # ------------------------------------------------------------------
        # CASE 3: no persistence information → return base prediction
        # ------------------------------------------------------------------
        return prediction

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the forecaster engine to a dictionary for persistence.

        Returns:
            A dictionary representation of the forecaster engine, including
            hyperparameters, statistics, drift monitor state, error trackers,
            and state persistence tracker state.
        """
        return {
            "stats": self._stats.to_dict(),
            "drift_monitor": self._drift_monitor.to_dict(),
            "short_term_error_tracker": self._short_term_error_tracker.to_dict(),
            "long_term_error_tracker": self._long_term_error_tracker.to_dict(),
            "state_persistence_tracker": self._state_persistence_tracker.to_dict(),
            "hyper_parameter_controller": self._hyper_parameter_controller.to_dict(),
        }

    def _prune(self: Self, timestamp: float) -> None:
        """
        Prunes low-weight statistics to prevent memory bloat.

        Pruning only occurs if enabled and if the minimum interval has elapsed
        since the last prune operation.

        Args:
            timestamp: Current timestamp for checking prune interval.
        """
        if not self._hyper_parameter_controller.hyper_parameters.prune_enabled:
            return

        if self._last_prune_timestamp is None:
            self._last_prune_timestamp = timestamp
            return

        if (
            self._last_prune_timestamp
            + (
                self._hyper_parameter_controller.hyper_parameters.half_life
                * self._hyper_parameter_controller.hyper_parameters.min_prune_interval_factor
            )
        ) > timestamp:
            return

        self._stats.prune()  # TODO: min 20 keves lehet
        self._last_prune_timestamp = timestamp

    def _get_decay_factor(self: Self, duration: float) -> float:
        """
        Calculates exponential decay factor for a given time duration.

        Uses the formula: decay = 2^(-duration / half_life)

        Args:
            duration: Time duration in seconds.

        Returns:
            Decay factor between 0 and 1.
        """
        return 2 ** (
            -duration / self._hyper_parameter_controller.hyper_parameters.half_life
        )
