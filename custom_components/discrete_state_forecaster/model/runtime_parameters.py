"""Runtime parameters for the model."""

from dataclasses import dataclass

@dataclass(frozen=True)
class RuntimeParameters:
    """
    Parameters for runtime configuring the model.

    This dataclass contains all parameters used to configure the behavior
    of the model, including decay rates, drift detection thresholds,
    and persistence modeling.
    """

    slow_half_life_factor: float = 20
    """Multiplier for slow baseline tracking (typically 20)."""

    slow_epsilon: float = 1e-9
    """Small value to prevent numerical issues in slow tracker."""

    slow_prune_threshold: float = 1e-6
    """Threshold below which slow tracker entries are pruned."""

    fast_half_life_factor: float = 1.5
    """Multiplier for fast baseline tracking (typically 1.5)."""

    fast_epsilon: float = 1e-9
    """Small value to prevent numerical issues in fast tracker."""

    fast_prune_threshold: float = 1e-6
    """Threshold below which fast tracker entries are pruned."""

    drift_half_life_factor: float = 30
    """Multiplier for drift detection baseline (typically 30)."""

    tau_enter: float = 0.1
    """Threshold for entering drift state."""

    tau_exit: float = 0.05
    """Threshold for exiting drift state."""

    adaptive_tau: bool = True
    """Whether to use adaptive thresholds for drift detection."""

    n_enter: int = 3
    """Number of consecutive detections needed to enter drift state."""

    n_exit: int = 5
    """Number of consecutive stable readings needed to exit drift state."""

    short_term_error_half_life_factor: float = 4
    """
    Multiplier for short-term error tracking
    (typically 4, for 2-4 x base_half_life, enables quick reaction).
    """

    long_term_error_half_life_factor: float = 40
    """
    Multiplier for long-term error tracking
    (typically 40, for 20-50 x base_half_life,provides stable reference).
    """

    persistence_half_life_factor: float = 5.0
    """Multiplier for state persistence tracking."""

    min_prune_interval_factor: float = 5.0
    """
    Multiplier for minimum interval between prune operations
    (typically 5-10 x base_half_life).
    """

    min_support_factor: float = 7.5
    """Multiplier for minimum support threshold."""
