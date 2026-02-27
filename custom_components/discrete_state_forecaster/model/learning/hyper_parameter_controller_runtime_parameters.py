"""
Runtime parameters for adaptive hyper-parameter control.

This module defines the configuration for the HyperParameterController,
which dynamically adjusts model hyper-parameters in response to drift
and error signals.
"""

from dataclasses import dataclass


@dataclass
class AdaptationConfig:
    """
    Configuration for which hyper-parameters to adapt.

    Attributes:
        adapt_half_life: Whether to adapt the half-life parameter.
        adapt_persistence: Whether to adapt persistence strength.
        adapt_prune_interval: Whether to adapt pruning interval.

    """

    adapt_half_life: bool = True
    """Whether to adapt the half-life parameter based on model performance."""

    adapt_persistence: bool = True
    """Whether to adapt persistence strength based on model performance."""

    adapt_prune_interval: bool = True
    """Whether to adapt pruning interval based on model performance."""


@dataclass
class HyperParameterControllerRuntimeParameters:
    """
    Runtime parameters for hyper-parameter controller.

    Defines the baseline values and constraints for adaptive hyper-parameter
    control, including half-life bounds and adaptation configuration.
    """

    base_half_life: float
    """Initial half-life value (in seconds) for exponential decay weights."""

    adaptation_config: AdaptationConfig
    """Configuration for which hyper-parameters should be adapted."""

    base_state_inertia_strength: float
    """Strength of persistence modeling (0.0 = no persistence, 1.0 = full persistence)."""

    min_prune_interval_factor: float = 5.0
    """Minimum interval factor for pruning operations (relative to half-life)."""

    min_half_life: float = 60.0
    """Lower bound for half-life adaptation (in seconds)."""

    max_half_life: float = 3600.0 * 8760
    """Upper bound for half-life adaptation (in seconds, default: 1 year)."""

    background_decay_half_life_factor: float = 0.0
    """
    Multiplier for background (dormant-key) decay.

    ``0.0`` disables background decay completely (pure per-key
    observation-weighted decay, the default).  A positive value ``f`` causes
    *all* keys – including dormant ones – to receive a slow exponential decay
    whose effective half-life is ``f * base_half_life``.  For example,
    ``20.0`` means dormant keys (e.g. ``season=winter`` slots during summer)
    decay 20× slower than actively observed keys, preventing completely stale
    data from persisting indefinitely while still preserving long-term seasonal
    statistics across off-season periods.
    """
