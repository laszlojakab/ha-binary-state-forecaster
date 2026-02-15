from dataclasses import dataclass


@dataclass
class AdaptationConfig:
    adapt_half_life: bool = True
    adapt_persistence: bool = True
    adapt_prune_interval: bool = True


@dataclass
class HyperParameterControllerRuntimeParameters:
    base_half_life: float

    adaptation_config: AdaptationConfig

    base_persistence_strength: float
    """Strength of persistence modeling (0.0 = no persistence, 1.0 = full persistence)."""

    min_prune_interval_factor: float = 5.0

    min_half_life: float = 60.0
    max_half_life: float = 3600.0 * 48  # TODO: is this too low?
