from dataclasses import dataclass


@dataclass
class HyperParameterControllerRuntimeParameters:
    min_prune_interval_factor: float = 5.0
