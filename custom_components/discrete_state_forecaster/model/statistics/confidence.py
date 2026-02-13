from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    support: float
    depth: int
    max_probability: float
    entropy_confidence: float
