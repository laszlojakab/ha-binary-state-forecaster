"""Confidence metrics for hierarchical state statistics predictions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    """Confidence metrics for hierarchical state statistics predictions."""

    support: float
    """
    Total support (sum of state weights) at the level that produced the prediction.

    Higher support indicates more data backing the prediction, which generally
    leads to higher confidence.

    Lower support means the prediction is based on less data and is therefore less
    reliable.
    """

    depth: int
    """
    Depth of the level that produced the prediction (0 for specific key, 1 for parent, etc.).

    Depth 0 (specific key) is the most confident, while higher depths indicate more
    general patterns were used, which may be less accurate for the specific prediction.
    """

    max_probability: float
    """
    Maximum probability in the predicted distribution at the level that produced the prediction.

    Higher max_probability indicates a more confident prediction, as it shows a stronger preference
    for a particular state.
    """

    entropy_confidence: float
    """
    Confidence derived from the entropy of the predicted distribution.

    Lower entropy (more peaked distribution) indicates higher confidence, while higher entropy
    (more uniform distribution) indicates lower confidence.

    Higher entropy confidence means the model is more certain about the prediction,
    while lower entropy confidence indicates more uncertainty and
    less reliability in the prediction.
    """
