"""
Confidence metrics for state predictions.

This module provides the Confidence dataclass which encapsulates multiple
confidence metrics for evaluating the reliability of state predictions.
These metrics help assess how trustworthy a prediction is based on
probability distribution characteristics and historical data availability.
"""

from dataclasses import dataclass


@dataclass
class Confidence:
    """
    Confidence metrics for state prediction quality.

    Aggregates multiple confidence indicators to provide a comprehensive
    assessment of prediction reliability. Each metric captures a different
    aspect of confidence:
    - Probability-based confidence (how dominant is the predicted state)
    - Entropy-based confidence (how certain vs. uncertain is the distribution)
    - Support-based confidence (how much historical data backs the prediction)

    These metrics can be combined or used individually to determine whether
    a prediction should be trusted or requires more data collection.

    Attributes:
        max_probability: The highest probability among all possible states
            (0.0 to 1.0). Higher values indicate a dominant predicted state.
            For example, 0.95 means the most likely state has 95% probability.
        entropy_confidence: Confidence derived from probability distribution
            entropy (0.0 to 1.0). Higher values indicate lower entropy (more
            certainty). A value near 1.0 means the distribution is concentrated
            on one state, while near 0.0 means uniform distribution across states.
        support_time: Total time (in seconds) of historical observations used
            for the prediction. Higher values indicate more historical data
            supporting the prediction, generally increasing reliability.

    Example:
        ```
        >>> # High confidence prediction
        >>> conf = Confidence(
        ...     max_probability=0.95,
        ...     entropy_confidence=0.90,
        ...     support_time=3600.0
        ... )
        >>> # Low confidence (uncertain, limited data)
        >>> conf = Confidence(
        ...     max_probability=0.55,
        ...     entropy_confidence=0.40,
        ...     support_time=120.0
        ... )
        ```
    """

    max_probability: float
    entropy_confidence: float
    support_time: float
    depth: int
