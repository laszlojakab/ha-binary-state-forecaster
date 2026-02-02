"""
Prediction result container for state forecasting.

This module provides the Prediction dataclass which encapsulates the complete
result of a state prediction operation, including the predicted state, the
full probability distribution, and confidence metrics.

The Prediction class is the primary return type for forecasting operations,
providing all the information needed to understand and evaluate a prediction.
"""

from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.confidence import Confidence
from custom_components.discrete_state_forecaster.model.state import State
from custom_components.discrete_state_forecaster.model.time_indexers.time_key import (
    TimeKey,
)


@dataclass
class Prediction:
    """
    Complete prediction result for a state forecast.

    Encapsulates the predicted state, the full probability distribution over
    all possible states, and confidence metrics assessing the reliability of
    the prediction. This provides a comprehensive view of what the model
    predicts and how certain it is about that prediction.

    Attributes:
        state: The most likely state (highest probability), or None if no
            data exists for the queried time bucket. This is the primary
            prediction value.
        distribution: Dictionary mapping states to their probabilities (values
            between 0 and 1 that sum to 1). Provides the complete probability
            distribution. Empty dict if no data exists.
        confidence: Confidence metrics including max probability, entropy-based
            confidence, and total support time. Assesses prediction reliability.
            All metrics are 0 if no data exists.

    Example:
        ```
        >>> prediction = Prediction(
        ...     state="on",
        ...     distribution={"on": 0.75, "off": 0.25},
        ...     confidence=Confidence(
        ...         max_probability=0.75,
        ...         entropy_confidence=0.81,
        ...         support_time=3600.0
        ...     )
        ... )
        >>> print(f"Predicted: {prediction.state}")
        Predicted: on
        >>> print(f"Confidence: {prediction.confidence.max_probability:.2%}")
        Confidence: 75.00%
        ```
    """

    state: State | None
    distribution: dict[State, float]
    confidence: Confidence
    key: TimeKey | None = None
