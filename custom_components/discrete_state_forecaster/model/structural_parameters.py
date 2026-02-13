"""Module of structural parameters for TimeAwareForecaster."""

from dataclasses import dataclass

from custom_components.discrete_state_forecaster.model.temporal.time_indexer import (
    TimeIndexer,
)


@dataclass(frozen=True)
class StructuralParameters:
    """
    Structural parameters for the model.

    Changing these parameters would alter the fundamental structure of the forecaster and
    require retraining from scratch. They include the time indexer, which defines how timestamps are
    converted to temporal keys and thus how patterns are learned across different temporal contexts.
    """

    indexer: TimeIndexer
    """TimeIndexer for converting timestamps to temporal keys."""
