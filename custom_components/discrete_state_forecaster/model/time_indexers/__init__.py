"""
Time indexers for temporal pattern analysis.

This package provides various indexing strategies for partitioning time into
discrete buckets. Time indexers enable efficient modeling of temporal patterns
by grouping timestamps based on different time dimensions such as time of day,
day of week, month, or combinations thereof.

Available indexers:
    - TimeOfDayIndexer: Buckets by time of day (e.g., hourly, 30-minute intervals)
    - DayOfWeekIndexer: Buckets by day of week (Monday-Sunday)
    - MonthIndexer: Buckets by calendar month (January-December)
    - CompositeIndexer: Combines multiple indexers for multi-dimensional keys
    - TimeIndexer: Protocol defining the indexer interface
"""

from .composite_indexer import CompositeIndexer
from .day_of_week_indexer import DayOfWeekIndexer
from .month_indexer import MonthIndexer
from .time_indexer import TimeIndexer
from .time_key import TimeKey
from .time_of_day_indexer import TimeOfDayIndexer

__all__ = [
    "CompositeIndexer",
    "DayOfWeekIndexer",
    "MonthIndexer",
    "TimeIndexer",
    "TimeKey",
    "TimeOfDayIndexer",
]
