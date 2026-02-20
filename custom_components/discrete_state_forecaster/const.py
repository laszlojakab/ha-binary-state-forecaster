"""Module of discrete state forecaster constants."""

import logging
from typing import Final

DOMAIN: Final = "discrete_state_forecaster"
LOGGER: Final = logging.getLogger(__package__)

CONF_TARGET_ENTITY_ID: Final = "target_entity_id"
CONF_STABILITY: Final = "stability"
CONF_TIME_BUCKET_SIZE_IN_MINUTES: Final = "time_bucket_size_in_minutes"
CONF_USE_DAY_OF_WEEK_FEATURE: Final = "use_day_of_week_feature"
CONF_USE_MONTH_OF_YEAR_FEATURE: Final = "use_month_of_year_feature"
CONF_CALENDAR_FEATURES: Final = "calendar_features"
CONF_FORECASTER_FEATURES: Final = "forecaster_features"
CONF_DECAY_SECONDS: Final = "decay_seconds"

# Indexer configuration
CONF_USE_DAY_OF_WEEK: Final = "use_day_of_week"
CONF_USE_MONTH_OF_YEAR: Final = "use_month_of_year"
CONF_USE_SEASON: Final = "use_season"

# Prediction configuration
# CONF_STATE_PERSISTENCE_FACTOR: Final = "state_persistence_factor"
# CONF_ADAPTIVE_PERSISTENCE: Final = "adaptive_persistence"
CONF_HALF_LIFE_HOURS: Final = "half_life_hours"

# Default configuration values
DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES: Final = "60"
DEFAULT_USE_DAY_OF_WEEK: Final = True
DEFAULT_USE_MONTH_OF_YEAR: Final = False
DEFAULT_USE_SEASON: Final = False
# DEFAULT_STATE_PERSISTENCE_FACTOR: Final = 0.3
# DEFAULT_ADAPTIVE_PERSISTENCE: Final = True
DEFAULT_HALF_LIFE_HOURS: Final = 0.0

SUPPORTED_TARGET_DOMAINS: Final = [
    "alarm_control_panel",
    "automation",
    "binary_sensor",
    "calendar",
    "climate",
    "cover",
    "device_tracker",
    "fan",
    "input_boolean",
    "light",
    "lock",
    "media_player",
    "person",
    "switch",
    "timer",
    "update",
    "valve",
]

SUPPORTED_STABILITY_OPTIONS: Final = [
    "stable",
    "semi_stable",
    "quick_changing",
]

SUPPORTED_BUCKET_SIZES: Final = [
    "1",
    "5",
    "10",
    "15",
    "30",
    "60",
    "120",
    "180",
    "240",
    "360",
    "720",
    "1440",
]

######## OLD
ATTR_CURRENT_TIME_BLOCK_STATE: Final = "current_time_block_state"
ATTR_CURRENT_STATE: Final = "current_state"
ATTR_CURRENT_TIME_BLOCK: Final = "current_time_block"
ATTR_PROBABILITIES: Final = "probabilities"
ATTR_PROBABILITY: Final = "probability"


STORING_TIME_PATTERN: Final = {
    "minute": 0,
    "second": 0,
    "hour": "*",
}
"""The pattern specifying the time at which to store the state of the discrete state forecaster."""
