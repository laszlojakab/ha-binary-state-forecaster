"""Module of discrete state forecaster constants."""

import logging
from typing import Final

DOMAIN: Final = "discrete_state_forecaster"
LOGGER: Final = logging.getLogger(__package__)

CONF_TARGET_ENTITY_ID: Final = "target_entity_id"
CONF_STABILITY: Final = "stability"
CONF_TIME_BUCKET_SIZE: Final = "time_bucket_size"
CONF_USE_DAY_OF_WEEK_FEATURE: Final = "use_day_of_week_feature"
CONF_CALENDAR_FEATURES: Final = "calendar_features"
CONF_FORECASTER_FEATURES: Final = "forecaster_features"
CONF_DECAY_SECONDS: Final = "decay_seconds"


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
CONF_TIME_BLOCK_PERIOD: Final = "time_block_period"


CONF_BINARY_SENSOR: Final = "binary_sensor"
CONF_FILTER_BINARY_SENSOR: Final = "filter_binary_sensor"
DAY_OF_WEEK_FEATURE: Final = "day_of_week"

ATTR_CURRENT_TIME_BLOCK_STATE: Final = "current_time_block_state"
ATTR_CURRENT_STATE: Final = "current_state"
ATTR_CURRENT_TIME_BLOCK: Final = "current_time_block"
ATTR_PROBABILITIES: Final = "probabilities"
ATTR_PROBABILITY: Final = "probability"


TIME_BLOCK_PERIOD_IN_MINUTES: Final = 5
"""The interval at which to update data from the discrete state forecaster."""

STORING_TIME_PATTERN: Final = {
    "minute": 0,
    "second": 0,
    "hour": "*",
}
"""The pattern specifying the time at which to store the state of the discrete state forecaster."""
