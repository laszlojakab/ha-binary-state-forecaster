"""Module of binary state forecaster constants."""

import logging
from typing import Final

DOMAIN: Final = "binary_state_forecaster"

CONF_BINARY_SENSOR: Final = "binary_sensor"
CONF_CALENDAR_FEATURES: Final = "calendar_features"
CONF_FORECASTER_FEATURES: Final = "forecaster_features"
CONF_USE_DAY_OF_WEEK_FEATURE: Final = "use_day_of_week_feature"
CONF_FADING: Final = "fading"
CONF_THRESHOLD: Final = "threshold"
CONF_PERIOD: Final = "period"
CONF_TIME_BLOCK_PERIOD: Final = "time_block_period"
CONF_FILTER_BINARY_SENSOR: Final = "filter_binary_sensor"
DAY_OF_WEEK_FEATURE: Final = "day_of_week"

ATTR_CURRENT_TIME_BLOCK_STATE: Final = "current_time_block_state"
ATTR_CURRENT_STATE: Final = "current_state"
ATTR_CURRENT_TIME_BLOCK: Final = "current_time_block"
ATTR_PROBABILITIES: Final = "probabilities"
ATTR_PROBABILITY: Final = "probability"

LOGGER: Final = logging.getLogger(__package__)

TIME_BLOCK_PERIOD_IN_MINUTES: Final = 5
"""The interval at which to update data from the binary state forecaster."""

STORING_TIME_PATTERN: Final = {
    "minute": 0,
    "second": 0,
    "hour": "*",
}
"""The pattern specifying the time at which to store the state of the binary state forecaster."""

# TRACKED_FEATURES_BY_DOMAIN: Final = {
#     "binary_sensor": ["on", "off"],
#     "valve": ["open", "closed"],
#     "input_boolean": ["on", "off"],
#     "switch": ["on", "off"],
#     "cover": ["open", "closed"],
#     "lock": ["unlocked", "locked"],
#     "light": ["on", "off"],
#     "calendar": ["on", "off"],
#     "alarm_control_panel": ["armed", "disarmed"],
#     "automation": ["on", "off"],
#     "fan": ["on", "off"],
#     "media_player": ["on", "off", "playing", "paused"],
#     "climate": ["heat", "cool", "heat_cool", "off", "auto", "dry", "fan_only"],
#     "climate.hvac_action": ["heating", "cooling", "idle"],
#     "device_tracker": ["home", "not_home"],
#     "timer": ["active", "idle"],
#     "update": ["on", "off"],
#     "person": ["home", "not_home"],
# }

# STATE_MAPPING_BY_DOMAIN: Final = {
#     "binary_sensor": {
#         "on": "on",
#         "off": "off",
#     },
#     "valve": {
#         "open": "on",
#         "closed": "off",
#     },
#     "input_boolean": {
#         "on": "on",
#         "off": "off",
#     },
#     "switch": {
#         "on": "on",
#         "off": "off",
#     },
#     "cover": {
#         "open": "on",
#         "closed": "off",
#     },
#     "lock": {
#         "unlocked": "on",
#         "locked": "off",
#     },
#     "light": {
#         "on": "on",
#         "off": "off",
#     },
#     "calendar": {
#         "on": "on",
#         "off": "off",
#     },
#     "alarm_control_panel": {
#         "armed": "on",
#         "disarmed": "off",
#     },
#     "automation": {
#         "on": "on",
#         "off": "off",
#     },
#     "fan": {
#         "on": "on",
#         "off": "off",
#     },
#     "media_player": {
#         "on": "on",
#         "playing": "on",
#         "off": "off",
#         "paused": "off",
#     },
#     "device_tracker": {
#         "home": "on",
#         "not_home": "off",
#     },
#     "timer": {
#         "active": "on",
#         "idle": "off",
#     },
#     "update": {
#         "on": "on",
#         "off": "off",
#     },
# }
