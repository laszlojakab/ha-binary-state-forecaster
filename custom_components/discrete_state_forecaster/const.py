"""Module of discrete state forecaster constants."""

import logging
from typing import Final

DOMAIN: Final = "discrete_state_forecaster"
LOGGER: Final = logging.getLogger(__package__)

# Configurations
CONF_TARGET_ENTITY_ID: Final = "target_entity_id"
CONF_ADVANCED_CONFIGURATION: Final = "advanced_configuration"
CONF_PRESET: Final = "preset"

# Indexer configuration
CONF_TIME_BUCKET_SIZE_IN_MINUTES: Final = "time_bucket_size_in_minutes"
CONF_USE_DAY_OF_WEEK: Final = "use_day_of_week"
CONF_USE_DAY_OF_WEEK_FEATURE: Final = "use_day_of_week_feature"
CONF_USE_MONTH_OF_YEAR_FEATURE: Final = "use_month_of_year_feature"
CONF_USE_MONTH_OF_YEAR: Final = "use_month_of_year"
CONF_USE_SEASON: Final = "use_season"
# CONF_CALENDAR_FEATURES: Final = "calendar_features"
# CONF_FORECASTER_FEATURES: Final = "forecaster_features"

# Adaptivity configuration
CONF_ENABLE_ADAPTIVE_HALF_LIFE: Final = "enable_adaptive_half_life"
CONF_ADAPTIVE_PRUNE_INTERVAL: Final = "adaptive_prune_interval"
CONF_ADAPTIVE_PERSISTENCE: Final = "adaptive_persistence"

# Learning
CONF_HALF_LIFE_HOURS: Final = "half_life_hours"
CONF_HALF_LIFE_HOURS_MIN: Final = 0.0
CONF_HALF_LIFE_HOURS_MAX: Final = 8760.0  # 1 year
CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR: Final = "short_term_error_half_life_factor"
CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR_MAX: Final = 100.0
CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR: Final = "long_term_error_half_life_factor"
CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR_MAX: Final = 100.0

# Persistence
CONF_BASE_STATE_INERTIA_STRENGTH: Final = "base_state_inertia_strength"
CONF_BASE_STATE_INERTIA_STRENGTH_MIN: Final = 0.0
CONF_BASE_STATE_INERTIA_STRENGTH_MAX: Final = 1.0
CONF_PERSISTENCE_HALF_LIFE_FACTOR: Final = "persistence_half_life_factor"
CONF_PERSISTENCE_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_PERSISTENCE_HALF_LIFE_FACTOR_MAX: Final = 100.0

# Drift
CONF_FAST_BASELINE_HALF_LIFE_FACTOR: Final = "fast_baseline_half_life_factor"
CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_FAST_BASELINE_HALF_LIFE_FACTOR_MAX: Final = 100.0
CONF_SLOW_BASELINE_HALF_LIFE_FACTOR: Final = "slow_baseline_half_life_factor"
CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_SLOW_BASELINE_HALF_LIFE_FACTOR_MAX: Final = 100.0
CONF_TAU_ENTER: Final = "tau_enter"
CONF_TAU_ENTER_MIN: Final = 0.0
CONF_TAU_ENTER_MAX: Final = 1.0
CONF_TAU_EXIT: Final = "tau_exit"
CONF_TAU_EXIT_MIN: Final = 0.0
CONF_TAU_EXIT_MAX: Final = 1.0
CONF_ADAPTIVE_TAU: Final = "adaptive_tau"

# Pruning
CONF_MIN_PRUNE_INTERVAL_FACTOR: Final = "min_prune_interval_factor"
CONF_MIN_PRUNE_INTERVAL_FACTOR_MIN: Final = 0.1
CONF_MIN_PRUNE_INTERVAL_FACTOR_MAX: Final = 100.0

# Background decay
CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR: Final = "background_decay_half_life_factor"
CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR_MIN: Final = 0.0
CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR_MAX: Final = 1000.0

# Default configuration values
DEFAULT_TIME_BUCKET_SIZE_IN_MINUTES: Final = "60"
DEFAULT_USE_DAY_OF_WEEK: Final = False
DEFAULT_USE_MONTH_OF_YEAR: Final = False
DEFAULT_USE_SEASON: Final = False
DEFAULT_HALF_LIFE_HOURS: Final = 168.0  # TODO: remove...
DEFAULT_PRESET: Final = "moderate"

PRESET_STABLE = "stable"
PRESET_MODERATE = "moderate"
PRESET_DYNAMIC = "dynamic"
PRESET_CUSTOM = "custom"

PRESETS = [PRESET_STABLE, PRESET_MODERATE, PRESET_DYNAMIC, PRESET_CUSTOM]

PRESET_CONFIGURATIONS = {
    PRESET_STABLE: {
        CONF_HALF_LIFE_HOURS: 72,
        CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR: 6.0,
        CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR: 60.0,
        CONF_BASE_STATE_INERTIA_STRENGTH: 0.7,
        CONF_PERSISTENCE_HALF_LIFE_FACTOR: 8,
        CONF_FAST_BASELINE_HALF_LIFE_FACTOR: 3.0,
        CONF_SLOW_BASELINE_HALF_LIFE_FACTOR: 30.0,
        CONF_TAU_ENTER: 0.15,
        CONF_TAU_EXIT: 0.08,
        CONF_ADAPTIVE_TAU: False,
        CONF_MIN_PRUNE_INTERVAL_FACTOR: 8.0,
        CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR: 0.0,
    },
    PRESET_MODERATE: {
        CONF_HALF_LIFE_HOURS: 48,
        CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR: 4.0,
        CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR: 40.0,
        CONF_BASE_STATE_INERTIA_STRENGTH: 0.5,
        CONF_PERSISTENCE_HALF_LIFE_FACTOR: 5,
        CONF_FAST_BASELINE_HALF_LIFE_FACTOR: 1.5,
        CONF_SLOW_BASELINE_HALF_LIFE_FACTOR: 20.0,
        CONF_TAU_ENTER: 0.1,
        CONF_TAU_EXIT: 0.05,
        CONF_ADAPTIVE_TAU: False,
        CONF_MIN_PRUNE_INTERVAL_FACTOR: 5.0,
        CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR: 0.0,
    },
    PRESET_DYNAMIC: {
        CONF_HALF_LIFE_HOURS: 24,
        CONF_SHORT_TERM_ERROR_HALF_LIFE_FACTOR: 2.5,
        CONF_LONG_TERM_ERROR_HALF_LIFE_FACTOR: 20,
        CONF_BASE_STATE_INERTIA_STRENGTH: 0.3,
        CONF_PERSISTENCE_HALF_LIFE_FACTOR: 3,
        CONF_FAST_BASELINE_HALF_LIFE_FACTOR: 0.8,
        CONF_SLOW_BASELINE_HALF_LIFE_FACTOR: 10.0,
        CONF_TAU_ENTER: 0.07,
        CONF_TAU_EXIT: 0.03,
        CONF_ADAPTIVE_TAU: True,
        CONF_MIN_PRUNE_INTERVAL_FACTOR: 3.0,
        CONF_BACKGROUND_DECAY_HALF_LIFE_FACTOR: 0.0,
    },
}

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

SUPPORTED_BUCKET_SIZES: Final = [
    "1",  # TODO: remove
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


STORING_TIME_PATTERN: Final = {
    "minute": 0,
    "second": 0,
    "hour": "*",
}
"""The pattern specifying the time at which to store the state of the discrete state forecaster."""
