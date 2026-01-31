from datetime import datetime, timedelta
from custom_components.discrete_state_forecaster.model.time_aware_forecaster import TimeAwareForecaster
from custom_components.discrete_state_forecaster.model.time_indexers.composite_indexer import CompositeIndexer
from custom_components.discrete_state_forecaster.model.time_indexers.time_of_day_indexer import TimeOfDayIndexer

# 60 minutes = 1 hour per bucket
indexer = CompositeIndexer([TimeOfDayIndexer(60)])  # 60 minutes per bucket
forecaster = TimeAwareForecaster(indexer)

# Simulate 7 days of data (like the test)
for day in range(7):
    base_date = datetime(2024, 1, 1) + timedelta(days=day)
    
    # Morning: off (0:00 - 8:00)
    forecaster.update_interval(
        base_date.replace(hour=0, minute=0),
        base_date.replace(hour=8, minute=0),
        'off',
    )
    
    # Work hours: on (8:00 - 17:00)
    forecaster.update_interval(
        base_date.replace(hour=8, minute=0),
        base_date.replace(hour=17, minute=0),
        'on',
    )
    
    # Evening: off (17:00 - 23:59)
    forecaster.update_interval(
        base_date.replace(hour=17, minute=0),
        base_date.replace(hour=23, minute=59),
        'off',
    )

# Predict for next day at various hours
for hour in [6, 12, 20]:
    pred = forecaster.predict(datetime(2024, 1, 8, hour, 0))
    key = indexer.key(datetime(2024, 1, 8, hour, 0))
    print(f'Hour {hour:2d}: key={key}, state={pred.state}, distribution={pred.distribution}')
    print(f'          confidence: {pred.confidence}')
    print()
