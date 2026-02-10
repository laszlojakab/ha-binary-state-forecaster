from dataclasses import dataclass
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.statistics import PredictionResult
from custom_components.discrete_state_forecaster.model.temporal import (
    TimeIndexer,
)

from .forecaster_engine import (
    ForecasterEngine,
    ForecasterEngineParameters,
)
from .state import (
    State,
)


@dataclass(frozen=True)
class TimeAwareForecasterParameters(ForecasterEngineParameters):
    indexer: TimeIndexer


class TimeAwareForecaster:
    def __init__(self: Self, parameters: TimeAwareForecasterParameters) -> None:
        self._engine: Final = ForecasterEngine(parameters)
        self._indexer: Final = parameters.indexer

    def update(self: Self, timestamp: float, state: State) -> None:
        key = self._indexer.key(timestamp)
        self._engine.update(key, state, timestamp)

    def predict(self: Self, timestamp: float) -> PredictionResult | None:
        key = self._indexer.key(timestamp)
        return self._engine.predict(key)

    def predict_interval(
        self: Self,
        start_ts: float,
        end_ts: float,
        resolution: float,
        current_state: State | None = None,
        current_state_duration: float | None = None,
        simulate_state_path: bool = True,
    ) -> list[tuple[float, PredictionResult]]:
        """
        Forward simulation forecast.

        - Figyelembe veszi a TimeKey boundary-ket
        - Persistence-aware prediction-t használ
        - Frissíti a state duration-t minden step után
        - Opcionálisan szimulálja a jövőbeli state path-et (argmax)
        """
        ts = start_ts
        results: list[tuple[float, PredictionResult]] = []

        sim_state = current_state
        sim_duration = current_state_duration or 0.0

        while ts < end_ts:
            key = self._indexer.key(ts)

            next_boundary = self._indexer.next_boundary(ts)
            step_end = min(ts + resolution, next_boundary, end_ts)
            step_dt = step_end - ts

            prediction = self._engine.predict_with_persistence(
                key=key,
                current_state=sim_state,
                current_state_duration=sim_duration,
            )

            if prediction is None:
                ts = step_end
                continue

            results.append((ts, prediction))

            # ----- Forward simulation of state -----

            if simulate_state_path:
                dist = prediction.distribution.distribution()

                if dist:
                    # Argmax state
                    next_state = max(dist.items(), key=lambda x: x[1])[0]

                    if sim_state == next_state:
                        sim_duration += step_dt
                    else:
                        sim_state = next_state
                        sim_duration = step_dt
                else:
                    sim_state = None
                    sim_duration = 0.0
            # Ha nem szimulálunk state path-et,
            # csak növeljük a duration-t ha van current state
            elif sim_state is not None:
                sim_duration += step_dt

            ts = step_end

        return results
