"""Model for discrete conditional forecasting model."""

import math
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from itertools import combinations
from typing import Self

FeatureName = str
FeatureLabel = Hashable
Duration = float
Probability = float
Score = float

TargetLabel = Hashable
FeatureLabels = dict[FeatureName, FeatureLabel]

FeatureKey = tuple[tuple[FeatureName, FeatureLabel], ...]


@dataclass
class DriftState:
    """
    Tracks concept drift state using exponential moving averages.

    Concept drift is detected by monitoring changes in prediction characteristics:
    - Increasing entropy (predictions become less certain)
    - Decreasing max probability (top prediction becomes less confident)
    - Decreasing support (less training data available)
    """

    entropy_ema: float = 0.0
    """Exponential moving average of normalized entropy."""

    maxp_ema: float = 0.0
    """Exponential moving average of maximum probability."""

    support_ema: float = 0.0
    """Exponential moving average of normalized support."""

    drift_score: float = 0.0
    """Current drift score in range [0, 1]. Higher values indicate stronger drift."""

    last_t: float = 0.0
    """Last update time in model time units."""


@dataclass
class DriftWeights:
    """
    Weights for combining drift detection signals.

    Each weight determines the influence of a specific drift indicator.
    Weights must be in [0, 1] and sum to ≤ 1.0.
    """

    entropy: float = 0.4
    """Weight for entropy increase signal (predictions becoming less certain)."""

    max_p: float = 0.4
    """Weight for max probability decrease signal (top prediction less confident)."""

    support: float = 0.2
    """Weight for support decrease signal (less training data available)."""

    def __post_init__(self) -> None:
        """Validate weights are in valid ranges."""
        if not 0 <= self.entropy <= 1:
            raise ValueError(f"entropy weight must be in [0, 1], got {self.entropy}")
        if not 0 <= self.max_p <= 1:
            raise ValueError(f"max_p weight must be in [0, 1], got {self.max_p}")
        if not 0 <= self.support <= 1:
            raise ValueError(f"support weight must be in [0, 1], got {self.support}")
        total = self.entropy + self.max_p + self.support
        if total > 1.0:
            raise ValueError(f"drift weights must sum to <= 1.0, got {total}")


@dataclass
class ConfidenceWeights:
    """
    Weights for combining confidence signals.

    Each weight determines the influence of a specific confidence indicator.
    Weights must be in [0, 1] and sum to ≤ 1.0.
    """

    entropy: float = 0.5
    """Weight for entropy-based confidence (prediction certainty)."""

    max_p: float = 0.3
    """Weight for maximum probability confidence."""

    support: float = 0.2
    """Weight for support-based confidence (data availability)."""

    def __post_init__(self) -> None:
        """Validate weights are in valid ranges."""
        if not 0 <= self.entropy <= 1:
            raise ValueError(f"entropy weight must be in [0, 1], got {self.entropy}")
        if not 0 <= self.max_p <= 1:
            raise ValueError(f"max_p weight must be in [0, 1], got {self.max_p}")
        if not 0 <= self.support <= 1:
            raise ValueError(f"support weight must be in [0, 1], got {self.support}")
        total = self.entropy + self.max_p + self.support
        if total > 1.0:
            raise ValueError(f"confidence weights must sum to <= 1.0, got {total}")


@dataclass
class FeatureState:
    """
    State for a given feature subset.

    Tracks the accumulated duration for each target label observed with
    this specific combination of features, along with the last update time
    for time-based decay calculations.
    """

    y: dict[TargetLabel, Duration]
    """Accumulated duration for each target label."""

    last_t: float
    """Last update time in model time units."""


@dataclass
class Confidence:
    """Confidence metrics for predictions."""

    max_probability: float
    """The maximum predicted probability for the target."""

    entropy_confidence: float
    """Confidence based on entropy (0.0 to 1.0)."""

    support_time: float
    """Total support time for the prediction. Higher means more data."""

    used_features: dict[FeatureName, FeatureLabel]
    """Features used for the prediction."""


class DiscreteConditionalModel:
    """
    Explainable discrete conditional model with:
    - duration-weighted learning
    - time-based (half-life) decay using internal model time
    - hierarchical feature subset selection (falls back to simpler models)
    - adaptive feature importance
    - adaptive feature interaction learning
    """

    def __init__(
        self: Self,
        alpha: float = 1.0,
        decay: float = 3600.0,  # half-life in same unit as duration
        priority_decay: float = 0.98,
        interaction_decay: float = 0.98,
        max_interactions: int = 100,
        min_interaction_support: float = 1.0,
        min_interaction_score: float = 0.05,
        drift_weights: DriftWeights | None = None,
        confidence_weights: ConfidenceWeights | None = None,
        support_tau: float = 3600.0,
        min_step_confidence: float = 0.0,
        _internal_state: dict | None = None,
    ) -> None:
        """
        Initializes the model.

        Args:
            alpha: Laplace smoothing parameter for probability estimates (must be > 0).
                   Higher values make predictions more uniform.
            decay: Half-life for temporal decay in same units as duration (must be >= 0).
                   After this time, old observations have half their original weight.
                   Set to 0 to disable decay.
            priority_decay: Exponential moving average factor for feature importance (0.0 to 1.0).
                           Higher values give more weight to historical scores.
            interaction_decay: Exponential moving average factor for interaction
                              scores (0.0 to 1.0). Higher values give more weight
                              to historical scores.
            max_interactions: Maximum number of feature interactions to keep (must be > 0).
                             When exceeded, only top-K interactions by absolute score are retained.
            min_interaction_support: Minimum accumulated duration support required to keep
                                    an interaction (must be >= 0). Interactions with less
                                    support are pruned as noise.
            min_interaction_score: Minimum absolute normalized score (0.0 to 1.0) required
                                  to keep an interaction. Weak interactions below this
                                  threshold are pruned.
            drift_weights: Weights for combining drift signals.
                          If None, uses default: DriftWeights(entropy=0.4, max_p=0.4, support=0.2).
                          Higher weights give more influence to that signal.
            confidence_weights: Weights for combining confidence signals
                               (entropy, max_p, support).
                               If None, uses default: ConfidenceWeights(
                                   entropy=0.5, max_p=0.3, support=0.2
                               ).
            support_tau: Time scale for support confidence saturation (must be > 0).
                        Higher values make confidence increase more slowly with support time.
            min_step_confidence: Minimum rolling confidence threshold for forecast early stopping.
                                Must be in [0, 1]. Set to 0.0 to disable early stopping.
            _internal_state: Internal state dict for deserialization. Not for public use.

        Example:
            ```python
            # For sensor data updated every minute with 1-hour decay
            model = DiscreteConditionalModel(
                alpha=1.0,
                decay=3600.0,  # 1 hour in seconds
            )
            ```

        Note:
            max_depth is automatically adjusted based on drift detection,
            prediction confidence, and the number of available features.
            See _adapt_max_depth() for details.

        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if decay < 0:
            raise ValueError(f"decay must be non-negative, got {decay}")
        if not 0 <= priority_decay <= 1:
            raise ValueError(f"priority_decay must be in [0, 1], got {priority_decay}")
        if not 0 <= interaction_decay <= 1:
            raise ValueError(
                f"interaction_decay must be in [0, 1], got {interaction_decay}"
            )
        if max_interactions < 1:
            raise ValueError(
                f"max_interactions must be at least 1, got {max_interactions}"
            )
        if min_interaction_support < 0:
            raise ValueError(
                f"min_interaction_support must be non-negative, got {min_interaction_support}"
            )
        if not 0 <= min_interaction_score <= 1:
            raise ValueError(
                f"min_interaction_score must be in [0, 1], got {min_interaction_score}"
            )
        if not 0 <= min_step_confidence <= 1:
            raise ValueError(
                f"min_step_confidence must be in [0, 1], got {min_step_confidence}"
            )
        if support_tau <= 0:
            raise ValueError(f"support_tau must be positive, got {support_tau}")

        # Validate and set drift_weights
        if drift_weights is None:
            drift_weights = DriftWeights()

        # Validate and set confidence_weights
        if confidence_weights is None:
            confidence_weights = ConfidenceWeights()

        self.alpha = alpha
        """Smoothing parameter for probability estimates."""

        self.decay = decay
        """Half-life for temporal decay (same unit as duration)."""

        self.priority_decay = priority_decay
        """Decay factor for feature importance scores."""

        self.interaction_decay = interaction_decay
        """Decay factor for interaction scores."""

        self.max_interactions = max_interactions
        """Maximum number of feature interactions to keep."""

        self.min_interaction_support = min_interaction_support
        """Minimum support required to keep an interaction."""

        self.min_interaction_score = min_interaction_score
        """Minimum absolute normalized score to keep an interaction."""

        self.drift_weights: DriftWeights = drift_weights
        """Weights for combining drift detection signals."""

        self.confidence_weights: ConfidenceWeights = confidence_weights
        """Weights for combining confidence signals."""

        self.support_tau = support_tau
        """Time scale for support confidence saturation."""

        self.min_step_confidence = min_step_confidence
        """Optional early-stop threshold for rolling confidence."""

        # Initialize or restore internal state
        if _internal_state is not None:
            self._restore_state(_internal_state)
        else:
            self._initialize_fresh_state()

    def _restore_state(self, state: dict) -> None:
        """Restore model internal state from dictionary."""
        self._max_depth: int | None = state["_max_depth"]
        self._t: float = state["_t"]
        self._last_duration: Duration = state["_last_duration"]
        self._y_domain: set[TargetLabel] = set(state["_y_domain"])

        # Restore feature scores
        self._feature_scores: dict[FeatureName, Score] = defaultdict(float)
        for k, v in state["_feature_scores"].items():
            self._feature_scores[k] = v

        # Restore interaction scores
        self._interaction_scores: dict[tuple[FeatureName, FeatureName], Score] = (
            defaultdict(float)
        )
        for k, v in state["_interaction_scores"].items():
            f1, f2 = k.split("|")
            self._interaction_scores[(f1, f2)] = v

        # Restore interaction support
        self._interaction_support: dict[tuple[FeatureName, FeatureName], Duration] = (
            defaultdict(float)
        )
        for k, v in state["_interaction_support"].items():
            f1, f2 = k.split("|")
            self._interaction_support[(f1, f2)] = v

        # Restore states
        self._states: dict[FeatureKey, FeatureState] = {}
        for key_str, state_data in state["_states"].items():
            key = (
                tuple(tuple(pair.split("=")) for pair in key_str.split("|"))
                if key_str
                else ()
            )
            self._states[key] = FeatureState(
                y=dict(state_data["y"]),
                last_t=state_data["last_t"],
            )

        # Restore drift state
        drift_data = state["_drift"]
        self._drift = DriftState(
            entropy_ema=drift_data["entropy_ema"],
            maxp_ema=drift_data["maxp_ema"],
            support_ema=drift_data["support_ema"],
            drift_score=drift_data["drift_score"],
            last_t=drift_data["last_t"],
        )

    def _initialize_fresh_state(self) -> None:
        """Initialize fresh model state."""
        self._max_depth: int | None = None
        """Maximum feature subset size to consider (dynamically adjusted)."""

        self._t: float = 0.0
        """Internal model time (monotonic, user-driven, by updates)"""

        self._states: dict[FeatureKey, FeatureState] = {}
        """Mapping of feature subsets to their states."""

        self._y_domain: set[TargetLabel] = set()
        """Set of observed target labels."""

        self._feature_scores: dict[FeatureName, Score] = defaultdict(float)
        """Importance scores for individual features."""

        self._interaction_scores: dict[tuple[FeatureName, FeatureName], Score] = (
            defaultdict(float)
        )
        """Importance scores for feature interactions."""

        self._last_duration: Duration = 1.0
        """Duration of the last observation, used for time-aware EMA updates.

        Longer durations cause faster adaptation of feature and interaction scores,
        compensating for infrequent updates during concept drift.
        """

        self._interaction_support = defaultdict(float)
        """Accumulated duration support for interactions."""

        self._drift = DriftState()
        """Internal state for concept drift detection."""

    def update(
        self: Self, features: FeatureLabels, y: TargetLabel, duration: Duration = 1.0
    ) -> None:
        """
        Updates the model with a new observation.

        Example:
          The target (forecasted variable) is the weather condition (sunny, rainy, cloudy).
          The observation is that the weather was "sunny" when the temperature was "high"
          and humidity was "low" for a duration of 180 seconds:

        ```
          model.update(
            features={"temp": "high", "humidity": "low"},
            y="sunny",
            duration=180.0
          )
        ```

        Args:
          features: Feature values for the observation.
          y: Target label for the observation.
          duration: Duration (weight) of this observation. Longer durations have more influence.
        """
        if duration <= 0:
            return

        self._last_duration = duration

        # We update the internal time
        self._t += duration

        # We add the target to the domain
        self._y_domain.add(y)

        updated = False

        # We get all available subsets based on the given features.
        for key in self._feature_subsets(features):
            state = self._get_state(key)
            # We update the duration for the target label
            state.y[y] += duration
            updated = True

        # If no subsets were found, we update the global state
        if not updated:
            state = self._get_state(())
            state.y[y] += duration

        # Finally we also update feature importance scores and interaction scores
        self._update_feature_scores(features)
        self._update_interaction_scores(features, duration)
        self._prune_interactions()

    def distribution_with_key(
        self: Self, features: FeatureLabels
    ) -> tuple[dict[TargetLabel, Probability], FeatureKey]:
        """
        Gets the predicted distribution along with the feature key used.

        Args:
          features: Feature values to evaluate.

        Returns:
          A tuple of the predicted distribution and the feature key used.
        """
        for key in self._feature_subsets(features):
            if key in self._states:
                return self._distribution_for_key(key), key
        return self._distribution_for_key(()), ()

    def distribution(
        self: Self, features: FeatureLabels
    ) -> dict[TargetLabel, Probability]:
        """
        Gets the predicted distribution for the given features.

        Args:
          features: Feature values to evaluate.

        Returns:
          A dictionary mapping target values to their predicted probabilities.
          Probabilities sum to 1.0.

        Example:
          ```python
          dist = model.distribution({"temp": "high", "humidity": "low"})
          # Returns: {"sunny": 0.7, "cloudy": 0.2, "rainy": 0.1}
          ```
        """
        dist, _ = self.distribution_with_key(features)
        return dist

    def predict(self: Self, features: FeatureLabels) -> TargetLabel | None:
        """
        Gets the predicted target for the given features.

        Args:
          features: Feature values to evaluate.

        Returns:
          The predicted target value with the highest probability,
          or `None` if no prediction can be made (empty model).

        Example:
          ```python
          prediction = model.predict({"temp": "high", "humidity": "low"})
          # Returns: "sunny" (the most likely weather condition)
          ```
        """
        dist = self.distribution(features)
        return max(dist, key=dist.get) if dist else None

    def forecast(
        self: Self,
        features: FeatureLabels,
        horizon: int,
        advance_time_features: Callable[[FeatureLabels, int], FeatureLabels],
        return_distributions: bool = False,
    ) -> list[dict]:
        """
        Produces rolling forecasts with confidence up to a given horizon.

        Args:
            features: Initial feature values to start forecasting from.
            horizon: Number of steps ahead to forecast.
            advance_time_features: Callback that advances time-dependent features.
                                  Takes (current_features, step_number) and returns
                                  updated features for that step.
            return_distributions: If True, includes full probability distributions
                                 in each forecast step.

        Returns:
            List of forecast dictionaries, one per step, each containing:
            - step (int): Step number (1 to horizon)
            - prediction: Most likely target label
            - step_confidence (float): Confidence for this step [0, 1]
            - rolling_confidence (float): Cumulative confidence [0, 1]
            - used_features (dict): Features used for this prediction
            - distribution (dict, optional): Full probability distribution if requested

            Returns empty list if model has no data.
            May return fewer than `horizon` steps if confidence drops below
            `min_step_confidence`.

        Example:
            ```python
            def advance_features(feats, step):
                # Advance time bucket by step
                new_feats = feats.copy()
                new_feats['time_bucket'] = (feats['time_bucket'] + step) % 288
                return new_feats

            forecasts = model.forecast(
                features={'temp': 'high', 'time_bucket': 100},
                horizon=10,
                advance_time_features=advance_features,
                return_distributions=True
            )
            ```
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        results = []

        features_t = features.copy()
        rolling_conf = 1.0

        for step in range(1, horizon + 1):
            dist = self.distribution(features_t)
            if not dist:
                break

            y_hat = max(dist, key=dist.get)
            conf = self.confidence(features_t, without_model_update=True)

            step_conf = self._step_confidence(conf)
            rolling_conf *= step_conf

            row = {
                "step": step,
                "prediction": y_hat,
                "step_confidence": step_conf,
                "rolling_confidence": rolling_conf,
                "used_features": conf.used_features,
            }

            if return_distributions:
                row["distribution"] = dist

            results.append(row)

            # Early stop if forecast becomes unreliable
            if rolling_conf < self.min_step_confidence:
                break

            features_t = advance_time_features(features_t, step)

        return results

    def confidence(
        self: Self, features: FeatureLabels, without_model_update: bool = False
    ) -> Confidence:
        """
        Gets confidence metrics for the given features.

        Args:
          features: Feature values to evaluate.
          without_model_update: If True, skip updating drift state and max_depth.
                               Use this for forecast simulations to avoid side effects.

        Returns:
          A `Confidence` object containing:
          - max_probability: Probability of the most likely outcome (0.0 to 1.0)
          - entropy_confidence: How certain the model is (0.0=uncertain, 1.0=certain)
                               Based on how concentrated the probability distribution is.
          - support_time: Total observation time for this feature combination.
                         Higher values indicate more reliable predictions.
          - used_features: Which features were actually used in the prediction.

        Example:
          ```python
          conf = model.confidence({"temp": "high", "humidity": "low"})
          if conf.entropy_confidence > 0.8 and conf.support_time > 3600:
              # High confidence prediction with sufficient data
              prediction = model.predict({"temp": "high", "humidity": "low"})
          ```
        """
        dist, key = self.distribution_with_key(features)
        if not dist:
            return Confidence(
                max_probability=0.0,
                entropy_confidence=0.0,
                support_time=0.0,
                used_features={},
            )

        entropy = self._entropy(dist)
        max_p = max(dist.values())

        state = self._states.get(key)
        support_time = sum(state.y.values()) if state else 0.0

        # Calculate entropy confidence: normalize entropy to [0, 1] range
        # max_entropy is the entropy of a uniform distribution
        # entropy_conf = 1.0 means deterministic (low entropy)
        # entropy_conf = 0.0 means uniform distribution (high entropy)
        max_entropy = math.log2(len(dist)) if len(dist) > 1 else 0.0
        entropy_conf = 1.0 - (entropy / max_entropy if max_entropy > 0 else 1.0)

        if not without_model_update:
            # We update concept drift state based on this confidence evaluation
            self._update_drift(
                entropy=entropy,
                max_p=max_p,
                support=support_time,
            )

        conf = Confidence(
            max_probability=max_p,
            entropy_confidence=entropy_conf,
            support_time=support_time,
            used_features=dict(key),
        )

        if not without_model_update:
            # We adapt the maximum feature subset depth based on drift and confidence
            self._adapt_max_depth(features, conf)

        return conf

    def interaction_contributions(
        self: Self, features: FeatureLabels
    ) -> dict[tuple[FeatureName, FeatureName], float]:
        """
        Gets interaction contributions for the given features.

        Interaction contribution measures how much better we predict by considering
        two features together versus separately. Positive values mean synergy,
        negative values mean the features interfere with each other.

        Args:
          features: Feature values to evaluate.

        Returns:
          A dictionary mapping feature pairs to their interaction contributions.
          Higher values indicate stronger positive interactions.

        Example:
          ```python
          interactions = model.interaction_contributions({
              "temp": "high",
              "humidity": "low",
              "wind": "strong"
          })
          # Returns: {("temp", "humidity"): 0.15, ("temp", "wind"): 0.02, ...}
          # temp+humidity together predict much better than separately
          ```
        """
        dist, key = self.distribution_with_key(features)
        if len(key) < 2:  # noqa: PLR2004
            # No interactions possible when less than 2 features are known
            return {}

        y_star = max(dist, key=dist.get)
        base = dist[y_star]
        interactions = {}

        for i in range(len(key)):
            for j in range(i + 1, len(key)):
                f1, f2 = key[i][0], key[j][0]
                delta = self._calculate_interaction_gain(y_star, key[i], key[j], base)
                interactions[(f1, f2)] = delta

        return interactions

    def drift_level(self) -> float:
        """
        Returns the current concept drift score.

        The drift score is a normalized value in [0, 1] that indicates how much
        the model's predictions are changing over time. It combines:
        - Entropy increase: Predictions becoming less certain
        - Max probability decrease: Top prediction becoming less confident

        Returns:
            Drift score in range [0, 1]:
            - 0.0-0.1: Stable predictions, no drift detected
            - 0.1-0.2: Minor fluctuations, normal variation
            - 0.2-0.4: Moderate drift, monitor closely
            - 0.4+: Strong drift, consider model retraining or intervention

        Example:
            ```python
            drift = model.drift_level()
            if drift > 0.3:
                print(f"Warning: High drift detected ({drift:.2f})")
            ```

        Note:
            Drift is calculated incrementally during confidence() calls.
            Call confidence() regularly to keep drift detection up to date.
        """
        return self._drift.drift_score

    def is_drifting(self, threshold: float = 0.2) -> bool:
        """
        Checks if concept drift exceeds a threshold.

        Args:
            threshold: Drift score threshold (0.0 to 1.0).
                      Recommended values:
                      - 0.1: Very sensitive, detects minor changes
                      - 0.2: Moderate sensitivity (default)
                      - 0.3: Conservative, only significant drift

        Returns:
            True if current drift score exceeds threshold.

        Example:
            ```python
            if model.is_drifting(threshold=0.25):
                # Take action: retrain model, alert user, etc.
                print("Concept drift detected!")
                model.drift_level()  # Get exact drift score
            ```
        """
        return self._drift.drift_score > threshold

    def _distribution_for_key(
        self: Self, key: FeatureKey
    ) -> dict[TargetLabel, Probability]:
        """
        Gets the predicted distribution for a specific feature key.

        Args:
          key: Feature key to evaluate.

        Returns:
          A dictionary mapping target values to their predicted probabilities.
        """
        # We get the number of known target labels
        k = len(self._y_domain)

        if key not in self._states:
            # If the key is unknown, return uniform distribution.
            return {y: 1.0 / k for y in self._y_domain} if k else {}

        # We get the state for the specified features
        state = self._get_state(key)

        # We calculate the total duration observed for all target labels
        total = sum(state.y.values())

        if total <= 0:
            # No data observed for this key
            return {}

        # We apply Laplace smoothing to calculate probabilities for each target label
        denominator = total + self.alpha * k
        return {
            y: (state.y.get(y, 0.0) + self.alpha) / denominator for y in self._y_domain
        }

    def _subset_score(self: Self, key: FeatureKey) -> Score:
        """
        Calculates the score of a feature subset based on feature and interaction scores.

        Args:
          key: Feature subset key.

        Returns:
          The calculated score for the feature subset.
        """
        score: Score = 0.0
        feats = [f for f, _ in key]

        for f in feats:
            # We add the individual feature scores
            score += self._normalize_feature_score(self._feature_scores[f])

        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                # We add the interaction scores (use get() to handle pruned interactions)
                interaction_score = self._interaction_scores.get(
                    (feats[i], feats[j]), 0.0
                )
                score += self._normalize_interaction_score(interaction_score)

        return score

    def _feature_subsets(self: Self, features: FeatureLabels) -> list[FeatureKey]:
        """
        Gets all feature subsets sorted by importance score. Higher importance
        subsets are returned first.

        Args:
          features: Feature values to evaluate.

        Returns:
          A list of feature subset keys sorted by importance score.
        """
        # We take only features with known (non-None) values
        known = [(k, v) for k, v in features.items() if v is not None]

        # We calculate the maximum subset size
        max_r = len(known)
        if self._max_depth is not None:
            max_r = min(max_r, self._max_depth)

        # We generate all possible subsets
        subsets = [
            tuple(sorted(combo))
            for r in range(max_r, 0, -1)
            for combo in combinations(known, r)
        ]

        # We sort subsets by their importance scores
        subsets.sort(key=self._subset_score, reverse=True)
        return subsets

    def _entropy(self: Self, dist: dict[TargetLabel, Probability]) -> float:
        """Calculates the entropy of a distribution."""
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def _update_feature_scores(self: Self, features: FeatureLabels) -> None:
        """
        Updates the feature importance scores based on information gain,
        using time-aware exponential moving average (EMA).

        The time-aware EMA adjusts adaptation speed based on observation duration
        to handle variable update frequencies:
        - duration < 1.0: Uses standard EMA (prevents over-dampening)
        - duration ≥ 1.0: Accelerates adaptation proportionally

        This prevents slow learning during concept drift when updates are rare.
        Example: If updates occur every 60 seconds with duration=60, adaptation
        is ~60x faster than with fixed decay.

        Formula: d_eff = priority_decay^max(1.0, duration)
        - duration = 0.5 → d_eff = 0.98^1.0 = 0.980 (new weight = 2%)
        - duration = 1.0 → d_eff = 0.98^1.0 = 0.980 (new weight = 2%)
        - duration = 60 → d_eff = 0.98^60 ≈ 0.294 (new weight ≈ 71%)
        """
        global_dist = self._distribution_for_key(())
        if not global_dist:
            return

        base_entropy = self._entropy(global_dist)

        for f, v in features.items():
            if v is None:
                continue

            key = ((f, v),)
            if key not in self._states:
                continue

            cond_dist = self._distribution_for_key(key)
            ig = base_entropy - self._entropy(cond_dist)

            # Time-aware EMA: max(1.0, duration) provides baseline for frequent
            # updates while accelerating adaptation for infrequent updates
            d_eff = self.priority_decay ** max(1.0, self._last_duration)

            self._feature_scores[f] = (
                d_eff * self._feature_scores[f] + (1.0 - d_eff) * ig
            )

    def _update_interaction_scores(
        self: Self, features: dict[str, Hashable], duration: float
    ) -> None:
        known = [(f, v) for f, v in features.items() if v is not None]
        if len(known) < 2:  # noqa: PLR2004
            return

        full_key = tuple(sorted(known))
        full_dist = self._distribution_for_key(full_key)
        if not full_dist:
            return

        y_star = max(full_dist, key=full_dist.get)
        base = full_dist[y_star]

        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                f1, f2 = known[i][0], known[j][0]

                d1 = self._distribution_for_key((known[i],))
                d2 = self._distribution_for_key((known[j],))
                best_single = max(
                    d1.get(y_star, 0.0),
                    d2.get(y_star, 0.0),
                )

                delta = base - best_single
                key = (f1, f2)

                # EMA update (use get() to handle pruned interactions)
                current_score = self._interaction_scores.get(key, 0.0)
                self._interaction_scores[key] = (
                    self.interaction_decay * current_score
                    + (1 - self.interaction_decay) * delta
                )

                # support accumulation (duration-aware)
                self._interaction_support[key] = (
                    self._interaction_support.get(key, 0.0) + duration
                )

    def _calculate_interaction_gain(
        self: Self,
        y_star: TargetLabel,
        feature1: tuple[FeatureName, FeatureLabel],
        feature2: tuple[FeatureName, FeatureLabel],
        base_probability: float,
    ) -> float:
        """
        Calculates the interaction gain between two features.

        The interaction gain is the additional predictive value obtained by
        considering two features together versus considering each individually.

        Args:
            y_star: The target label to evaluate.
            feature1: First feature (name, value) tuple.
            feature2: Second feature (name, value) tuple.
            base_probability: Probability of y_star with both features.

        Returns:
            The interaction gain (can be negative if interaction hurts prediction).
        """
        d1 = self._distribution_for_key((feature1,))
        d2 = self._distribution_for_key((feature2,))
        best_single = max(
            d1.get(y_star, 0.0),
            d2.get(y_star, 0.0),
        )
        return base_probability - best_single

    def _time_decay(self: Self, last_t: float, now: float) -> float:
        """
        Calculates time-based decay factor using exponential half-life.

        Uses base-2 exponential decay: after `self.decay` time units,
        the weight is halved. After 2*decay time units, weight is 1/4, etc.

        Args:
            last_t: Previous timestamp.
            now: Current timestamp.

        Returns:
            Decay factor between 0.0 and 1.0.
        """
        if self.decay <= 0:
            return 1.0
        dt = max(0.0, now - last_t)
        return 2 ** (-dt / self.decay)

    def _apply_decay(self: Self, key: FeatureKey) -> None:
        """
        Applies time-based decay to a feature state.

        Args:
            key: Feature key identifying the state to decay.
        """
        state = self._states[key]
        w = self._time_decay(state.last_t, self._t)

        if w < 1.0:
            for y in state.y:
                state.y[y] *= w
            state.last_t = self._t

    def _get_state(self: Self, key: FeatureKey) -> FeatureState:
        """
        Gets or creates a feature state, applying decay if it exists.

        Args:
            key: Feature key to retrieve.

        Returns:
            The feature state for the given key.
        """
        if key not in self._states:
            self._states[key] = FeatureState(y=defaultdict(float), last_t=self._t)
        else:
            self._apply_decay(key)
        return self._states[key]

    def _normalize_feature_score(self: Self, score: float) -> float:
        """
        Normalizes feature score to (-1, 1) range using algebraic normalization.

        This prevents features with extremely high information gain from
        dominating the subset selection process. Uses the formula:
        normalized = x / (1 + |x|)

        Args:
            score: Raw feature importance score (unbounded).

        Returns:
            Normalized score in range (-1, 1).
        """
        return score / (1.0 + abs(score))

    def _normalize_interaction_score(self: Self, score: float) -> float:
        """
        Normalizes interaction score to (-1, 1) range using hyperbolic tangent.

        This prevents feature interactions with extreme values from dominating
        subset selection. tanh provides smooth, bounded output that handles
        both positive (synergy) and negative (interference) interactions.

        Args:
            score: Raw interaction score (unbounded).

        Returns:
            Normalized score in range (-1, 1).
        """
        return math.tanh(score)

    def _prune_interactions(self: Self) -> None:
        """
        Prunes feature interactions to manage memory and reduce noise.

        Applies two-stage pruning:
        1. Quality-based filtering: Removes interactions with insufficient support
           or weak normalized scores (likely noise or irrelevant patterns).
        2. Capacity-based limiting: If interactions exceed max_interactions,
           keeps only the top-K by absolute score (strongest signals).

        This prevents:
        - Memory bloat from tracking too many interactions
        - Noise from rarely observed or weak interaction patterns
        - Overfitting to spurious correlations

        Called after each update to maintain model health.
        """
        # Stage 1: Filter by score and support thresholds
        for key in list(self._interaction_scores.keys()):
            score = self._interaction_scores[key]
            norm_score = math.tanh(score)
            support = self._interaction_support.get(key, 0.0)

            # Remove weak interactions (likely noise)
            if abs(norm_score) < self.min_interaction_score:
                del self._interaction_scores[key]
                self._interaction_support.pop(key, None)
                continue

            # Remove interactions with insufficient support
            if support < self.min_interaction_support:
                del self._interaction_scores[key]
                self._interaction_support.pop(key, None)

        # Stage 2: Top-K pruning to limit memory usage
        if len(self._interaction_scores) > self.max_interactions:
            keep = sorted(
                self._interaction_scores.items(), key=lambda x: abs(x[1]), reverse=True
            )[: self.max_interactions]

            keep_keys = {k for k, _ in keep}
            self._interaction_scores = dict(keep)
            self._interaction_support = {
                k: v for k, v in self._interaction_support.items() if k in keep_keys
            }

    def _ema_update(self, old: float, new: float, last_t: float) -> float:
        if last_t == 0.0:
            return new

        w = self._time_decay(last_t, self._t)
        return w * old + (1.0 - w) * new

    def _update_drift(self, entropy: float, max_p: float, support: float) -> None:
        """
        Updates concept drift detection state.

        Monitors prediction stability by tracking:
        1. Entropy: Measures prediction uncertainty (0 = certain, higher = uncertain)
        2. Max probability: Confidence in top prediction (0 = uncertain, 1 = certain)
        3. Support: Amount of training data (normalized to [0, 1])

        Drift is detected when:
        - Entropy increases (predictions becoming more uncertain)
        - Max probability decreases (less confidence in top prediction)
        - Support decreases (less training data available)

        Uses time-aware exponential moving average to track baselines and
        detect significant shifts from expected behavior.

        Args:
            entropy: Current prediction entropy (unnormalized).
            max_p: Current maximum probability.
            support: Current support time.

        Updates:
            self._drift.drift_score with combined weighted drift signal [0, 1].
        """
        # Normalize metrics to [0, 1]
        norm_entropy = (
            entropy / math.log2(len(self._y_domain)) if len(self._y_domain) > 1 else 0.0
        )
        norm_maxp = max_p
        # log-scaling for support to avoid quick saturation
        norm_support = math.log1p(support) / (math.log1p(support) + 1.0)

        d = self._drift

        # EMA updates for baselines
        d.entropy_ema = self._ema_update(d.entropy_ema, norm_entropy, d.last_t)
        d.maxp_ema = self._ema_update(d.maxp_ema, norm_maxp, d.last_t)
        d.support_ema = self._ema_update(d.support_ema, norm_support, d.last_t)

        w = self.drift_weights

        # Shift signals: only positive shifts indicate drift
        entropy_shift = max(0.0, norm_entropy - d.entropy_ema)  # Increasing uncertainty
        maxp_shift = max(0.0, d.maxp_ema - norm_maxp)  # Decreasing confidence
        support_shift = max(0.0, d.support_ema - norm_support)

        # Combine signals with weights and saturate to [0, 1]
        d.drift_score = (
            w.entropy * entropy_shift + w.max_p * maxp_shift + w.support * support_shift
        )
        d.drift_score = min(max(d.drift_score, 0.0), 1.0)
        d.last_t = self._t

    def _adapt_max_depth(self, features: FeatureLabels, conf: Confidence) -> None:
        """
        Dynamically adjusts max_depth for optimal prediction accuracy.

        Balances model complexity with robustness by considering:
        - **Feature count**: Prevents overfitting in high-dimensional spaces
        - **Concept drift**: Reduces complexity when drift detected (> 0.3)
        - **Prediction confidence**: Simplifies model when uncertain (< 0.6)

        Strategy:
        1. Base depth from feature count:
           - ≤5 features: Use all (low risk of overfitting)
           - 6-10 features: Cap at 3 (balance complexity/performance)
           - >10 features: Cap at 2 (prevent curse of dimensionality)
        2. Apply proportional scaling based on drift score:
           - adaptive_depth = base_depth * (1 - drift_score)
           - High drift (e.g., 0.5) reduces depth by 50%
        3. Further reduce by 1 if confidence < 0.6 (minimum 1)

        This ensures the model remains stable during distribution shifts
        and falls back to simpler, more robust patterns when uncertain.

        Args:
            features: Current feature values (used to count known features).
            conf: Current prediction confidence metrics.
        """
        n_features = sum(1 for v in features.values() if v is not None)

        # Base max depth from feature count
        if n_features <= 5:  # noqa: PLR2004
            base_depth = n_features
        elif n_features <= 10:  # noqa: PLR2004
            base_depth = min(3, n_features)
        else:
            base_depth = 2

        # Adaptive scaling: reduce depth proportionally to drift score
        drift_factor = 1.0 - self._drift.drift_score  # 0 = full drift, 1 = no drift
        adaptive_depth = max(1, int(base_depth * drift_factor))

        # Further reduce if confidence low
        if conf.entropy_confidence < 0.6:  # noqa: PLR2004
            adaptive_depth = max(1, adaptive_depth - 1)

        self._max_depth = adaptive_depth

    def to_dict(self: Self) -> dict:
        """
        Serializes the model state to a dictionary.

        Returns:
            A dictionary containing all model parameters and state that can be
            used to recreate the model via from_dict().

        Example:
            ```python
            model = DiscreteConditionalModel(alpha=1.0, decay=3600.0)
            # ... train model ...
            state = model.to_dict()
            # Save state to file, database, etc.
            ```
        """
        return {
            # Hyperparameters
            "alpha": self.alpha,
            "decay": self.decay,
            "priority_decay": self.priority_decay,
            "interaction_decay": self.interaction_decay,
            "max_interactions": self.max_interactions,
            "min_interaction_support": self.min_interaction_support,
            "min_interaction_score": self.min_interaction_score,
            "drift_weights": {
                "entropy": self.drift_weights.entropy,
                "max_p": self.drift_weights.max_p,
                "support": self.drift_weights.support,
            },
            "confidence_weights": {
                "entropy": self.confidence_weights.entropy,
                "max_p": self.confidence_weights.max_p,
                "support": self.confidence_weights.support,
            },
            "support_tau": self.support_tau,
            "min_step_confidence": self.min_step_confidence,
            # Internal state
            "_max_depth": self._max_depth,
            "_t": self._t,
            "_last_duration": self._last_duration,
            "_y_domain": list(self._y_domain),
            "_feature_scores": dict(self._feature_scores),
            "_interaction_scores": {
                f"{k[0]}|{k[1]}": v for k, v in self._interaction_scores.items()
            },
            "_interaction_support": {
                f"{k[0]}|{k[1]}": v for k, v in self._interaction_support.items()
            },
            "_states": {
                "|".join(f"{fname}={flabel}" for fname, flabel in key): {
                    "y": dict(state.y),
                    "last_t": state.last_t,
                }
                for key, state in self._states.items()
            },
            "_drift": {
                "entropy_ema": self._drift.entropy_ema,
                "maxp_ema": self._drift.maxp_ema,
                "support_ema": self._drift.support_ema,
                "drift_score": self._drift.drift_score,
                "last_t": self._drift.last_t,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Deserializes a model from a dictionary created by to_dict().

        Args:
            data: Dictionary containing model state from to_dict().

        Returns:
            A new DiscreteConditionalModel instance with restored state.

        Example:
            ```python
            # Load state from file, database, etc.
            state = load_model_state()
            model = DiscreteConditionalModel.from_dict(state)
            # Continue using the model
            ```

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data contains invalid values.
        """
        # Create drift weights
        drift_weights = DriftWeights(
            entropy=data["drift_weights"]["entropy"],
            max_p=data["drift_weights"]["max_p"],
            support=data["drift_weights"]["support"],
        )

        # Create confidence weights
        confidence_weights_data = data.get("confidence_weights")
        if confidence_weights_data:
            confidence_weights = ConfidenceWeights(
                entropy=confidence_weights_data["entropy"],
                max_p=confidence_weights_data["max_p"],
                support=confidence_weights_data["support"],
            )
        else:
            confidence_weights = None

        # Extract internal state
        internal_state = {
            "_max_depth": data["_max_depth"],
            "_t": data["_t"],
            "_last_duration": data["_last_duration"],
            "_y_domain": data["_y_domain"],
            "_feature_scores": data["_feature_scores"],
            "_interaction_scores": data["_interaction_scores"],
            "_interaction_support": data["_interaction_support"],
            "_states": data["_states"],
            "_drift": data["_drift"],
        }

        # Create model with all parameters including internal state
        return cls(
            alpha=data["alpha"],
            decay=data["decay"],
            priority_decay=data["priority_decay"],
            interaction_decay=data["interaction_decay"],
            max_interactions=data["max_interactions"],
            min_interaction_support=data["min_interaction_support"],
            min_interaction_score=data["min_interaction_score"],
            drift_weights=drift_weights,
            confidence_weights=confidence_weights,
            support_tau=data.get("support_tau", 3600.0),
            min_step_confidence=data.get("min_step_confidence", 0.0),
            _internal_state=internal_state,
        )

    def _support_confidence(self: Self, support_time: float) -> float:
        """Maps support time to [0, 1] using exponential saturation."""
        if support_time <= 0:
            return 0.0
        return 1.0 - math.exp(-support_time / self.support_tau)

    def _step_confidence(self: Self, conf: Confidence) -> float:
        """Computes a single-step confidence scalar in [0, 1]."""
        w = self.confidence_weights

        # Entropy confidence already in [0, 1]
        c_entropy = conf.entropy_confidence

        # Max probability rescaled from [1/K, 1] → [0, 1]
        k = max(1, len(self._y_domain))
        min_p = 1.0 / k
        c_max_p = (
            (conf.max_probability - min_p) / (1.0 - min_p)
            if conf.max_probability > min_p
            else 0.0
        )

        # Support confidence (bounded)
        c_support = self._support_confidence(conf.support_time)

        c = w.entropy * c_entropy + w.max_p * c_max_p + w.support * c_support

        # Hard clamp for safety
        return max(0.0, min(1.0, c))
