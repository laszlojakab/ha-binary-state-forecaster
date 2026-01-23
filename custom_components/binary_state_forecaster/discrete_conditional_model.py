"""Model for discrete conditional forecasting model."""

import math
from collections import defaultdict
from collections.abc import Hashable
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
        max_depth: int | None = None,
        priority_decay: float = 0.98,
        interaction_decay: float = 0.98,
    ) -> None:
        """
        Initializes the model.

        Args:
            alpha: Laplace smoothing parameter for probability estimates (must be > 0).
                   Higher values make predictions more uniform.
            decay: Half-life for temporal decay in same units as duration (must be >= 0).
                   After this time, old observations have half their original weight.
                   Set to 0 to disable decay.
            max_depth: Maximum feature subset size to consider. None means no limit.
                      Lower values improve performance but reduce model expressiveness.
            priority_decay: Exponential moving average factor for feature importance (0.0 to 1.0).
                           Higher values give more weight to historical scores.
            interaction_decay: Exponential moving average factor for interaction scores (0.0 to 1.0).
                              Higher values give more weight to historical scores.

        Example:
            ```python
            # For sensor data updated every minute with 1-hour decay
            model = DiscreteConditionalModel(
                alpha=1.0,
                decay=3600.0,  # 1 hour in seconds
                max_depth=3,   # Consider up to 3 features together
            )
            ```

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
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")

        self.alpha = alpha
        """Smoothing parameter for probability estimates."""

        self.decay = decay
        """Half-life for temporal decay (same unit as duration)."""

        self.max_depth = max_depth
        """Maximum feature subset size to consider."""

        self.priority_decay = priority_decay
        """Decay factor for feature importance scores."""

        self.interaction_decay = interaction_decay
        """Decay factor for interaction scores."""

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
        self._update_interaction_scores(features)

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

    def confidence(self: Self, features: FeatureLabels) -> Confidence:
        """
        Gets confidence metrics for the given features.

        Args:
          features: Feature values to evaluate.

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

        return Confidence(
            max_probability=max_p,
            entropy_confidence=entropy_conf,
            support_time=support_time,
            used_features=dict(key),
        )

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
            score += self._feature_scores[f]

        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                # We add the interaction scores
                score += self._interaction_scores[(feats[i], feats[j])]

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
        if self.max_depth is not None:
            max_r = min(max_r, self.max_depth)

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
        """Updates the feature importance scores based on information gain."""
        global_dist = self._distribution_for_key(())
        if not global_dist:
            # If no global distribution is available, we cannot compute information gain
            return

        base_entropy = self._entropy(global_dist)

        for f, v in features.items():
            if v is None:
                # We skip unknown feature values
                continue

            key = ((f, v),)
            if key not in self._states:
                # If the feature value has not been observed, skip.
                continue

            # We calculate information gain for the feature
            ig = base_entropy - self._entropy(self._distribution_for_key(key))

            # We update the feature score
            self._feature_scores[f] = (
                self.priority_decay * self._feature_scores[f]
                + (1 - self.priority_decay) * ig
            )

    def _update_interaction_scores(self: Self, features: FeatureLabels) -> None:
        """
        Updates interaction scores between feature pairs.

        Calculates how much predictive value is gained by considering two features
        together compared to considering them individually.

        Args:
            features: Feature values for the current observation.
        """
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
                delta = self._calculate_interaction_gain(
                    y_star, known[i], known[j], base
                )

                key = (f1, f2)
                self._interaction_scores[key] = (
                    self.interaction_decay * self._interaction_scores[key]
                    + (1 - self.interaction_decay) * delta
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
