"""Discrete conditional probability model for state prediction."""

import math
import time
from dataclasses import dataclass, field
from typing import Final, Self

from custom_components.discrete_state_forecaster.model.state import State


@dataclass
class StateStats:
    """
    Statistics for state occurrences and durations.

    Tracks the cumulative duration spent in each state and provides methods
    to compute total time and probability distributions over states. This is
    useful for modeling state transitions and predicting future states based
    on historical patterns.

    Attributes:
        durations: A dictionary mapping state values to their cumulative durations.
            The duration represents the total time spent in each state.
        last_update_ts: Timestamp of the last decay application. Used to calculate
            elapsed time for exponential decay.
        baseline: Reference probability distribution for concept drift detection.
            Set to current distribution on first drift check, updated when drift
            is detected.
        last_drift_ts: Timestamp of the last detected concept drift. Used to enforce
            cooldown periods between drift detections.
        fast_decay_updates: Counter for remaining updates with accelerated decay.
            Set to 15 when concept drift is detected, decrements on each update.
            When > 0, decay uses a 10x faster half-life to quickly forget old
            patterns and adapt to new behavior. Enables rapid model adaptation
            after detecting pattern changes.

    Example:
        ```
        >>> stats = StateStats()
        >>> stats.durations = {"on": 100.0, "off": 200.0}
        >>> stats.total()
        300.0
        >>> stats.distribution()
        {'on': 0.3333333333333333, 'off': 0.6666666666666666}
        ```
    """

    durations: Final[dict[State, float]] = field(default_factory=dict)
    last_update_ts: float | None = None
    baseline: dict[State, float] | None = None
    last_drift_ts: float = 0.0
    fast_decay_updates: int = 0

    def total(self: Self) -> float:
        """
        Calculates the total duration across all states.

        Sums the durations of all tracked states to provide the total
        time represented in the statistics.

        Returns:
            The sum of all state durations. Returns 0.0 if no durations
            are recorded.

        Example:
            ```
            >>> stats = StateStats()
            >>> stats.durations = {"on": 100.0, "off": 200.0}
            >>> stats.total()
            300.0
            ```
        """
        return sum(self.durations.values())

    def update_duration(
        self: Self,
        state: State,
        duration: float,
    ) -> None:
        """
        Adds duration to a specific state's cumulative time.

        Updates the tracked duration for a given state by adding the specified
        duration value to its current total. If the state doesn't exist yet in
        the durations dictionary, it is initialized with the provided duration.

        This method is the primary way to record state observations, accumulating
        time spent in each state incrementally as new observations arrive.

        Args:
            state: The state identifier to update. Can be any hashable type
                (typically a string like "on", "off", "heating", etc.).
            duration: The duration in seconds to add to this state's total.
                Should be non-negative in typical usage, though negative values
                are technically allowed and will reduce the state's total duration.

        Side Effects:
            Modifies the `durations` dictionary by either:
            - Creating a new entry if the state doesn't exist
            - Adding to the existing duration if the state already exists

        Examples:
            ```
            >>> stats = StateStats()
            >>> stats.update_duration("on", 100.0)
            >>> stats.durations
            {'on': 100.0}

            >>> # Adding more time to the same state
            >>> stats.update_duration("on", 50.0)
            >>> stats.durations
            {'on': 150.0}

            >>> # Adding a new state
            >>> stats.update_duration("off", 200.0)
            >>> stats.durations
            {'on': 150.0, 'off': 200.0}

            >>> # Multiple states tracked independently
            >>> stats = StateStats()
            >>> stats.update_duration("heating", 300.0)
            >>> stats.update_duration("cooling", 150.0)
            >>> stats.update_duration("heating", 100.0)
            >>> stats.durations
            {'heating': 400.0, 'cooling': 150.0}
            ```

        Note:
            This method does not update `last_update_ts` or trigger any decay
            calculations. It only modifies the `durations` dictionary. For time-aware
            updates with decay, use this in conjunction with `apply_decay()`.
        """
        self.durations[state] = self.durations.get(state, 0.0) + duration

    def distribution(self: Self) -> dict[State, float]:
        """
        Calculates the probability distribution over states.

        Computes the normalized probability for each state based on the
        proportion of total time spent in that state. Each probability
        represents the fraction of total time spent in the corresponding state.

        Returns:
            A dictionary mapping state values to their probabilities
            (values between 0 and 1). Returns an empty dictionary if
            the total duration is zero.

        Example:
        ```
            >>> stats = StateStats()
            >>> stats.durations = {"on": 100.0, "off": 200.0}
            >>> stats.distribution()
            {'on': 0.3333333333333333, 'off': 0.6666666666666666}
            >>> stats = StateStats()
            >>> stats.distribution()
            {}
        ```
        """
        total = self.total()
        if total == 0:
            return {}

        return {k: v / total for k, v in self.durations.items()}

    def apply_decay(self: Self, timestamp: float | None = None, half_life: float = 0.0) -> None:
        """
        Applies exponential decay to all state durations.

        Scales each state's cumulative duration by an exponential decay factor
        based on the elapsed time since the last update. This models the idea
        that older observations should have diminishing influence over time.

        The decay uses a half-life parameter, where observations lose 50% of their
        weight after the specified time period. The decay factor is computed as:

            decay_factor = exp(-ln(2) * elapsed / half_life)
                        = 0.5 ^ (elapsed / half_life)

        where ``elapsed`` is ``timestamp - last_update_ts``. When ``last_update_ts``
        is ``None`` (i.e., this is the first update), the method initializes
        ``last_update_ts`` to ``timestamp`` and returns without modifying durations.

        Args:
            timestamp: The current timestamp as a float (e.g., seconds since epoch).
                If None, uses the current system time (time.time()). This allows
                for deterministic behavior in tests while providing convenience in
                production use.
            half_life: The half-life in seconds. After this duration, observations
                retain 50% of their original weight. A value of 0 or negative
                disables decay (durations unchanged). Larger values mean slower
                forgetting (e.g., 86400 = 1 day half-life). Defaults to 0.0.

        Behavior:
            - If ``timestamp`` is ``None``: uses current system time (``time.time()``).
            - If ``last_update_ts`` is ``None``: set it to ``timestamp`` and return.
            - If ``timestamp`` is less than or equal to ``last_update_ts``: do nothing.
            - If ``half_life`` is 0 or negative: update timestamp but skip decay.
            - Otherwise: apply decay to all entries in ``durations`` and update
              ``last_update_ts`` to ``timestamp``.

        Examples:
            ```
            >>> stats = StateStats(durations={"on": 100.0, "off": 200.0})
            >>> stats.last_update_ts = 0.0
            >>> # After one half-life, durations reduced to 50%
            >>> stats.apply_decay(timestamp=3600.0, half_life=3600.0)
            >>> stats.durations["on"]
            50.0
            >>> stats.durations["off"]
            100.0
            
            >>> # Using default current time
            >>> stats2 = StateStats(durations={"heating": 500.0})
            >>> stats2.apply_decay(half_life=3600.0)  # Uses time.time()
            >>> # Durations decay based on current system time
            ```
        """
        if timestamp is None:
            timestamp = time.time()
            
        if self.last_update_ts is None:
            self.last_update_ts = timestamp
            return

        elapsed = timestamp - self.last_update_ts
        if elapsed <= 0:
            return

        if half_life <= 0:
            self.last_update_ts = timestamp
            return

        decay_factor = 0.5 ** (elapsed / half_life)

        for state in self.durations:
            self.durations[state] *= decay_factor

        self.last_update_ts = timestamp

    def prune(
        self,
        min_state_duration: float,
    ) -> None:
        """
        Removes states with durations below a minimum threshold.

        Prunes the state statistics by removing any states whose cumulative
        duration falls below the specified minimum. This is useful for:

        - Reducing memory usage by removing rarely-seen states
        - Eliminating noise from states with insufficient data
        - Cleaning up after decay has reduced old state durations
        - Maintaining model quality by focusing on frequently-observed states

        Args:
            min_state_duration: The minimum duration threshold in seconds.
                States with durations strictly less than this value are removed.
                Must be non-negative. A value of 0 has no effect.

        Behavior:
            - Iterates through all states and identifies those with
              ``duration < min_state_duration``
            - Removes identified states from the ``durations`` dictionary
            - Does not modify ``last_update_ts``
            - If all states are removed, ``durations`` becomes empty

        Examples:
            ```
            >>> stats = StateStats(durations={"on": 100.0, "off": 5.0, "idle": 50.0})
            >>> stats.prune(min_state_duration=10.0)
            >>> stats.durations
            {'on': 100.0, 'idle': 50.0}

            >>> # After decay, prune low-weight states
            >>> stats = StateStats(durations={"on": 100.0, "off": 200.0})
            >>> stats.last_update_ts = 0.0
            >>> stats.apply_decay(current_ts=86400.0, half_life=3600.0)
            >>> # Durations now very small after 24 hours with 1-hour half-life
            >>> stats.prune(min_state_duration=1.0)
            >>> stats.durations
            {}
            ```
        """
        self.durations = {
            s: d for s, d in self.durations.items() if d >= min_state_duration
        }

    def prune_adaptive(
        self,
        epsilon: float = 0.003,
        absolute_min: float = 20.0,
    ) -> None:
        """
        Adaptively prunes states using relative and absolute thresholds.

        Removes states whose durations fall below a dynamically-calculated threshold
        based on the total duration. This adaptive approach automatically adjusts
        to the scale of the data, making it more robust than fixed thresholds when
        dealing with varying amounts of historical data.

        The threshold is calculated as: ``max(total * epsilon, absolute_min)``

        This means:
        - States must represent at least ``epsilon`` fraction of total duration
        - States must also meet the absolute minimum duration
        - Whichever threshold is higher takes precedence

        This dual-threshold approach ensures:
        - Relative pruning: Remove states contributing < ε% of total (e.g., 0.3%)
        - Absolute floor: Prevent removal of all states when total is small
        - Scale invariance: Works well with both small and large datasets

        Args:
            epsilon: Relative threshold as fraction of total duration (0 to 1).
                States contributing less than this fraction are removed.
                Default is 0.003 (0.3%), which removes states representing less
                than 0.3% of total time. Typical values: 0.001-0.01 (0.1%-1%).
            absolute_min: Absolute minimum duration in seconds. States below
                this value are removed regardless of their relative proportion.
                Default is 20.0 seconds. Provides a safety floor when total
                duration is small. Typical values: 10-60 seconds.

        Behavior:
            - Calculates total duration across all states
            - If total is 0, clears all durations and returns
            - Computes threshold as max(total x epsilon, absolute_min)
            - Removes states with duration < threshold
            - Does not modify ``last_update_ts``

        Examples:
            ```
            >>> # Example 1: Large dataset (total=10,000s)
            >>> stats = StateStats(durations={"on": 9000.0, "off": 980.0, "idle": 20.0})
            >>> stats.prune_adaptive(epsilon=0.003, absolute_min=20.0)
            >>> # threshold = max(10000 * 0.003, 20.0) = max(30, 20) = 30
            >>> # idle (20) < 30, removed
            >>> stats.durations
            {'on': 9000.0, 'off': 980.0}

            >>> # Example 2: Small dataset (total=100s)
            >>> stats = StateStats(durations={"on": 80.0, "off": 15.0, "idle": 5.0})
            >>> stats.prune_adaptive(epsilon=0.003, absolute_min=20.0)
            >>> # threshold = max(100 * 0.003, 20.0) = max(0.3, 20) = 20
            >>> # off (15) and idle (5) < 20, removed
            >>> stats.durations
            {'on': 80.0}

            >>> # Example 3: After decay with adaptive cleanup
            >>> stats = StateStats(durations={"heating": 500.0, "cooling": 300.0, "idle": 50.0})
            >>> stats.last_update_ts = 0.0
            >>> stats.apply_decay(current_ts=86400.0, half_life=43200.0)  # 12-hour half-life
            >>> # After 24 hours (2 half-lives): all durations x 0.25
            >>> # heating: 125, cooling: 75, idle: 12.5
            >>> stats.prune_adaptive(epsilon=0.01, absolute_min=15.0)
            >>> # threshold = max(212.5 * 0.01, 15.0) = max(2.125, 15) = 15
            >>> # idle (12.5) < 15, removed
            >>> stats.durations.keys()
            dict_keys(['heating', 'cooling'])

            >>> # Example 4: Empty durations
            >>> stats = StateStats()
            >>> stats.prune_adaptive()
            >>> stats.durations
            {}
            ```

        Note:
            Unlike ``prune()`` which uses a fixed threshold, ``prune_adaptive()``
            automatically scales the threshold based on total duration, making it
            ideal for:
            - Long-running systems with varying data volumes
            - Post-decay cleanup where totals decrease over time
            - Removing low-contribution states while preserving significant ones
        """
        threshold = max(self.total() * epsilon, absolute_min)

        self.prune(threshold)

    def check_drift(
        self,
        now_ts: float,
        min_support: float = 3600,
        threshold: float = 0.15,
        cooldown: float = 3600,
    ) -> bool:
        """
        Detects concept drift in state distributions over time.

        Monitors changes in the probability distribution of states to detect
        significant shifts in patterns (concept drift). Uses Jensen-Shannon
        divergence to measure the difference between current and baseline
        distributions, triggering drift detection when divergence exceeds
        a threshold.

        Concept drift detection is useful for:
        - Identifying when behavioral patterns change significantly
        - Triggering model retraining or adaptation
        - Monitoring system state evolution
        - Alerting on unexpected distribution shifts

        The method maintains a baseline distribution and compares each new
        distribution against it. When drift is detected, the baseline is
        reset to the current distribution to track new patterns.

        Args:
            now_ts: Current timestamp for drift detection. Used for cooldown
                period enforcement to prevent repeated drift signals.
            min_support: Minimum total duration required before drift detection
                is enabled. Default is 3600 seconds (1 hour). Prevents false
                positives when insufficient data has been collected.
                Typical values: 1800-7200 (30 minutes to 2 hours).
            threshold: Jensen-Shannon divergence threshold for drift detection.
                Default is 0.15. Values range from 0 (identical distributions)
                to 1 (completely different distributions). Higher values mean
                less sensitive (only detect major shifts).
                Typical values: 0.1-0.3 (10%-30% divergence).
            cooldown: Minimum time in seconds between drift detections. Default
                is 3600 seconds (1 hour). Prevents rapid successive drift
                signals when distributions fluctuate. Typical values: 1800-7200
                (30 minutes to 2 hours).

        Returns:
            True if concept drift is detected, False otherwise.

        Behavior:
            1. If total duration < min_support: returns False (insufficient data)
            2. Calculates current distribution from durations
            3. If no baseline exists: sets baseline to current, returns False
            4. If time since last drift < cooldown: returns False (in cooldown)
            5. Calculates Jensen-Shannon divergence between current and baseline
            6. If divergence > threshold:
               - Sets last_drift_ts to now_ts
               - Resets baseline to current distribution
               - Returns True (drift detected)
            7. Otherwise: returns False (no drift)

        Side Effects:
            - Sets self.baseline on first call or when drift detected
            - Updates self.last_drift_ts when drift detected
            - Modifies internal state to track baseline distribution

        Examples:
            ```
            >>> # Example 1: Initial baseline establishment
            >>> stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
            >>> drift = stats.check_drift(now_ts=1000.0)
            >>> drift
            False
            >>> stats.baseline
            {'on': 0.75, 'off': 0.25}

            >>> # Example 2: Significant pattern shift detected
            >>> stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
            >>> stats.baseline = {"on": 0.75, "off": 0.25}
            >>> stats.last_drift_ts = 0.0
            >>> # Pattern changes: now mostly 'off'
            >>> stats.durations = {"on": 500.0, "off": 3500.0}
            >>> drift = stats.check_drift(now_ts=5000.0, threshold=0.15)
            >>> drift
            True
            >>> # Baseline updated to new pattern
            >>> stats.baseline
            {'on': 0.125, 'off': 0.875}

            >>> # Example 3: Insufficient data (below min_support)
            >>> stats = StateStats(durations={"on": 100.0, "off": 50.0})
            >>> drift = stats.check_drift(now_ts=1000.0, min_support=1000.0)
            >>> drift
            False

            >>> # Example 4: Cooldown period prevents rapid detection
            >>> stats = StateStats(durations={"on": 1000.0, "off": 3000.0})
            >>> stats.baseline = {"on": 0.75, "off": 0.25}
            >>> stats.last_drift_ts = 1000.0
            >>> # Try to detect drift only 500s later (< cooldown of 3600s)
            >>> stats.durations = {"on": 500.0, "off": 3500.0}
            >>> drift = stats.check_drift(now_ts=1500.0, cooldown=3600.0)
            >>> drift
            False  # Blocked by cooldown
            >>> # After cooldown expires
            >>> drift = stats.check_drift(now_ts=5000.0, cooldown=3600.0)
            >>> drift
            True  # Now detected

            >>> # Example 5: Small changes don't trigger drift
            >>> stats = StateStats(durations={"on": 3000.0, "off": 1000.0})
            >>> stats.baseline = {"on": 0.75, "off": 0.25}
            >>> stats.last_drift_ts = 0.0
            >>> # Slight change: 73% vs 75%
            >>> stats.durations = {"on": 2900.0, "off": 1100.0}
            >>> drift = stats.check_drift(now_ts=5000.0, threshold=0.15)
            >>> drift
            False  # JS divergence too small
            ```

        Note:
            Jensen-Shannon divergence is a symmetric measure of distribution
            similarity, ranging from 0 (identical) to 1 (completely different).
            It's more stable than KL-divergence because it handles missing
            states gracefully and is always finite.

            The baseline reset strategy ensures the detector adapts to new
            patterns after drift is detected, preventing continuous drift
            signals once the new pattern stabilizes.
        """
        if self.total() < min_support:
            return False

        current = self.distribution()

        # Create baseline on first call
        if self.baseline is None:
            self.baseline = current
            return False

        if now_ts - self.last_drift_ts < cooldown:
            return False

        js = self._js_divergence(current, self.baseline)

        if js > threshold:
            self.last_drift_ts = now_ts

            # Reset baseline to current distribution
            self.baseline = current

            return True

        return False

    def _js_divergence(
        self,
        p: dict[State, float],
        q: dict[State, float],
    ) -> float:
        """
        Calculates the Jensen-Shannon divergence between two distributions.

        Jensen-Shannon (JS) divergence is a symmetric and smoothed measure of
        the difference between two probability distributions. It's based on
        Kullback-Leibler (KL) divergence but has several advantages:
        - Symmetric: JS(P||Q) = JS(Q||P)
        - Bounded: Always between 0 and 1 (when using log base 2)
        - Finite: No division-by-zero issues like raw KL divergence
        - Smooth: Small changes in distributions yield small changes in JS

        The JS divergence is calculated as:
            JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = (P + Q) / 2 is the average distribution.

        Args:
            p: First probability distribution as a dict mapping states to
                probabilities (values should sum to 1).
            q: Second probability distribution as a dict mapping states to
                probabilities (values should sum to 1).

        Returns:
            Jensen-Shannon divergence value between 0 and 1:
            - 0.0: Distributions are identical
            - ~0.1: Minor differences
            - ~0.3: Moderate differences
            - ~0.5: Major differences
            - 1.0: Completely different (disjoint support)

        Behavior:
            - Handles missing states by using smoothing constant (1e-12)
            - Considers all states present in either distribution
            - Uses log base 2, yielding results in [0, 1] range
            - Gracefully handles zero probabilities via smoothing

        Example:
            ```
            >>> stats = StateStats()
            >>> # Identical distributions
            >>> p1 = {"on": 0.7, "off": 0.3}
            >>> p2 = {"on": 0.7, "off": 0.3}
            >>> stats._js_divergence(p1, p2)
            0.0

            >>> # Moderate difference
            >>> p3 = {"on": 0.7, "off": 0.3}
            >>> p4 = {"on": 0.5, "off": 0.5}
            >>> stats._js_divergence(p3, p4)
            0.02...  # Small but measurable difference

            >>> # Large difference
            >>> p5 = {"on": 0.9, "off": 0.1}
            >>> p6 = {"on": 0.1, "off": 0.9}
            >>> stats._js_divergence(p5, p6)
            0.5...  # Substantial divergence

            >>> # Completely different states
            >>> p7 = {"on": 1.0}
            >>> p8 = {"off": 1.0}
            >>> stats._js_divergence(p7, p8)
            1.0  # Maximum divergence
            ```

        Note:
            The smoothing constant (1e-12) prevents log(0) errors and ensures
            numerical stability when comparing distributions with different
            state support.
        """
        states = set(p) | set(q)

        def kl(a: dict[State, float], b: dict[State, float]) -> float:
            """Calculates the Kullback-Leibler divergence between two distributions."""
            s = 0.0
            for st in states:
                pa = a.get(st, 1e-12)
                pb = b.get(st, 1e-12)
                s += pa * math.log2(pa / pb)
            return s

        m = {st: (p.get(st, 0.0) + q.get(st, 0.0)) / 2 for st in states}

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
