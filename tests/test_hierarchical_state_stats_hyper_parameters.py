"""
Unit tests for HierarchicalStateStatsHyperParameters.

Tests cover serialization via `to_dict` and reconstruction via `from_dict`.
"""


from custom_components.discrete_state_forecaster.model.hyper_parameters import (
    HyperParameters,
)
from custom_components.discrete_state_forecaster.model.statistics.hierarchical_state_stats_hyper_parameters import (
    HierarchicalStateStatsHyperParameters,
)


def test_to_dict_returns_min_support() -> None:
    base = HyperParameters(half_life=50.0, min_prune_interval=10.0, prune_enabled=True, persistence_strength=0.5)
    hp = HierarchicalStateStatsHyperParameters(base, min_support_factor=0.5)
    data = hp.to_dict()
    assert isinstance(data, dict)
    assert data.get("min_support") == 25.0


def test_from_dict_restores_from_min_support() -> None:
    base = HyperParameters(half_life=40.0, min_prune_interval=5.0, prune_enabled=False, persistence_strength=0.1)
    payload = {"min_support": 10.0}
    hp = HierarchicalStateStatsHyperParameters.from_dict(payload, hyper_parameters=base)
    # min_support should match payload value
    assert hp.min_support == 10.0
