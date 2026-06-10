"""Tests for Chromosome C (intrinsic reward / goal) genes.

Covers:
- the gene schema additions on the default chromosome,
- the ``reward_weights_from_chromosome`` / ``default_reward_weights`` helpers,
- the per-agent ``AgentCore._calculate_reward`` shaping that consumes them, and
- backward-compatibility (the default chromosome reproduces the historical
  reward formula exactly).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from farm.core.action import Action
from farm.core.agent.core import AgentCore
from farm.core.hyperparameter_chromosome import (
    INTRINSIC_REWARD_ACTION_BONUS_GENES,
    INTRINSIC_REWARD_GENE_NAMES,
    default_hyperparameter_chromosome,
    default_reward_weights,
    reward_weights_from_chromosome,
)


def _make_reward_agent(chromosome=None, alive: bool = True) -> AgentCore:
    """Skeletal AgentCore carrying just what ``_calculate_reward`` reads."""
    agent = object.__new__(AgentCore)
    agent.alive = alive
    agent.hyperparameter_chromosome = (
        chromosome if chromosome is not None else default_hyperparameter_chromosome()
    )
    return agent


def _state(resource_level: float, current_health: float):
    return SimpleNamespace(resource_level=resource_level, current_health=current_health)


def _action(name: str) -> Action:
    return Action(name, 1.0, lambda agent: None)


def test_all_reward_genes_present_on_default_chromosome():
    chromosome = default_hyperparameter_chromosome()
    for name in INTRINSIC_REWARD_GENE_NAMES:
        gene = chromosome.get_gene(name)
        assert gene is not None, f"missing gene {name}"
        assert gene.evolvable is True


def test_default_reward_weights_values():
    weights = default_reward_weights()
    assert weights["reward_resource_weight"] == 0.1
    assert weights["reward_health_weight"] == 0.5
    assert weights["reward_survival_weight"] == 0.1
    assert weights["reward_death_penalty"] == 10.0
    assert weights["reward_action_bonus"] == 0.05
    for action, gene in INTRINSIC_REWARD_ACTION_BONUS_GENES.items():
        assert weights[gene] == 0.0


def test_reward_weights_from_none_returns_defaults():
    assert reward_weights_from_chromosome(None) == default_reward_weights()


def test_default_chromosome_reproduces_historical_reward():
    """The default goal must equal the pre-Chromosome-C formula exactly."""
    agent = _make_reward_agent()
    pre = _state(resource_level=10.0, current_health=80.0)
    post = _state(resource_level=15.0, current_health=90.0)

    reward = agent._calculate_reward(pre, post, _action("gather"))

    expected = (15.0 - 10.0) * 0.1 + (90.0 - 80.0) * 0.5 + 0.1 + 0.05
    assert reward == pytest.approx(expected)


def test_death_uses_death_penalty_gene():
    agent = _make_reward_agent(alive=False)
    pre = _state(resource_level=10.0, current_health=10.0)
    post = _state(resource_level=10.0, current_health=0.0)

    reward = agent._calculate_reward(pre, post, _action("pass"))

    # resource_delta 0 + health_delta (-10 * 0.5) + (-death_penalty 10.0)
    assert reward == pytest.approx(-5.0 - 10.0)


def test_per_action_bonus_changes_reward():
    """A prosocial goal (share bonus) rewards sharing without changing outcome."""
    base = default_hyperparameter_chromosome()
    prosocial = base.with_overrides({"reward_share_bonus": 1.0})

    neutral_agent = _make_reward_agent(base)
    prosocial_agent = _make_reward_agent(prosocial)

    pre = _state(resource_level=10.0, current_health=50.0)
    post = _state(resource_level=8.0, current_health=50.0)  # sharing costs resources
    share = _action("share")

    neutral_reward = neutral_agent._calculate_reward(pre, post, share)
    prosocial_reward = prosocial_agent._calculate_reward(pre, post, share)

    assert prosocial_reward == pytest.approx(neutral_reward + 1.0)


def test_distinct_goals_produce_distinct_rewards():
    """Two agents with different goals score the same transition differently."""
    hoarder = default_hyperparameter_chromosome().with_overrides(
        {"reward_resource_weight": 2.0, "reward_health_weight": 0.0}
    )
    medic = default_hyperparameter_chromosome().with_overrides(
        {"reward_resource_weight": 0.0, "reward_health_weight": 2.0}
    )
    hoarder_agent = _make_reward_agent(hoarder)
    medic_agent = _make_reward_agent(medic)

    # A transition that gains more resources than health, so the two goals
    # disagree on its value.
    pre = _state(resource_level=0.0, current_health=0.0)
    post = _state(resource_level=10.0, current_health=4.0)
    action = _action("gather")

    hoarder_reward = hoarder_agent._calculate_reward(pre, post, action)
    medic_reward = medic_agent._calculate_reward(pre, post, action)

    # Hoarder values the larger resource gain; medic only the smaller health gain.
    assert hoarder_reward == pytest.approx(10.0 * 2.0 + 0.1 + 0.05)
    assert medic_reward == pytest.approx(4.0 * 2.0 + 0.1 + 0.05)
    assert hoarder_reward > medic_reward


def test_reward_weight_cache_refreshes_on_chromosome_swap():
    agent = _make_reward_agent()
    pre = _state(0.0, 0.0)
    post = _state(10.0, 0.0)
    action = _action("gather")

    first = agent._calculate_reward(pre, post, action)
    # Swap in a chromosome that values resources 10x more.
    agent.hyperparameter_chromosome = default_hyperparameter_chromosome().with_overrides(
        {"reward_resource_weight": 1.0}
    )
    second = agent._calculate_reward(pre, post, action)

    assert second > first
