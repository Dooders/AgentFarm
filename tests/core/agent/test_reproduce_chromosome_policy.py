"""Tests for the intrinsic-evolution policy path in `AgentCore.reproduce`.

Covers chromosome derivation logic in isolation (without spinning up a full
simulation): the no-policy passthrough, the mutation path, and the optional
crossover path with co-parent selection.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.decision.config import DecisionConfig
from farm.core.hyperparameter_chromosome import (
    CrossoverMode,
    chromosome_from_learning_config,
)
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy


def _make_agent(
    *,
    agent_id: str = "agent_1",
    agent_type: str = "system",
    learning_rate: float = 0.01,
    position: tuple = (0.0, 0.0),
    alive: bool = True,
) -> AgentCore:
    """Skeletal AgentCore stand-in carrying just what `_derive_child_chromosome` reads."""
    agent = object.__new__(AgentCore)
    agent.agent_id = agent_id
    agent.agent_type = agent_type
    agent.alive = alive
    agent.environment = None
    agent.state = SimpleNamespace(position=position)
    config = AgentComponentConfig.default()
    config.decision = DecisionConfig(learning_rate=learning_rate)
    agent.config = config
    agent.hyperparameter_chromosome = chromosome_from_learning_config(agent.config.decision)
    return agent


def _make_environment(agents: List[AgentCore], policy: Optional[IntrinsicEvolutionPolicy], rng):
    """Minimal environment object exposing what reproduction reads from us."""
    return SimpleNamespace(
        alive_agent_objects=[a for a in agents if a.alive],
        intrinsic_evolution_policy=policy,
        intrinsic_evolution_rng=rng,
    )


def test_derive_child_no_policy_returns_deep_copy():
    parent = _make_agent(learning_rate=0.05)
    parent.environment = _make_environment([parent], policy=None, rng=None)
    child = parent._derive_child_chromosome(parent.hyperparameter_chromosome)
    assert child is not parent.hyperparameter_chromosome
    assert child.get_value("learning_rate") == 0.05


def test_derive_child_disabled_policy_returns_deep_copy():
    parent = _make_agent(learning_rate=0.05)
    policy = IntrinsicEvolutionPolicy(enabled=False)
    parent.environment = _make_environment([parent], policy=policy, rng=random.Random(0))
    child = parent._derive_child_chromosome(parent.hyperparameter_chromosome)
    assert child.get_value("learning_rate") == 0.05


def test_derive_child_with_policy_mutates():
    parent = _make_agent(learning_rate=0.01)
    policy = IntrinsicEvolutionPolicy(mutation_rate=1.0, mutation_scale=0.2)
    parent.environment = _make_environment([parent], policy=policy, rng=None)

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=0.0), patch(
        "farm.core.hyperparameter_chromosome.random.gauss", return_value=0.002
    ):
        child = parent._derive_child_chromosome(parent.hyperparameter_chromosome)

    assert child.get_value("learning_rate") != parent.hyperparameter_chromosome.get_value(
        "learning_rate"
    )


def test_select_coparent_returns_none_when_alone():
    parent = _make_agent()
    policy = IntrinsicEvolutionPolicy(crossover_enabled=True)
    parent.environment = _make_environment([parent], policy=policy, rng=random.Random(0))
    assert parent._select_coparent(policy, random.Random(0)) is None


def test_select_coparent_skips_dead_and_other_types():
    parent = _make_agent(agent_id="p", position=(0.0, 0.0))
    dead = _make_agent(agent_id="d", position=(1.0, 0.0), alive=False)
    other_type = _make_agent(agent_id="o", agent_type="independent", position=(0.5, 0.0))
    sibling = _make_agent(agent_id="s", position=(2.0, 0.0))
    policy = IntrinsicEvolutionPolicy(crossover_enabled=True)
    parent.environment = _make_environment(
        [parent, dead, other_type, sibling], policy=policy, rng=random.Random(0)
    )
    chosen = parent._select_coparent(policy, random.Random(0))
    assert chosen is sibling


def test_select_coparent_respects_radius():
    parent = _make_agent(agent_id="p", position=(0.0, 0.0))
    near = _make_agent(agent_id="near", position=(1.0, 0.0))
    far = _make_agent(agent_id="far", position=(10.0, 0.0))
    policy = IntrinsicEvolutionPolicy(crossover_enabled=True, coparent_max_radius=5.0)
    parent.environment = _make_environment([parent, near, far], policy=policy, rng=random.Random(0))
    assert parent._select_coparent(policy, random.Random(0)) is near


def test_select_coparent_nearest_strategy():
    parent = _make_agent(agent_id="p", position=(0.0, 0.0))
    closer = _make_agent(agent_id="closer", position=(1.0, 0.0))
    farther = _make_agent(agent_id="farther", position=(5.0, 0.0))
    policy = IntrinsicEvolutionPolicy(
        crossover_enabled=True, coparent_strategy="nearest_alive_same_type"
    )
    parent.environment = _make_environment(
        [parent, closer, farther], policy=policy, rng=random.Random(0)
    )
    assert parent._select_coparent(policy, random.Random(0)) is closer


def test_select_coparent_random_strategy_uses_rng():
    parent = _make_agent(agent_id="p", position=(0.0, 0.0))
    options = [_make_agent(agent_id=f"o{i}", position=(float(i), 0.0)) for i in range(5)]
    policy = IntrinsicEvolutionPolicy(
        crossover_enabled=True, coparent_strategy="random_alive_same_type"
    )
    parent.environment = _make_environment([parent, *options], policy=policy, rng=random.Random(0))
    rng = random.Random(123)
    pick = parent._select_coparent(policy, rng)
    assert pick in options


def test_derive_child_with_crossover_uses_coparent():
    """When crossover is enabled, the co-parent's chromosome is consulted."""
    parent = _make_agent(agent_id="p", learning_rate=0.01, position=(0.0, 0.0))
    coparent = _make_agent(agent_id="c", learning_rate=0.05, position=(1.0, 0.0))
    policy = IntrinsicEvolutionPolicy(
        crossover_enabled=True,
        crossover_mode=CrossoverMode.UNIFORM,
        mutation_rate=0.0,
    )
    parent.environment = _make_environment(
        [parent, coparent], policy=policy, rng=random.Random(42)
    )
    rng_seed = random.Random(7)
    parent.environment.intrinsic_evolution_rng = rng_seed
    child = parent._derive_child_chromosome(parent.hyperparameter_chromosome)
    lr = child.get_value("learning_rate")
    assert math.isclose(lr, 0.01) or math.isclose(lr, 0.05), (
        "uniform crossover should pick learning_rate from one of the two parents"
    )
