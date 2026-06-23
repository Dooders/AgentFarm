"""Population cap enforcement for reproduction (issue #916)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from farm.config import SimulationConfig
from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.decision.config import DecisionConfig
from farm.core.hyperparameter_chromosome import chromosome_from_learning_config
from farm.core.inheritance_telemetry import InheritanceTelemetry
from farm.core.population import get_population_cap_status, is_population_at_cap
from farm.core.simulation import run_simulation


def _reproduction_friendly_config(max_population: int = 8) -> SimulationConfig:
    """Minimal config that encourages reproduction without blowing up runtime."""
    cfg = SimulationConfig.from_centralized_config(environment="testing")
    cfg.simulation_steps = 80
    cfg.max_steps = 80
    cfg.population.system_agents = 2
    cfg.population.independent_agents = 2
    cfg.population.control_agents = 0
    cfg.population.order_agents = 0
    cfg.population.chaos_agents = 0
    cfg.population.max_population = max_population
    cfg.resources.initial_resources = 400
    cfg.agent_behavior.initial_resource_level = 40.0
    cfg.agent_behavior.offspring_cost = 1
    cfg.agent_behavior.min_reproduction_resources = 1
    cfg.agent_behavior.reproduction_chance = 1.0
    cfg.agent_behavior.offspring_initial_resources = 5
    cfg.environment.width = 20
    cfg.environment.height = 20
    cfg.database.use_in_memory_db = True
    cfg.database.persist_db_on_completion = False
    return cfg


def _build_parent_agent_for_cap_test() -> AgentCore:
    agent = object.__new__(AgentCore)
    agent.agent_id = "parent_1"
    agent.agent_type = "system"
    agent.generation = 0
    agent.services = Mock()
    agent.environment = SimpleNamespace(
        agents=["existing"],
        config=SimpleNamespace(population=SimpleNamespace(max_population=1)),
        get_next_agent_id=lambda: "child_1",
        intrinsic_evolution_policy=None,
        intrinsic_evolution_rng=None,
        inheritance_telemetry=InheritanceTelemetry(),
    )
    agent.state = Mock()
    agent.state.position = (1.0, 2.0)
    agent.state._state = Mock()
    agent.state._state.model_copy.return_value = Mock()

    config = AgentComponentConfig.default()
    config.decision = DecisionConfig(learning_rate=0.01)
    agent.config = config

    resource_component = Mock()
    reproduction_component = Mock()
    reproduction_component.can_reproduce.return_value = True
    reproduction_component.config.offspring_cost = 1.0
    reproduction_component.config.offspring_initial_resources = 5.0

    agent._components = {
        "resource": resource_component,
        "reproduction": reproduction_component,
    }
    agent.hyperparameter_chromosome = chromosome_from_learning_config(agent.config.decision)
    agent.environment.add_agent = Mock()
    return agent


def test_get_population_cap_status_accepts_float_limits():
    env = SimpleNamespace(
        agents=["a", "b"],
        config=SimpleNamespace(population=SimpleNamespace(max_population=2.0)),
    )
    assert get_population_cap_status(env) == (True, 2, 2.0)
    assert is_population_at_cap(env) is True


def test_agent_core_reproduce_blocks_at_population_cap_without_adding_offspring():
    """Direct AgentCore.reproduce calls must respect max_population."""
    parent = _build_parent_agent_for_cap_test()

    success = AgentCore.reproduce(parent)

    assert success is False
    parent.environment.add_agent.assert_not_called()
    parent.get_component("resource").remove.assert_not_called()


def test_agent_core_reproduce_bypasses_cap_when_below_limit():
    parent = _build_parent_agent_for_cap_test()
    parent.environment.agents = []
    parent.environment.config.population.max_population = 2

    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    parent.environment.add_agent.assert_called_once_with(offspring, flush_immediately=True)


@pytest.mark.integration
def test_run_simulation_never_exceeds_max_population(tmp_path):
    """Acceptance criterion for #916: alive agents never exceed max_population."""
    max_pop = 8
    cfg = _reproduction_friendly_config(max_population=max_pop)
    initial_population = (
        cfg.population.system_agents
        + cfg.population.independent_agents
        + cfg.population.control_agents
        + cfg.population.order_agents
        + cfg.population.chaos_agents
    )
    peak_alive = {"count": initial_population}

    def on_step_end(environment, step):
        alive = len(environment.agents)
        peak_alive["count"] = max(peak_alive["count"], alive)
        assert alive <= max_pop, f"step {step}: alive={alive} exceeds max_population={max_pop}"

    env = run_simulation(
        num_steps=cfg.max_steps,
        config=cfg,
        path=str(tmp_path),
        save_config=False,
        seed=42,
        on_step_end=on_step_end,
    )

    assert len(env.agents) <= max_pop
    assert peak_alive["count"] <= max_pop
    assert peak_alive["count"] > initial_population, "expected reproduction to grow toward the cap"
