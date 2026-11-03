"""Engine-level regression tests for movement semantics."""

import pytest

from farm.config import SimulationConfig
from farm.core.action import ActionType, move_action
from farm.core.agent import AgentComponentConfig, AgentFactory, AgentServices
from farm.core.environment import Environment
from farm.core.services.implementations import EnvironmentValidationService


@pytest.fixture
def moving_environment():
    config = SimulationConfig()
    config.agent_behavior.max_movement = 3
    environment = Environment(
        width=30,
        height=30,
        resource_distribution={"amount": 0},
        config=config,
        db_path=None,
    )
    services = AgentServices(spatial_service=environment.spatial_service)
    agent = AgentFactory(services).create_default_agent(
        "moving",
        (10.0, 10.0),
        config=AgentComponentConfig.from_simulation_config(config),
        environment=environment,
    )
    environment.add_agent(agent)
    try:
        yield environment, agent
    finally:
        environment.cleanup()


def test_agents_move(moving_environment, monkeypatch):
    environment, agent = moving_environment
    monkeypatch.setattr("farm.core.action.random.choice", lambda _: (1, 0))
    old_position = agent.position
    assert agent.config.movement.max_movement == 3
    assert isinstance(agent.services.validation_service, EnvironmentValidationService)
    assert agent.get_component("movement").validation_service is agent.validation_service

    environment.reset()
    environment.step(int(ActionType.MOVE))

    assert agent.position == (old_position[0] + agent.config.movement.max_movement, old_position[1])
    assert agent.get_component("movement").position == agent.position
    assert agent in environment.get_nearby_agents(agent.position, 0.1)
    assert agent not in environment.get_nearby_agents(old_position, 0.1)


def test_move_rejects_out_of_bounds_target(moving_environment, monkeypatch):
    _, agent = moving_environment
    agent.position = (0.0, 0.0)
    monkeypatch.setattr("farm.core.action.random.choice", lambda _: (-1, 0))

    result = move_action(agent)

    assert not result["success"]
    assert result["details"]["validation_service_available"]
    assert agent.position == agent.get_component("movement").position == (0.0, 0.0)


def test_add_agent_preserves_explicit_validation(moving_environment):
    environment, _ = moving_environment
    validation = EnvironmentValidationService(environment)
    services = AgentServices(
        spatial_service=environment.spatial_service,
        validation_service=validation,
    )
    agent = AgentFactory(services).create_default_agent("explicit", (5.0, 5.0), environment=environment)

    environment.add_agent(agent)

    assert agent.validation_service is validation
    assert agent.get_component("movement").validation_service is validation
