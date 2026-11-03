"""Regression tests for component-backed random-cardinal movement."""

from unittest.mock import Mock

import pytest

from farm.core.action import move_action, validate_action_result
from farm.core.agent import AgentComponentConfig, AgentFactory, AgentServices
from farm.core.agent.config.component_configs import MovementConfig


@pytest.fixture
def agent():
    services = AgentServices(
        spatial_service=Mock(),
        validation_service=Mock(is_valid_position=Mock(return_value=True)),
    )
    config = AgentComponentConfig(movement=MovementConfig(max_movement=2.5))
    return AgentFactory(services).create_default_agent("moving", (10.0, 10.0), config=config)


@pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1)])
@pytest.mark.parametrize("distance", [2.5, 0.1, 0.3])
def test_move_action_uses_component_distance_and_updates_state(agent, monkeypatch, direction, distance):
    movement = agent.get_component("movement")
    movement.config = MovementConfig(max_movement=distance)
    monkeypatch.setattr("farm.core.action.random.choice", lambda _: direction)
    agent._obs_cache_valid = True

    result = move_action(agent)

    expected = (10.0 + direction[0] * distance, 10.0 + direction[1] * distance)
    assert result["success"], result
    assert agent.position == movement.position == agent.state.position == expected
    assert result["details"]["old_position"] == (10.0, 10.0)
    assert result["details"]["new_position"] == expected
    assert result["details"]["distance"] == distance
    assert result["details"]["position_changed"]
    assert validate_action_result(agent, "move", result)["valid"]
    assert not agent._obs_cache_valid
    agent.validation_service.is_valid_position.assert_called_once_with(expected)
    agent.spatial_service.mark_positions_dirty.assert_called_once_with()


def test_rejected_move_preserves_state_and_spatial_index(agent, monkeypatch):
    agent.validation_service.is_valid_position.return_value = False
    monkeypatch.setattr("farm.core.action.random.choice", lambda _: (1, 0))

    result = move_action(agent)

    assert not result["success"]
    assert result["details"]["validation_service_available"]
    assert agent.position == agent.get_component("movement").position == (10.0, 10.0)
    agent.spatial_service.mark_positions_dirty.assert_not_called()


def test_rejected_position_assignment_preserves_state(agent):
    agent.validation_service.is_valid_position.return_value = False

    agent.position = (12.0, 10.0)

    assert agent.position == agent.get_component("movement").position == (10.0, 10.0)
    agent.spatial_service.mark_positions_dirty.assert_not_called()


def test_move_to_enforces_component_distance_limit(agent):
    assert not agent.move_to((13.0, 10.0))
    assert agent.position == agent.get_component("movement").position == (10.0, 10.0)
    agent.validation_service.is_valid_position.assert_not_called()


def test_move_without_validation_service_uses_component(agent, monkeypatch):
    agent.validation_service = None
    monkeypatch.setattr("farm.core.action.random.choice", lambda _: (1, 0))

    assert move_action(agent)["success"]
    assert agent.position == (12.5, 10.0)


def test_move_without_component_fails_explicitly(agent):
    agent._components.pop("movement")

    result = move_action(agent)

    assert not result["success"]
    assert result["error"] == "Movement component unavailable"
    assert agent.position == (10.0, 10.0)
