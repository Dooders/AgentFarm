from unittest.mock import Mock

import numpy as np
import pytest

from farm.core.action import (
    Action,
    ActionType,
    action_name_to_index,
    calculate_euclidean_distance,
    find_closest_entity,
    get_action_count,
    get_action_names,
    get_action_space,
    validate_action_result,
)


def test_action_space_helpers_complete_and_consistent():
    space = get_action_space()
    names = get_action_names()
    count = get_action_count()

    # Ensure names map matches enum ordering
    assert set(space.keys()) == set(names)
    assert count == len(ActionType)

    # Spot check a couple indices
    assert action_name_to_index("move") == ActionType.MOVE.value
    assert action_name_to_index("unknown") == ActionType.DEFEND.value


def test_distance_and_closest_entity():
    a = (0.0, 0.0)
    b = (3.0, 4.0)
    assert calculate_euclidean_distance(a, b) == 5.0

    agent = Mock()
    agent.position = (1.0, 1.0)

    e1 = Mock()
    e1.position = (2.0, 1.0)
    e2 = Mock()
    e2.position = (5.0, 5.0)

    closest, dist = find_closest_entity(agent, [e1, e2])
    assert closest is e1
    assert dist == pytest.approx(1.0)


def test_validate_action_result_move_success_and_inconsistency():
    agent = Mock()
    agent.agent_id = "a1"
    agent.position = (5, 6)

    # Consistent details
    res = {"success": True, "details": {"old_position": (1, 2), "new_position": (5, 6)}}
    out = validate_action_result(agent, "move", res)
    assert out["valid"] is True

    # Inconsistent current position
    agent.position = (0, 0)
    out = validate_action_result(agent, "move", res)
    assert out["valid"] is False
    assert any("position" in s for s in out["issues"])  # mismatch message


def test_validate_action_result_gather_and_share_consistency_checks():
    agent = Mock()
    agent.agent_id = "a1"
    agent.get_component = Mock(return_value=Mock(level=10))

    # Gather should increase resources by amount_gathered
    res = {"success": True, "details": {"agent_resources_before": 5, "amount_gathered": 3}}
    agent.get_component.return_value.level = 8  # Should be 5 + 3 = 8 OK
    out = validate_action_result(agent, "gather", res)
    assert out["valid"] is True

    agent.get_component.return_value.level = 7  # Mismatch
    out = validate_action_result(agent, "gather", res)
    assert out["valid"] is False

    # Share should decrease resources by amount_shared
    res = {"success": True, "details": {"agent_resources_before": 9, "amount_shared": 2}}
    agent.get_component.return_value.level = 7  # 9 - 2
    out = validate_action_result(agent, "share", res)
    assert out["valid"] is True

    agent.get_component.return_value.level = 6  # mismatch
    out = validate_action_result(agent, "share", res)
    assert out["valid"] is False


def test_validate_attack_defend_pass_paths():
    agent = Mock()
    agent.agent_id = "a1"
    agent.get_component = Mock(return_value=Mock(is_defending=True, level=8))
    # Attack with non-positive damage just raises a warning, still valid
    out = validate_action_result(agent, "attack", {"success": True, "details": {"damage_dealt": 0}})
    assert out["valid"] is True

    # Defend must set combat component is_defending True and apply cost correctly
    agent.get_component.return_value.level = 8
    res = {"success": True, "details": {"resources_before": 10, "cost": 2}}
    out = validate_action_result(agent, "defend", res)
    # resources match: 10-2 = 8, so validation should pass
    assert out["valid"] is True

    # Test actual resource mismatch scenario
    agent.get_component.return_value.level = 6  # Should be 8 (10-2), but is 6
    out = validate_action_result(agent, "defend", res)
    assert out["valid"] is False
    assert any("Resource mismatch" in issue for issue in out["issues"])

    # Pass requires details but otherwise minimal validation
    out = validate_action_result(agent, "pass", {"success": True, "details": {}})
    assert out["valid"] is True

