from unittest.mock import Mock

import numpy as np
import pytest

np.random.seed(42)  # Set seed for reproducibility in tests

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
    weighted_random_choice,
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
    agent.resource_level = 10

    # Gather should increase resources by amount_gathered
    res = {"success": True, "details": {"agent_resources_before": 5, "amount_gathered": 3}}
    agent.resource_level = 8  # Should be 5 + 3 = 8 OK
    out = validate_action_result(agent, "gather", res)
    assert out["valid"] is True

    agent.resource_level = 7  # Mismatch
    out = validate_action_result(agent, "gather", res)
    assert out["valid"] is False

    # Share should decrease resources by amount_shared
    res = {"success": True, "details": {"agent_resources_before": 9, "amount_shared": 2}}
    agent.resource_level = 7  # 9 - 2
    out = validate_action_result(agent, "share", res)
    assert out["valid"] is True

    agent.resource_level = 6  # mismatch
    out = validate_action_result(agent, "share", res)
    assert out["valid"] is False


def test_validate_attack_defend_pass_paths():
    agent = Mock()
    agent.agent_id = "a1"
    agent.is_defending = True
    # Attack with non-positive damage just raises a warning, still valid
    out = validate_action_result(agent, "attack", {"success": True, "details": {"damage_dealt": 0}})
    assert out["valid"] is True

    # Defend must set agent.is_defending True and apply cost correctly
    agent.resource_level = 8
    res = {"success": True, "details": {"resources_before": 10, "cost": 2}}
    out = validate_action_result(agent, "defend", res)
    # resources match: 10-2 = 8, so validation should pass
    assert out["valid"] is True

    # Test actual resource mismatch scenario
    agent.resource_level = 6  # Should be 8 (10-2), but is 6
    out = validate_action_result(agent, "defend", res)
    assert out["valid"] is False
    assert any("Resource mismatch" in issue for issue in out["issues"])

    # Pass requires details but otherwise minimal validation
    out = validate_action_result(agent, "pass", {"success": True, "details": {}})
    assert out["valid"] is True


class TestWeightedRandomChoice:
    """Tests for weighted_random_choice function."""

    def test_weighted_selection_respects_weights(self):
        """Test that higher weight actions are selected more often."""
        actions = [
            Action("move", 0.7, Mock()),
            Action("gather", 0.2, Mock()),
            Action("attack", 0.1, Mock()),
        ]

        # Run many selections and count frequencies
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)  # For reproducibility
        for _ in range(num_selections):
            selected = weighted_random_choice(actions)
            selection_counts[selected.name] += 1

        # move should be selected most often
        assert selection_counts["move"] > selection_counts["gather"]
        assert selection_counts["gather"] > selection_counts["attack"]
        # With weights 0.7, 0.2, 0.1, move should be ~70% of selections
        assert selection_counts["move"] > 600  # Allow some variance

    def test_weights_are_normalized(self):
        """Test that weights are normalized before selection."""
        # Unnormalized weights (don't sum to 1)
        actions = [
            Action("move", 4.0, Mock()),
            Action("gather", 2.0, Mock()),
            Action("attack", 1.0, Mock()),
        ]

        # Should still work - weights will be normalized internally
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            selected = weighted_random_choice(actions)
            selection_counts[selected.name] += 1

        # move should still be most common
        assert selection_counts["move"] > selection_counts["gather"]

    def test_zero_weights_fallback_to_uniform(self):
        """Test that zero weights fall back to uniform distribution."""
        actions = [
            Action("move", 0.0, Mock()),
            Action("gather", 0.0, Mock()),
            Action("attack", 0.0, Mock()),
        ]

        # Should not raise error, should use uniform distribution
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            selected = weighted_random_choice(actions)
            selection_counts[selected.name] += 1

        # All should be roughly equal (within reasonable variance)
        # Each should be roughly 1/3 of selections
        for count in selection_counts.values():
            assert 250 < count < 450  # Allow variance around 333

    def test_negative_weights_clamped_to_zero(self):
        """Test that negative weights are clamped to zero."""
        actions = [
            Action("move", 0.5, Mock()),
            Action("gather", -0.2, Mock()),  # Negative weight
            Action("attack", 0.5, Mock()),
        ]

        # Should not raise error, negative weight treated as zero
        selected = weighted_random_choice(actions)
        assert selected.name in ["move", "attack"]  # gather should never be selected

    def test_enabled_actions_subset(self):
        """Test weighted selection with enabled_actions parameter."""
        all_actions = [
            Action("move", 0.4, Mock()),
            Action("gather", 0.3, Mock()),
            Action("attack", 0.3, Mock()),
        ]

        # Only enable first two actions
        enabled = all_actions[:2]
        
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            selected = weighted_random_choice(all_actions, enabled)
            selection_counts[selected.name] += 1

        # attack should never be selected
        assert selection_counts["attack"] == 0
        # move and gather should be selected based on their relative weights
        assert selection_counts["move"] > selection_counts["gather"]

    def test_empty_actions_raises_error(self):
        """Test that empty actions list raises ValueError."""
        with pytest.raises(ValueError, match="No actions available"):
            weighted_random_choice([])

    def test_single_action_always_selected(self):
        """Test that single action is always selected."""
        action = Action("move", 1.0, Mock())
        
        # Should always return the same action
        for _ in range(10):
            selected = weighted_random_choice([action])
            assert selected.name == "move"

