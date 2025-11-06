"""Extended tests for action system covering execution and validation."""

import unittest
from unittest.mock import Mock

from farm.core.action import (
    ActionType,
    calculate_euclidean_distance,
    check_resource_requirement,
    validate_action_result,
)


class TestActionExtended(unittest.TestCase):
    """Extended tests for action system."""

    def test_calculate_euclidean_distance(self):
        """Test distance calculation."""
        pos1 = (0, 0)
        pos2 = (3, 4)

        distance = calculate_euclidean_distance(pos1, pos2)

        self.assertEqual(distance, 5.0)

    def test_check_resource_requirement(self):
        """Test resource requirement checking."""
        agent = Mock()
        agent.resource_level = 50.0  # Note: uses resource_level, not resources
        agent.agent_id = "test_agent"

        # Sufficient resources
        result = check_resource_requirement(agent, 30.0, "move")
        self.assertTrue(result)

        # Insufficient resources
        result = check_resource_requirement(agent, 100.0, "move")
        self.assertFalse(result)

    def test_validate_action_result(self):
        """Test action result validation."""
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.position = (10.0, 20.0)

        result = {"success": True, "reward": 0.5, "details": {"new_position": (11.0, 21.0)}}
        validated = validate_action_result(agent, "move", result)

        self.assertIn("valid", validated)
        self.assertIn("issues", validated)

    def test_action_type_enum(self):
        """Test ActionType enum."""
        self.assertEqual(ActionType.MOVE, 4)
        self.assertEqual(ActionType.DEFEND, 0)
        self.assertIsInstance(ActionType.MOVE, int)


if __name__ == "__main__":
    unittest.main()

