"""Unit tests for the simple gather action.

This module tests the simple gathering functionality that finds the nearest resource
and gathers from it without using DQN.
"""

import unittest
from unittest.mock import Mock

from farm.core.action import gather_action


class TestSimpleGatherAction(unittest.TestCase):
    """Test cases for the simple gather action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.resource_level = 5
        self.agent.config = Mock()
        self.agent.config.gathering_range = 30
        self.agent.config.max_amount = 10
        self.agent.total_reward = 0.0

        self.environment = Mock()
        self.agent.environment = self.environment

    def test_gather_action_no_config(self):
        """Test gather action when agent has no config."""
        self.agent.config = None

        gather_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_gather_action_no_nearby_resources(self):
        """Test gather action when no resources are nearby."""
        self.environment.get_nearby_resources.return_value = []

        gather_action(self.agent)

        # Should return early without gathering
        self.environment.get_nearby_resources.assert_called_once_with((0, 0), 30)

    def test_gather_action_depleted_resources_only(self):
        """Test gather action when only depleted resources are nearby."""
        # Create a depleted resource
        resource = Mock()
        resource.amount = 0
        resource.is_depleted.return_value = True

        self.environment.get_nearby_resources.return_value = [resource]

        gather_action(self.agent)

        # Should return early without gathering
        resource.consume.assert_not_called()

    def test_gather_action_successful_gathering(self):
        """Test successful gathering from nearest resource."""
        # Create a valid resource
        resource = Mock()
        resource.position = (5, 0)  # 5 units away
        resource.amount = 8
        resource.is_depleted.return_value = False
        resource.consume.return_value = 8
        resource.resource_id = "test_resource"

        # Create a farther resource
        far_resource = Mock()
        far_resource.position = (20, 0)  # 20 units away
        far_resource.amount = 5
        far_resource.is_depleted.return_value = False

        self.environment.get_nearby_resources.return_value = [far_resource, resource]
        self.environment.log_interaction_edge = Mock()

        gather_action(self.agent)

        # Should gather from the closer resource
        resource.consume.assert_called_once_with(8)
        far_resource.consume.assert_not_called()

        # Should increase agent's resources
        self.assertEqual(self.agent.resource_level, 5 + 8)  # 5 initial + 8 gathered

        # Should log the interaction
        self.environment.log_interaction_edge.assert_called_once()

    def test_gather_action_partial_gathering(self):
        """Test gathering limited by max_amount config."""
        # Create a resource with more than max_amount
        resource = Mock()
        resource.position = (3, 0)
        resource.amount = 15  # More than max_amount (10)
        resource.is_depleted.return_value = False
        resource.consume.return_value = 10  # Limited by max_amount
        resource.resource_id = "test_resource"

        self.agent.config.max_amount = 10

        self.environment.get_nearby_resources.return_value = [resource]
        self.environment.log_interaction_edge = Mock()

        gather_action(self.agent)

        # Should gather only up to max_amount
        resource.consume.assert_called_once_with(10)

    def test_gather_action_distance_calculation(self):
        """Test that gathering selects the closest resource."""
        # Create resources at different distances
        close_resource = Mock()
        close_resource.position = (2, 0)  # 2 units away
        close_resource.amount = 5
        close_resource.is_depleted.return_value = False
        close_resource.consume.return_value = 5
        close_resource.resource_id = "close_resource"

        far_resource = Mock()
        far_resource.position = (10, 0)  # 10 units away
        far_resource.amount = 10
        far_resource.is_depleted.return_value = False
        far_resource.consume.return_value = 10
        far_resource.resource_id = "far_resource"

        self.environment.get_nearby_resources.return_value = [
            far_resource,
            close_resource,
        ]
        self.environment.log_interaction_edge = Mock()

        gather_action(self.agent)

        # Should gather from the closer resource
        close_resource.consume.assert_called_once_with(5)
        far_resource.consume.assert_not_called()

    def test_gather_action_reward_calculation(self):
        """Test that gathering provides appropriate rewards."""
        resource = Mock()
        resource.position = (4, 0)
        resource.amount = 6
        resource.is_depleted.return_value = False
        resource.consume.return_value = 6
        resource.resource_id = "test_resource"

        initial_reward = self.agent.total_reward

        self.environment.get_nearby_resources.return_value = [resource]
        self.environment.log_interaction_edge = Mock()

        gather_action(self.agent)

        # Should increase total reward (6 gathered * 0.1 = 0.6 reward)
        expected_reward = initial_reward + 0.6
        self.assertEqual(self.agent.total_reward, expected_reward)


if __name__ == "__main__":
    unittest.main()
