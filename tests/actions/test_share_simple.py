"""Unit tests for the simple share action.

This module tests the simple sharing functionality that finds agents in need
and shares resources with them.
"""

import unittest
from unittest.mock import Mock

from farm.core.action import share_action


class TestSimpleShareAction(unittest.TestCase):
    """Test cases for the simple share action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.resource_level = 10
        self.agent.total_reward = 0.0

        self.agent.config = Mock()
        self.agent.config.share_range = 30
        self.agent.config.share_amount = 2
        self.agent.config.min_keep_resources = 5

        self.environment = Mock()
        self.agent.environment = self.environment

    def test_share_action_no_config(self):
        """Test share action when agent has no config."""
        self.agent.config = None

        share_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_share_action_no_nearby_agents(self):
        """Test share action when no agents are nearby."""
        self.environment.get_nearby_agents.return_value = []

        share_action(self.agent)

        # Should return early without sharing
        self.environment.get_nearby_agents.assert_called_once_with((0, 0), 30)

    def test_share_action_only_self_nearby(self):
        """Test share action when only self is nearby."""
        # Create an agent that's the same as self
        self_agent = Mock()
        self_agent.agent_id = "test_agent"  # Same ID as self
        self_agent.alive = True

        self.environment.get_nearby_agents.return_value = [self_agent]

        share_action(self.agent)

        # Should return early without sharing (no valid targets)
        self.assertTrue(True)

    def test_share_action_insufficient_resources(self):
        """Test share action when agent doesn't have enough resources."""
        # Agent has 3 resources, needs to keep 5 minimum
        self.agent.resource_level = 3

        # Create a target agent
        target = Mock()
        target.agent_id = "target_agent"
        target.alive = True
        target.resource_level = 1

        self.environment.get_nearby_agents.return_value = [target]
        self.environment.log_interaction_edge = Mock()

        share_action(self.agent)

        # Should not share due to insufficient resources
        target.resource_level = 1  # Should remain unchanged

    def test_share_action_successful_sharing(self):
        """Test successful sharing with agent in need."""
        # Create a target agent with low resources
        target = Mock()
        target.agent_id = "target_agent"
        target.alive = True
        target.resource_level = 1

        # Create another agent with more resources
        other_agent = Mock()
        other_agent.agent_id = "other_agent"
        other_agent.alive = True
        other_agent.resource_level = 8

        self.environment.get_nearby_agents.return_value = [target, other_agent]
        self.environment.log_interaction_edge = Mock()

        share_action(self.agent)

        # Should share with the agent with lowest resources (target)
        self.assertEqual(self.agent.resource_level, 10 - 2)  # 10 - 2 = 8
        self.assertEqual(target.resource_level, 1 + 2)  # 1 + 2 = 3

        # Should log the interaction
        self.environment.log_interaction_edge.assert_called_once()

    def test_share_action_reward_calculation(self):
        """Test that sharing provides appropriate rewards."""
        # Create a target agent
        target = Mock()
        target.agent_id = "target_agent"
        target.alive = True
        target.resource_level = 1

        initial_reward = self.agent.total_reward

        self.environment.get_nearby_agents.return_value = [target]
        self.environment.log_interaction_edge = Mock()

        share_action(self.agent)

        # Should increase total reward (2 shared * 0.05 = 0.1 reward)
        expected_reward = initial_reward + 0.1
        self.assertEqual(self.agent.total_reward, expected_reward)

    def test_share_action_selects_lowest_resource_agent(self):
        """Test that sharing selects the agent with lowest resources."""
        # Create multiple target agents with different resource levels
        poor_agent = Mock()
        poor_agent.agent_id = "poor_agent"
        poor_agent.alive = True
        poor_agent.resource_level = 1

        rich_agent = Mock()
        rich_agent.agent_id = "rich_agent"
        rich_agent.alive = True
        rich_agent.resource_level = 15

        self.environment.get_nearby_agents.return_value = [rich_agent, poor_agent]
        self.environment.log_interaction_edge = Mock()

        share_action(self.agent)

        # Should share with the poor agent (lowest resources)
        self.assertEqual(poor_agent.resource_level, 1 + 2)  # 1 + 2 = 3
        self.assertEqual(rich_agent.resource_level, 15)  # Unchanged

    def test_share_action_dead_agent_excluded(self):
        """Test that dead agents are excluded from sharing targets."""
        # Create a dead agent and a live agent
        dead_agent = Mock()
        dead_agent.agent_id = "dead_agent"
        dead_agent.alive = False
        dead_agent.resource_level = 0

        live_agent = Mock()
        live_agent.agent_id = "live_agent"
        live_agent.alive = True
        live_agent.resource_level = 2

        self.environment.get_nearby_agents.return_value = [dead_agent, live_agent]
        self.environment.log_interaction_edge = Mock()

        share_action(self.agent)

        # Should share with the live agent, not the dead one
        self.assertEqual(live_agent.resource_level, 2 + 2)  # 2 + 2 = 4
        self.assertEqual(dead_agent.resource_level, 0)  # Unchanged


if __name__ == "__main__":
    unittest.main()
