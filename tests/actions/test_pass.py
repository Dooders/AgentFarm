"""Unit tests for the simple pass action.

This module tests the simple pass functionality that allows agents
to do nothing for a turn, which can be useful for strategic waiting.
"""

import unittest
from unittest.mock import Mock

from farm.core.action import pass_action


class TestSimplePassAction(unittest.TestCase):
    """Test cases for the simple pass action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.total_reward = 0.0

        self.agent.config = Mock()
        self.environment = Mock()
        self.agent.environment = self.environment

    def test_pass_action_no_config(self):
        """Test pass action when agent has no config."""
        self.agent.config = None

        pass_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_pass_action_successful_pass(self):
        """Test successful pass action."""
        initial_reward = self.agent.total_reward

        self.environment.log_interaction_edge = Mock()

        pass_action(self.agent)

        # Should get small reward for passing
        self.assertEqual(self.agent.total_reward, initial_reward + 0.01)

        # Should log the interaction
        self.environment.log_interaction_edge.assert_called_once()

    def test_pass_action_with_zero_initial_reward(self):
        """Test pass action when agent starts with zero reward."""
        self.agent.total_reward = 0.0

        pass_action(self.agent)

        # Should have small positive reward
        self.assertEqual(self.agent.total_reward, 0.01)

    def test_pass_action_with_existing_reward(self):
        """Test pass action when agent already has reward."""
        self.agent.total_reward = 5.0

        pass_action(self.agent)

        # Should add to existing reward
        self.assertEqual(self.agent.total_reward, 5.01)

    def test_pass_action_no_environment_logging(self):
        """Test pass action when environment doesn't support logging."""
        # Mock environment that raises exception on log_interaction_edge
        self.environment.log_interaction_edge.side_effect = Exception("No logging")

        # Should not raise exception
        pass_action(self.agent)

        # Should still get reward
        self.assertEqual(self.agent.total_reward, 0.01)


if __name__ == "__main__":
    unittest.main()
