"""Unit tests for the simple defend action.

This module tests the simple defensive functionality that allows agents
to enter a defensive stance with healing and damage reduction benefits.
"""

import unittest
from unittest.mock import Mock

from farm.core.action import defend_action


class TestSimpleDefendAction(unittest.TestCase):
    """Test cases for the simple defend action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.resource_level = 10
        self.agent.total_reward = 0.0
        self.agent.current_health = 8
        self.agent.starting_health = 10
        self.agent.is_defending = False

        self.agent.config = Mock()
        self.agent.config.defense_duration = 3
        self.agent.config.defense_healing = 2
        self.agent.config.defense_cost = 1

        self.environment = Mock()
        self.agent.environment = self.environment

    def test_defend_action_no_config(self):
        """Test defend action when agent has no config."""
        self.agent.config = None

        defend_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_defend_action_insufficient_resources(self):
        """Test defend action when agent doesn't have enough resources."""
        self.agent.resource_level = 0  # No resources
        self.agent.config.defense_cost = 2  # Requires 2 resources

        defend_action(self.agent)

        # Should not enter defensive state due to insufficient resources
        self.assertFalse(
            hasattr(self.agent, "is_defending") and self.agent.is_defending
        )

    def test_defend_action_successful_defense(self):
        """Test successful defense action."""
        initial_resources = self.agent.resource_level
        initial_health = self.agent.current_health
        initial_reward = self.agent.total_reward

        self.environment.log_interaction_edge = Mock()

        defend_action(self.agent)

        # Should enter defensive state
        self.assertTrue(self.agent.is_defending)
        self.assertEqual(self.agent.defense_timer, 3)

        # Should pay the defense cost
        self.assertEqual(self.agent.resource_level, initial_resources - 1)

        # Should heal
        self.assertEqual(self.agent.current_health, initial_health + 2)

        # Should get reward
        self.assertEqual(self.agent.total_reward, initial_reward + 0.02)

        # Should log the interaction
        self.environment.log_interaction_edge.assert_called_once()

    def test_defend_action_no_healing_needed(self):
        """Test defend action when agent is at full health."""
        self.agent.current_health = 10  # Already at max health
        self.agent.starting_health = 10

        defend_action(self.agent)

        # Should enter defensive state but not heal
        self.assertTrue(self.agent.is_defending)
        self.assertEqual(self.agent.current_health, 10)  # Health unchanged

    def test_defend_action_no_cost(self):
        """Test defend action with zero cost."""
        self.agent.config.defense_cost = 0
        initial_resources = self.agent.resource_level

        defend_action(self.agent)

        # Should not change resources
        self.assertEqual(self.agent.resource_level, initial_resources)
        self.assertTrue(self.agent.is_defending)

    def test_defend_action_custom_parameters(self):
        """Test defend action with custom configuration parameters."""
        self.agent.config.defense_duration = 5
        self.agent.config.defense_healing = 3
        self.agent.config.defense_cost = 0

        defend_action(self.agent)

        # Should use custom parameters
        self.assertTrue(self.agent.is_defending)
        self.assertEqual(self.agent.defense_timer, 5)
        self.assertEqual(
            self.agent.current_health, 10
        )  # Healed to max health (8 + 2, capped at 10)


if __name__ == "__main__":
    unittest.main()
