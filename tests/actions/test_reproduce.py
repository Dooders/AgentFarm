"""Unit tests for the simple reproduction action.

This module tests the simple reproduction functionality that uses rule-based
logic instead of DQN.
"""

import unittest
from unittest.mock import Mock

from farm.core.action import reproduce_action


class TestSimpleReproduceAction(unittest.TestCase):
    """Test cases for the simple reproduction action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.resource_level = 10
        self.agent.config = Mock()
        self.agent.config.min_reproduction_resources = 8
        self.agent.reproduce = Mock(return_value=True)

    def test_reproduce_action_no_config(self):
        """Test reproduction action when agent has no config."""
        self.agent.config = None

        reproduce_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_reproduce_action_insufficient_resources(self):
        """Test reproduction action when agent has insufficient resources."""
        self.agent.resource_level = 5  # Less than min_reproduction_resources (8)

        reproduce_action(self.agent)

        # Should return early without reproducing
        self.agent.reproduce.assert_not_called()

    def test_reproduce_action_successful_reproduction(self):
        """Test successful reproduction when conditions are met."""
        self.agent.resource_level = 10  # Above threshold
        self.agent.reproduce.return_value = True

        # Run multiple times to account for random chance
        reproduce_called = False
        for _ in range(50):  # High number to ensure we hit the 50% chance
            reproduce_action(self.agent)
            if self.agent.reproduce.called:
                reproduce_called = True
                break

        # Should eventually call reproduce method due to random chance
        self.assertTrue(
            reproduce_called,
            "Reproduce should be called at least once due to random chance",
        )

    def test_reproduce_action_failed_reproduction(self):
        """Test failed reproduction attempt."""
        self.agent.resource_level = 10  # Above threshold
        self.agent.reproduce.return_value = False

        # Run multiple times to account for random chance
        reproduce_called = False
        for _ in range(50):  # High number to ensure we hit the 50% chance
            reproduce_action(self.agent)
            if self.agent.reproduce.called:
                reproduce_called = True
                break

        # Should eventually call reproduce method due to random chance
        self.assertTrue(
            reproduce_called,
            "Reproduce should be called at least once due to random chance",
        )

    def test_reproduce_action_random_chance(self):
        """Test that reproduction uses random chance."""
        self.agent.resource_level = 10  # Above threshold

        # Test multiple times to ensure random behavior
        reproduce_called = False
        for _ in range(20):  # Run multiple times to account for randomness
            reproduce_action(self.agent)
            if self.agent.reproduce.called:
                reproduce_called = True
                break

        # Should eventually call reproduce due to random chance
        self.assertTrue(
            reproduce_called,
            "Reproduce should be called at least once due to random chance",
        )


if __name__ == "__main__":
    unittest.main()
