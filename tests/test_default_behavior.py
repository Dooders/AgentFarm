"""Tests for DefaultAgentBehavior with weighted action selection."""

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from farm.core.action import Action
from farm.core.agent.behaviors.default import DefaultAgentBehavior

np.random.seed(42)  # For reproducibility in tests


class TestDefaultAgentBehavior(unittest.TestCase):
    """Test cases for DefaultAgentBehavior with weighted selection."""

    def setUp(self):
        """Set up test fixtures."""
        self.behavior = DefaultAgentBehavior()
        self.mock_core = Mock()
        
    def test_weighted_selection_respects_action_weights(self):
        """Test that weighted selection respects action weights."""
        # Create actions with different weights
        actions = [
            Action("move", 0.6, Mock()),
            Action("gather", 0.3, Mock()),
            Action("attack", 0.1, Mock()),
        ]
        self.mock_core.actions = actions
        
        # Run many selections and count frequencies
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            selected = self.behavior.decide_action(self.mock_core, torch.zeros(1), None)
            selection_counts[selected.name] += 1
        
        # move should be selected most often
        self.assertGreater(selection_counts["move"], selection_counts["gather"])
        self.assertGreater(selection_counts["gather"], selection_counts["attack"])
        # With weight 0.6, move should be roughly 60% of selections
        self.assertGreater(selection_counts["move"], 500)

    def test_enabled_actions_respected(self):
        """Test that enabled_actions parameter is respected."""
        actions = [
            Action("move", 0.4, Mock()),
            Action("gather", 0.3, Mock()),
            Action("attack", 0.3, Mock()),
        ]
        self.mock_core.actions = actions
        
        # Only enable first two actions
        enabled = actions[:2]
        
        selection_counts = {"move": 0, "gather": 0, "attack": 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            selected = self.behavior.decide_action(self.mock_core, torch.zeros(1), enabled)
            selection_counts[selected.name] += 1
        
        # attack should never be selected
        self.assertEqual(selection_counts["attack"], 0)
        # move and gather should be selected based on their relative weights
        self.assertGreater(selection_counts["move"], selection_counts["gather"])

    def test_action_history_updated(self):
        """Test that action history is updated correctly."""
        actions = [
            Action("move", 0.5, Mock()),
            Action("gather", 0.5, Mock()),
        ]
        self.mock_core.actions = actions
        
        initial_history_len = len(self.behavior.action_history)
        
        action = self.behavior.decide_action(self.mock_core, torch.zeros(1), None)
        
        self.assertEqual(len(self.behavior.action_history), initial_history_len + 1)
        self.assertEqual(self.behavior.action_history[-1], action.name)

    def test_zero_weights_handled(self):
        """Test that zero weights are handled correctly (fallback to uniform)."""
        actions = [
            Action("move", 0.0, Mock()),
            Action("gather", 0.0, Mock()),
            Action("attack", 0.0, Mock()),
        ]
        self.mock_core.actions = actions
        
        # Should not raise error, should use uniform distribution
        action = self.behavior.decide_action(self.mock_core, torch.zeros(1), None)
        self.assertIn(action.name, ["move", "gather", "attack"])

    def test_backward_compatibility(self):
        """Test backward compatibility - behavior still works with existing code."""
        actions = [
            Action("move", 0.4, Mock()),
            Action("gather", 0.3, Mock()),
            Action("attack", 0.3, Mock()),
        ]
        self.mock_core.actions = actions
        
        # Should return an action
        action = self.behavior.decide_action(self.mock_core, torch.zeros(1), None)
        self.assertIsInstance(action, Action)
        self.assertIn(action.name, ["move", "gather", "attack"])

if __name__ == "__main__":
    unittest.main()

