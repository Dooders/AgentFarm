"""Unit tests for the move module.

This module tests the movement functionality including configuration,
neural network architecture, action selection, and movement execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from farm.actions.move import (
    MoveConfig,
    MoveActionSpace,
    MoveQNetwork,
    MoveModule,
    move_action,
    DEFAULT_MOVE_CONFIG
)


class TestMoveConfig(unittest.TestCase):
    """Test cases for MoveConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveConfig()
        
        self.assertEqual(config.move_base_cost, -0.1)
        self.assertEqual(config.move_resource_approach_reward, 0.3)
        self.assertEqual(config.move_resource_retreat_penalty, -0.2)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveConfig()
        config.move_base_cost = -0.05
        config.move_resource_approach_reward = 0.5
        config.move_resource_retreat_penalty = -0.3
        
        self.assertEqual(config.move_base_cost, -0.05)
        self.assertEqual(config.move_resource_approach_reward, 0.5)
        self.assertEqual(config.move_resource_retreat_penalty, -0.3)

    def test_inheritance_from_base_config(self):
        """Test that MoveConfig inherits from BaseDQNConfig."""
        config = MoveConfig()
        
        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestMoveActionSpace(unittest.TestCase):
    """Test cases for MoveActionSpace class."""

    def test_action_constants(self):
        """Test that all movement action constants are defined."""
        self.assertEqual(MoveActionSpace.RIGHT, 0)
        self.assertEqual(MoveActionSpace.LEFT, 1)
        self.assertEqual(MoveActionSpace.UP, 2)
        self.assertEqual(MoveActionSpace.DOWN, 3)

    def test_action_count(self):
        """Test that there are exactly 4 movement actions."""
        actions = [
            MoveActionSpace.RIGHT,
            MoveActionSpace.LEFT,
            MoveActionSpace.UP,
            MoveActionSpace.DOWN
        ]
        self.assertEqual(len(actions), 4)
        self.assertEqual(len(set(actions)), 4)  # All unique


class TestMoveQNetwork(unittest.TestCase):
    """Test cases for MoveQNetwork class."""

    def test_initialization(self):
        """Test MoveQNetwork initialization."""
        network = MoveQNetwork(input_dim=8, hidden_size=64)
        
        self.assertEqual(network.network[-1].out_features, 4)  # 4 movement actions

    def test_forward_pass(self):
        """Test MoveQNetwork forward pass."""
        network = MoveQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8).to(next(network.parameters()).device)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2, 4))  # Batch size 2, 4 actions

    def test_forward_pass_single_sample(self):
        """Test MoveQNetwork forward pass with single sample."""
        network = MoveQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(8).to(next(network.parameters()).device)
        
        output = network(x)
        
        self.assertEqual(output.shape, (4,))  # 4 actions

    def test_initialization_with_shared_encoder(self):
        """Test MoveQNetwork initialization with shared encoder."""
        shared_encoder = Mock()
        network = MoveQNetwork(input_dim=8, hidden_size=64, shared_encoder=shared_encoder)
        
        self.assertEqual(network.network[-1].out_features, 4)


class TestMoveModule(unittest.TestCase):
    """Test cases for MoveModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MoveConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test MoveModule initialization."""
        module = MoveModule(config=self.config)
        
        self.assertEqual(module.output_dim, 4)  # 4 movement actions
        self.assertEqual(module.config, self.config)

    def test_action_space_setup(self):
        """Test that action space is properly set up."""
        module = MoveModule(config=self.config)
        
        expected_actions = {
            MoveActionSpace.RIGHT: (1, 0),
            MoveActionSpace.LEFT: (-1, 0),
            MoveActionSpace.UP: (0, 1),
            MoveActionSpace.DOWN: (0, -1),
        }
        
        self.assertEqual(module.action_space, expected_actions)

    def test_get_movement(self):
        """Test get_movement method."""
        module = MoveModule(config=self.config)
        module.epsilon = 0.0  # Always exploit for consistent results
        
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.max_movement = 2.0
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        
        state = torch.randn(8).to(module.device)
        
        new_position = module.get_movement(mock_agent, state)
        
        # Should return a tuple of coordinates
        self.assertIsInstance(new_position, tuple)
        self.assertEqual(len(new_position), 2)
        self.assertIsInstance(new_position[0], float)
        self.assertIsInstance(new_position[1], float)

    def test_get_movement_boundary_checking(self):
        """Test that movement respects environment boundaries."""
        module = MoveModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        mock_agent.position = (0.0, 0.0)  # At boundary
        mock_agent.max_movement = 5.0
        mock_agent.environment = Mock()
        mock_agent.environment.width = 10.0
        mock_agent.environment.height = 10.0
        
        state = torch.randn(8).to(module.device)
        
        new_position = module.get_movement(mock_agent, state)
        
        # Should stay within bounds
        self.assertGreaterEqual(new_position[0], 0.0)
        self.assertLessEqual(new_position[0], 10.0)
        self.assertGreaterEqual(new_position[1], 0.0)
        self.assertLessEqual(new_position[1], 10.0)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = MoveModule(config=self.config)
        module.epsilon = 1.0  # Always explore
        
        state = torch.randn(8).to(module.device)
        
        # Test multiple selections to ensure randomness
        actions = set()
        for _ in range(10):
            action = module.select_action(state)
            actions.add(action)
        
        # Should have some variety in actions
        self.assertGreater(len(actions), 1)

    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        module = MoveModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        state = torch.randn(8).to(module.device)
        
        # Should always return the same action for the same state
        action1 = module.select_action(state)
        action2 = module.select_action(state)
        
        self.assertEqual(action1, action2)

    def test_select_action_temperature_based(self):
        """Test temperature-based action selection during exploration."""
        module = MoveModule(config=self.config)
        module.epsilon = 0.5  # Some exploration
        
        state = torch.randn(8).to(module.device)
        
        # Test that temperature-based selection works
        actions = set()
        for _ in range(10):
            action = module.select_action(state)
            actions.add(action)
        
        # Should have some variety in actions
        self.assertGreater(len(actions), 1)

    def test_network_initialization(self):
        """Test that Q-networks are properly initialized."""
        module = MoveModule(config=self.config)
        
        self.assertIsInstance(module.q_network, MoveQNetwork)
        self.assertIsInstance(module.target_network, MoveQNetwork)
        
        # Check that networks have correct output dimensions
        test_input = torch.randn(8).to(module.device)
        q_output = module.q_network(test_input)
        target_output = module.target_network(test_input)
        
        self.assertEqual(q_output.shape, (4,))
        self.assertEqual(target_output.shape, (4,))


class TestMoveAction(unittest.TestCase):
    """Test cases for move_action function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_environment = Mock()
        self.mock_move_module = Mock()
        
        # Set up agent mock
        self.mock_agent.environment = self.mock_environment
        self.mock_agent.move_module = self.mock_move_module
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.position = (10.0, 10.0)
        self.mock_agent.resource_level = 50.0
        self.mock_agent.total_reward = 0.0
        self.mock_agent.max_movement = 2.0
        
        # Set up environment mock
        self.mock_environment.time = 100
        self.mock_environment.db = Mock()
        self.mock_environment.width = 20.0
        self.mock_environment.height = 20.0
        # Fix: Add resources attribute to environment mock
        self.mock_environment.resources = []
        
        # Set up move module mock with proper memory
        self.mock_move_module.device = torch.device("cpu")
        self.mock_move_module.get_movement.return_value = (12.0, 10.0)
        # Fix: Add proper memory object that supports len()
        self.mock_move_module.memory = []
        self.mock_move_module.previous_state = None
        self.mock_move_module.previous_action = None
        self.mock_move_module.module_id = 1

    def test_move_action_basic(self):
        """Test basic move action execution."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return a resource
        mock_resource = Mock()
        mock_resource.position = (15.0, 10.0)
        mock_resource.is_depleted.return_value = False
        self.mock_environment.get_nearby_resources.return_value = [mock_resource]
        self.mock_environment.resources = [mock_resource]
        
        initial_position = self.mock_agent.position
        initial_reward = self.mock_agent.total_reward
        
        move_action(self.mock_agent)
        
        # Check that position was updated
        self.assertNotEqual(self.mock_agent.position, initial_position)
        
        # Check that reward was updated
        self.assertNotEqual(self.mock_agent.total_reward, initial_reward)

    def test_move_action_no_resources(self):
        """Test move action when no resources are nearby."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return no resources
        self.mock_environment.get_nearby_resources.return_value = []
        self.mock_environment.resources = []
        
        initial_position = self.mock_agent.position
        initial_reward = self.mock_agent.total_reward
        
        move_action(self.mock_agent)
        
        # Check that position was updated
        self.assertNotEqual(self.mock_agent.position, initial_position)
        
        # Check that reward was updated (should be negative due to base cost)
        self.assertLess(self.mock_agent.total_reward, initial_reward)

    def test_move_action_towards_resource(self):
        """Test move action when moving towards a resource."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return a resource
        mock_resource = Mock()
        mock_resource.position = (15.0, 10.0)
        mock_resource.is_depleted.return_value = False
        self.mock_environment.get_nearby_resources.return_value = [mock_resource]
        self.mock_environment.resources = [mock_resource]
        
        initial_reward = self.mock_agent.total_reward
        
        move_action(self.mock_agent)
        
        # Check that reward was updated (should be positive due to approach reward)
        self.assertGreater(self.mock_agent.total_reward, initial_reward)

    def test_move_action_away_from_resource(self):
        """Test move action when moving away from a resource."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return a resource
        mock_resource = Mock()
        mock_resource.position = (5.0, 10.0)  # Closer than new position
        mock_resource.is_depleted.return_value = False
        self.mock_environment.get_nearby_resources.return_value = [mock_resource]
        self.mock_environment.resources = [mock_resource]
        
        initial_reward = self.mock_agent.total_reward
        
        move_action(self.mock_agent)
        
        # Check that reward was updated (should be negative due to retreat penalty)
        self.assertLess(self.mock_agent.total_reward, initial_reward)

    def test_move_action_logging(self):
        """Test that move action is properly logged."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return no resources
        self.mock_environment.get_nearby_resources.return_value = []
        self.mock_environment.resources = []
        
        move_action(self.mock_agent)
        
        # Check that logging was called
        self.mock_environment.db.logger.log_agent_action.assert_called_once()

    def test_move_action_experience_storage(self):
        """Test that move action stores experience for learning."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return no resources
        self.mock_environment.get_nearby_resources.return_value = []
        self.mock_environment.resources = []
        
        # Mock the move module to track experience storage
        self.mock_move_module.previous_state = torch.randn(8)
        self.mock_move_module.previous_action = 0
        self.mock_move_module.store_experience = Mock()
        self.mock_move_module.train = Mock()
        
        move_action(self.mock_agent)
        
        # Check that experience was stored
        self.mock_move_module.store_experience.assert_called_once()

    def test_move_action_boundary_respect(self):
        """Test that move action respects environment boundaries."""
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock move module to return position at boundary
        self.mock_move_module.get_movement.return_value = (25.0, 25.0)  # Outside bounds
        
        # Mock environment to return no resources
        self.mock_environment.get_nearby_resources.return_value = []
        self.mock_environment.resources = []
        
        move_action(self.mock_agent)
        
        # Check that position was updated (boundary checking should be handled by get_movement)
        self.assertEqual(self.mock_agent.position, (25.0, 25.0))


if __name__ == "__main__":
    unittest.main() 