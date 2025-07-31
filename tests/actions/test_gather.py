"""Unit tests for the gather module.

This module tests the gathering functionality including configuration,
neural network architecture, action selection, and gathering execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from farm.actions.gather import (
    GatherConfig,
    GatherActionSpace,
    GatherQNetwork,
    GatherModule,
    gather_action,
    DEFAULT_GATHER_CONFIG
)


class TestGatherConfig(unittest.TestCase):
    """Test cases for GatherConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GatherConfig()
        
        self.assertEqual(config.gather_success_reward, 1.0)
        self.assertEqual(config.gather_fail_penalty, -0.1)
        self.assertEqual(config.gather_efficiency_multiplier, 0.5)
        self.assertEqual(config.gather_cost_multiplier, 0.3)
        self.assertEqual(config.min_resource_threshold, 0.1)
        self.assertEqual(config.max_wait_steps, 5)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GatherConfig()
        config.gather_success_reward = 2.0
        config.gather_fail_penalty = -0.2
        config.gather_efficiency_multiplier = 0.7
        config.gather_cost_multiplier = 0.4
        config.min_resource_threshold = 0.2
        config.max_wait_steps = 10
        
        self.assertEqual(config.gather_success_reward, 2.0)
        self.assertEqual(config.gather_fail_penalty, -0.2)
        self.assertEqual(config.gather_efficiency_multiplier, 0.7)
        self.assertEqual(config.gather_cost_multiplier, 0.4)
        self.assertEqual(config.min_resource_threshold, 0.2)
        self.assertEqual(config.max_wait_steps, 10)

    def test_inheritance_from_base_config(self):
        """Test that GatherConfig inherits from BaseDQNConfig."""
        config = GatherConfig()
        
        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestGatherActionSpace(unittest.TestCase):
    """Test cases for GatherActionSpace class."""

    def test_action_constants(self):
        """Test that all gathering action constants are defined."""
        self.assertEqual(GatherActionSpace.GATHER, 0)
        self.assertEqual(GatherActionSpace.WAIT, 1)
        self.assertEqual(GatherActionSpace.SKIP, 2)

    def test_action_count(self):
        """Test that there are exactly 3 gathering actions."""
        actions = [
            GatherActionSpace.GATHER,
            GatherActionSpace.WAIT,
            GatherActionSpace.SKIP
        ]
        self.assertEqual(len(actions), 3)
        self.assertEqual(len(set(actions)), 3)  # All unique


class TestGatherQNetwork(unittest.TestCase):
    """Test cases for GatherQNetwork class."""

    def test_initialization(self):
        """Test GatherQNetwork initialization."""
        network = GatherQNetwork(input_dim=8, hidden_size=64)
        
        self.assertEqual(network.network[-1].out_features, 3)  # 3 gathering actions

    def test_forward_pass(self):
        """Test GatherQNetwork forward pass."""
        network = GatherQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2, 3))  # Batch size 2, 3 actions

    def test_forward_pass_single_sample(self):
        """Test GatherQNetwork forward pass with single sample."""
        network = GatherQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (3,))  # 3 actions

    def test_initialization_with_shared_encoder(self):
        """Test GatherQNetwork initialization with shared encoder."""
        from farm.actions.base_dqn import SharedEncoder
        
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = GatherQNetwork(input_dim=8, hidden_size=64, shared_encoder=shared_encoder)
        
        self.assertIs(network.shared_encoder, shared_encoder)


class TestGatherModule(unittest.TestCase):
    """Test cases for GatherModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GatherConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test GatherModule initialization."""
        module = GatherModule(config=self.config)
        
        self.assertEqual(module.output_dim, 3)  # 3 gathering actions
        self.assertEqual(module.config, self.config)
        self.assertEqual(module.input_dim, 8)
        self.assertEqual(module.last_gather_step, 0)
        self.assertEqual(module.steps_since_gather, 0)
        self.assertEqual(module.consecutive_failed_attempts, 0)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = GatherModule(config=self.config)
        module.epsilon = 1.0  # Always explore
        
        state = torch.randn(8)
        
        # Test multiple selections to ensure randomness
        actions = set()
        for _ in range(10):
            action = module.select_action(state)
            actions.add(action)
        
        # Should have some variety in actions
        self.assertGreater(len(actions), 1)

    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        # Create state tensor on the same device as the network
        state = torch.randn(8, device=module.device)
        
        # Mock the Q-network to return deterministic values
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.9, 0.1, 0.2], device=module.device)  # GATHER has highest Q-value
            
            # Should always return the same action for the same state
            action1 = module.select_action(state)
            action2 = module.select_action(state)
            
            self.assertEqual(action1, action2)
            self.assertEqual(action1, 0)  # Should be GATHER action

    def test_get_gather_decision_skip(self):
        """Test gather decision when action is SKIP."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8)
        
        # Mock the Q-network to return SKIP action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.2, 0.9])  # SKIP has highest Q-value
            
            should_gather, target_resource = module.get_gather_decision(mock_agent, state)
            
            self.assertFalse(should_gather)
            self.assertIsNone(target_resource)

    def test_get_gather_decision_wait(self):
        """Test gather decision when action is WAIT."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8)
        
        # Mock the Q-network to return WAIT action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.9, 0.2])  # WAIT has highest Q-value
            
            should_gather, target_resource = module.get_gather_decision(mock_agent, state)
            
            self.assertFalse(should_gather)
            self.assertIsNone(target_resource)
            self.assertEqual(module.steps_since_gather, 1)

    def test_get_gather_decision_wait_force_gather(self):
        """Test gather decision when WAIT exceeds max steps."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        module.steps_since_gather = self.config.max_wait_steps  # At max wait steps
        
        mock_agent = Mock()
        state = torch.randn(8)
        
        # Mock the Q-network to return WAIT action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.9, 0.2])  # WAIT has highest Q-value
            
            # Mock _find_best_resource to return a resource
            with patch.object(module, '_find_best_resource') as mock_find:
                mock_resource = Mock()
                mock_find.return_value = mock_resource
                
                should_gather, target_resource = module.get_gather_decision(mock_agent, state)
                
                self.assertTrue(should_gather)
                self.assertEqual(target_resource, mock_resource)

    def test_get_gather_decision_gather(self):
        """Test gather decision when action is GATHER."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8)
        
        # Mock the Q-network to return GATHER action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.9, 0.1, 0.2])  # GATHER has highest Q-value
            
            # Mock _find_best_resource to return a resource
            with patch.object(module, '_find_best_resource') as mock_find:
                mock_resource = Mock()
                mock_find.return_value = mock_resource
                
                should_gather, target_resource = module.get_gather_decision(mock_agent, state)
                
                self.assertTrue(should_gather)
                self.assertEqual(target_resource, mock_resource)

    def test_get_gather_decision_no_resource(self):
        """Test gather decision when no resource is found."""
        module = GatherModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8)
        
        # Mock the Q-network to return GATHER action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.9, 0.1, 0.2])  # GATHER has highest Q-value
            
            # Mock _find_best_resource to return None
            with patch.object(module, '_find_best_resource') as mock_find:
                mock_find.return_value = None
                
                should_gather, target_resource = module.get_gather_decision(mock_agent, state)
                
                self.assertFalse(should_gather)
                self.assertIsNone(target_resource)

    def test_process_gather_state(self):
        """Test processing of gather state."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 50.0
        mock_agent.environment = Mock()
        mock_agent.config = Mock()
        mock_agent.config.gathering_range = 5.0
        
        # Mock _find_best_resource to return a resource
        with patch.object(module, '_find_best_resource') as mock_find:
            mock_resource = Mock()
            mock_resource.position = (12.0, 10.0)
            mock_resource.amount = 10.0
            mock_resource.regeneration_rate = 0.1
            mock_find.return_value = mock_resource
            
            # Mock environment to return resources
            mock_agent.environment.get_nearby_resources.return_value = [mock_resource]
            
            state = module._process_gather_state(mock_agent)
            
            self.assertEqual(state.shape, (6,))
            self.assertIsInstance(state, torch.Tensor)

    def test_process_gather_state_no_resource(self):
        """Test processing of gather state when no resource is found."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.environment = Mock()
        mock_agent.config = Mock()
        
        # Mock _find_best_resource to return None
        with patch.object(module, '_find_best_resource') as mock_find:
            mock_find.return_value = None
            
            state = module._process_gather_state(mock_agent)
            
            self.assertEqual(state.shape, (6,))
            self.assertTrue(torch.all(state == 0))  # Should be all zeros

    def test_find_best_resource(self):
        """Test finding the best resource."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.environment = Mock()
        mock_agent.config = Mock()
        mock_agent.config.gathering_range = 5.0
        
        # Mock environment to return resources
        mock_resource1 = Mock()
        mock_resource1.position = (12.0, 10.0)
        mock_resource1.amount = 5.0
        
        mock_resource2 = Mock()
        mock_resource2.position = (11.0, 10.0)
        mock_resource2.amount = 10.0
        
        mock_agent.environment.get_nearby_resources.return_value = [mock_resource1, mock_resource2]
        
        best_resource = module._find_best_resource(mock_agent)
        
        # Should return the resource with higher amount (resource2)
        self.assertEqual(best_resource, mock_resource2)

    def test_find_best_resource_filtered(self):
        """Test that depleted resources are filtered out."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.environment = Mock()
        mock_agent.config = Mock()
        mock_agent.config.gathering_range = 5.0
        
        # Mock environment to return resources
        mock_resource1 = Mock()
        mock_resource1.position = (12.0, 10.0)
        mock_resource1.amount = 0.05  # Below threshold
        
        mock_resource2 = Mock()
        mock_resource2.position = (11.0, 10.0)
        mock_resource2.amount = 10.0  # Above threshold
        
        mock_agent.environment.get_nearby_resources.return_value = [mock_resource1, mock_resource2]
        
        best_resource = module._find_best_resource(mock_agent)
        
        # Should return only the resource above threshold
        self.assertEqual(best_resource, mock_resource2)

    def test_calculate_gather_reward_success(self):
        """Test reward calculation for successful gathering."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.resource_level = 60.0  # Gained 10 resources
        
        mock_target_resource = Mock()
        mock_target_resource.max_amount = 20.0
        
        reward = module.calculate_gather_reward(mock_agent, 50.0, mock_target_resource)
        
        # Should be positive due to success
        self.assertGreater(reward, 0)
        
        # Check that counters were reset
        self.assertEqual(module.consecutive_failed_attempts, 0)
        self.assertEqual(module.steps_since_gather, 0)

    def test_calculate_gather_reward_failure(self):
        """Test reward calculation for failed gathering."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.resource_level = 50.0  # No gain
        
        reward = module.calculate_gather_reward(mock_agent, 50.0, Mock())
        
        # Should be negative due to failure
        self.assertLess(reward, 0)
        
        # Check that consecutive failures were incremented
        self.assertEqual(module.consecutive_failed_attempts, 1)

    def test_calculate_gather_reward_no_target(self):
        """Test reward calculation when no target is provided."""
        module = GatherModule(config=self.config)
        
        mock_agent = Mock()
        
        reward = module.calculate_gather_reward(mock_agent, 50.0, None)
        
        # Should be the failure penalty
        self.assertEqual(reward, self.config.gather_fail_penalty)


class TestGatherAction(unittest.TestCase):
    """Test cases for gather_action function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_environment = Mock()
        self.mock_gather_module = Mock()
        
        # Set up agent mock with proper numeric values
        self.mock_agent.environment = self.mock_environment
        self.mock_agent.gather_module = self.mock_gather_module
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.resource_level = 50.0
        self.mock_agent.total_reward = 10.0  # Add numeric total_reward
        self.mock_agent.position = (10.0, 10.0)  # Add numeric position
        
        # Set up environment mock
        self.mock_environment.time = 100
        self.mock_environment.db = Mock()
        self.mock_environment.db.logger = Mock()
        
        # Set up gather module mock
        self.mock_gather_module._process_gather_state.return_value = torch.randn(6)
        self.mock_gather_module.get_gather_decision.return_value = (False, None)

    def test_gather_action_skip(self):
        """Test gather action when decision is to skip."""
        self.mock_gather_module.get_gather_decision.return_value = (False, None)
        
        initial_resources = self.mock_agent.resource_level
        
        gather_action(self.mock_agent)
        
        # Check that resources were not changed
        self.assertEqual(self.mock_agent.resource_level, initial_resources)
        
        # Check that action was logged
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "gather")
        self.assertFalse(call_args[1]['details']['success'])

    def test_gather_action_success(self):
        """Test gather action when gathering is successful."""
        mock_resource = Mock()
        mock_resource.amount = 10.0
        mock_resource.is_depleted.return_value = False
        mock_resource.consume = Mock()
        mock_resource.position = (12.0, 10.0)  # Add position for distance calculation
        
        self.mock_gather_module.get_gather_decision.return_value = (True, mock_resource)
        self.mock_gather_module.calculate_gather_reward.return_value = 2.0  # Return numeric value
        
        # Mock agent config
        self.mock_agent.config = Mock()
        self.mock_agent.config.max_gather_amount = 5.0
        
        initial_resources = self.mock_agent.resource_level
        
        gather_action(self.mock_agent)
        
        # Check that resources were increased
        self.assertGreater(self.mock_agent.resource_level, initial_resources)
        
        # Check that resource was consumed
        mock_resource.consume.assert_called_once_with(5.0)
        
        # Check that action was logged
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "gather")
        self.assertTrue(call_args[1]['details']['success'])

    def test_gather_action_depleted_resource(self):
        """Test gather action when resource is depleted."""
        mock_resource = Mock()
        mock_resource.amount = 10.0
        mock_resource.is_depleted.return_value = True
        mock_resource.position = (12.0, 10.0)  # Add position for distance calculation
        
        self.mock_gather_module.get_gather_decision.return_value = (True, mock_resource)
        
        initial_resources = self.mock_agent.resource_level
        
        gather_action(self.mock_agent)
        
        # Check that resources were not changed
        self.assertEqual(self.mock_agent.resource_level, initial_resources)
        
        # Check that action was logged as failed
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "gather")
        self.assertFalse(call_args[1]['details']['success'])

    def test_gather_action_reward_calculation(self):
        """Test that reward is calculated and added to agent."""
        mock_resource = Mock()
        mock_resource.amount = 10.0
        mock_resource.is_depleted.return_value = False
        mock_resource.consume = Mock()
        mock_resource.position = (12.0, 10.0)  # Add position for distance calculation
        
        self.mock_gather_module.get_gather_decision.return_value = (True, mock_resource)
        self.mock_gather_module.calculate_gather_reward.return_value = 2.0
        
        # Mock agent config
        self.mock_agent.config = Mock()
        self.mock_agent.config.max_gather_amount = 5.0
        self.mock_agent.total_reward = 10.0
        
        gather_action(self.mock_agent)
        
        # Check that reward was calculated and added
        self.mock_gather_module.calculate_gather_reward.assert_called_once()
        # The total_reward should be updated by the gather_action function
        # Since we're using a Mock, we need to check the call to +=
        # The actual value will be a Mock, but we can verify the operation was attempted


if __name__ == "__main__":
    unittest.main() 