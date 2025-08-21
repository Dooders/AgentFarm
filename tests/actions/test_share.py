"""Unit tests for the share module.

This module tests the sharing functionality including configuration,
neural network architecture, action selection, and sharing execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from farm.actions.share import (
    ShareConfig,
    ShareActionSpace,
    ShareQNetwork,
    ShareModule,
    share_action,
    DEFAULT_SHARE_CONFIG
)


class TestShareConfig(unittest.TestCase):
    """Test cases for ShareConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ShareConfig()
        
        self.assertEqual(config.range, 30.0)
        self.assertEqual(config.min_amount, 1)
        self.assertEqual(config.success_reward, 0.3)
        self.assertEqual(config.failure_penalty, -0.1)
        self.assertEqual(config.altruism_bonus, 0.2)
        self.assertEqual(config.cooperation_memory, 100)
        self.assertEqual(config.max_resources, 30)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ShareConfig()
        config.range = 50.0
        config.min_amount = 2
        config.success_reward = 0.5
        config.failure_penalty = -0.2
        config.altruism_bonus = 0.3
        config.cooperation_memory = 200
        config.max_resources = 50
        
        self.assertEqual(config.range, 50.0)
        self.assertEqual(config.min_amount, 2)
        self.assertEqual(config.success_reward, 0.5)
        self.assertEqual(config.failure_penalty, -0.2)
        self.assertEqual(config.altruism_bonus, 0.3)
        self.assertEqual(config.cooperation_memory, 200)
        self.assertEqual(config.max_resources, 50)

    def test_inheritance_from_base_config(self):
        """Test that ShareConfig inherits from BaseDQNConfig."""
        config = ShareConfig()
        
        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestShareActionSpace(unittest.TestCase):
    """Test cases for ShareActionSpace class."""

    def test_action_constants(self):
        """Test that all sharing action constants are defined."""
        self.assertEqual(ShareActionSpace.NO_SHARE, 0)
        self.assertEqual(ShareActionSpace.SHARE_LOW, 1)
        self.assertEqual(ShareActionSpace.SHARE_MEDIUM, 2)
        self.assertEqual(ShareActionSpace.SHARE_HIGH, 3)

    def test_action_count(self):
        """Test that there are exactly 4 sharing actions."""
        actions = [
            ShareActionSpace.NO_SHARE,
            ShareActionSpace.SHARE_LOW,
            ShareActionSpace.SHARE_MEDIUM,
            ShareActionSpace.SHARE_HIGH
        ]
        self.assertEqual(len(actions), 4)
        self.assertEqual(len(set(actions)), 4)  # All unique


class TestShareQNetwork(unittest.TestCase):
    """Test cases for ShareQNetwork class."""

    def test_initialization(self):
        """Test ShareQNetwork initialization."""
        network = ShareQNetwork(input_dim=8, hidden_size=64)
        
        self.assertEqual(network.network[-1].out_features, 4)  # 4 sharing actions

    def test_forward_pass(self):
        """Test ShareQNetwork forward pass."""
        network = ShareQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2, 4))  # Batch size 2, 4 actions

    def test_forward_pass_single_sample(self):
        """Test ShareQNetwork forward pass with single sample."""
        network = ShareQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (4,))  # 4 actions

    def test_initialization_with_shared_encoder(self):
        """Test ShareQNetwork initialization with shared encoder."""
        from farm.actions.base_dqn import SharedEncoder
        
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = ShareQNetwork(input_dim=8, hidden_size=64, shared_encoder=shared_encoder)
        
        self.assertIs(network.shared_encoder, shared_encoder)


class TestShareModule(unittest.TestCase):
    """Test cases for ShareModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ShareConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test ShareModule initialization."""
        module = ShareModule(config=self.config)
        
        self.assertEqual(module.output_dim, 4)  # 4 sharing actions
        self.assertEqual(module.config, self.config)

    def test_action_space_setup(self):
        """Test that action space is properly set up."""
        module = ShareModule(config=self.config)
        
        expected_actions = {
            ShareActionSpace.NO_SHARE: 0,
            ShareActionSpace.SHARE_LOW: 1,
            ShareActionSpace.SHARE_MEDIUM: 2,
            ShareActionSpace.SHARE_HIGH: 3,
        }
        
        self.assertEqual(module.action_space, expected_actions)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = ShareModule(config=self.config)
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
        module = ShareModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        # Ensure state tensor is on the same device as the network
        state = torch.randn(8).to(module.device)
        
        # Should always return the same action for the same state
        action1 = module.select_action(state)
        action2 = module.select_action(state)
        
        self.assertEqual(action1, action2)

    def test_get_share_decision_no_share(self):
        """Test share decision when action is NO_SHARE."""
        module = ShareModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return NO_SHARE action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.9, 0.1, 0.2, 0.3]).to(module.device)  # NO_SHARE has highest Q-value
            
            action, target, amount = module.get_share_decision(mock_agent, state)
            
            self.assertEqual(action, ShareActionSpace.NO_SHARE)
            self.assertIsNone(target)
            self.assertEqual(amount, 0)

    def test_get_share_decision_share_success(self):
        """Test share decision when sharing is successful."""
        module = ShareModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        mock_agent.resource_level = 20.0
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return SHARE_MEDIUM action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.2, 0.9, 0.3]).to(module.device)  # SHARE_MEDIUM has highest Q-value
            
            # Mock _get_nearby_agents to return a target
            with patch.object(module, '_get_nearby_agents') as mock_get_nearby:
                mock_target = Mock()
                mock_target.resource_level = 5.0  # Low resources
                mock_get_nearby.return_value = [mock_target]
                
                # Mock _select_target to return the target
                with patch.object(module, '_select_target') as mock_select:
                    mock_select.return_value = mock_target
                    
                    action, target, amount = module.get_share_decision(mock_agent, state)
                    
                    self.assertEqual(action, ShareActionSpace.SHARE_MEDIUM)
                    self.assertEqual(target, mock_target)
                    self.assertGreater(amount, 0)

    def test_get_share_decision_no_targets(self):
        """Test share decision when no targets are available."""
        module = ShareModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return SHARE_MEDIUM action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.2, 0.9, 0.3]).to(module.device)  # SHARE_MEDIUM has highest Q-value
            
            # Mock _get_nearby_agents to return no targets
            with patch.object(module, '_get_nearby_agents') as mock_get_nearby:
                mock_get_nearby.return_value = []
                
                action, target, amount = module.get_share_decision(mock_agent, state)
                
                self.assertEqual(action, ShareActionSpace.NO_SHARE)
                self.assertIsNone(target)
                self.assertEqual(amount, 0)

    def test_get_nearby_agents(self):
        """Test getting nearby agents for sharing."""
        module = ShareModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.environment = Mock()
        mock_agent.config = Mock()
        mock_agent.config.share_range = 30.0
        
        # Mock environment to return nearby agents
        mock_nearby_agent = Mock()
        mock_nearby_agent.position = (15.0, 10.0)
        mock_nearby_agent.resource_level = 10.0
        mock_agent.environment.get_nearby_agents.return_value = [mock_nearby_agent]
        
        nearby_agents = module._get_nearby_agents(mock_agent)
        
        self.assertEqual(len(nearby_agents), 1)
        self.assertEqual(nearby_agents[0], mock_nearby_agent)

    def test_select_target(self):
        """Test selecting the best target for sharing."""
        module = ShareModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.resource_level = 20.0
        
        # Create mock nearby agents with different resource levels
        mock_agent1 = Mock()
        mock_agent1.resource_level = 15.0  # Higher resources
        mock_agent1.config = Mock()
        mock_agent1.config.starvation_threshold = 10.0
        
        mock_agent2 = Mock()
        mock_agent2.resource_level = 5.0   # Lower resources (better target)
        mock_agent2.config = Mock()
        mock_agent2.config.starvation_threshold = 10.0
        
        nearby_agents = [mock_agent1, mock_agent2]
        
        target = module._select_target(mock_agent, nearby_agents)  # type: ignore
        
        # Should select the agent with lower resources
        self.assertEqual(target, mock_agent2)

    def test_calculate_share_amount(self):
        """Test calculating share amount based on action."""
        module = ShareModule(config=self.config)
        
        mock_agent = Mock()
        mock_agent.resource_level = 20.0
        
        # Test different actions
        amount1 = module._calculate_share_amount(mock_agent, ShareActionSpace.NO_SHARE)
        amount2 = module._calculate_share_amount(mock_agent, ShareActionSpace.SHARE_LOW)
        amount3 = module._calculate_share_amount(mock_agent, ShareActionSpace.SHARE_MEDIUM)
        amount4 = module._calculate_share_amount(mock_agent, ShareActionSpace.SHARE_HIGH)
        
        self.assertEqual(amount1, 0)
        self.assertGreater(amount2, 0)
        self.assertGreater(amount3, amount2)
        self.assertGreater(amount4, amount3)

    def test_get_cooperation_score(self):
        """Test getting cooperation score for an agent."""
        module = ShareModule(config=self.config)
        
        # Test with non-existent agent
        score = module._get_cooperation_score("nonexistent_agent")
        self.assertEqual(score, 0.0)
        
        # Test with existing agent
        module.cooperation_history["test_agent"] = [1.0, 1.0, -1.0, 1.0]  # 3/4 = 0.75
        score = module._get_cooperation_score("test_agent")
        self.assertEqual(score, 0.5)  # (1+1-1+1)/4 = 0.5

    def test_update_cooperation(self):
        """Test updating cooperation score for an agent."""
        module = ShareModule(config=self.config)
        
        # Test updating cooperation history
        module.update_cooperation("test_agent", True)
        self.assertIn("test_agent", module.cooperation_history)
        self.assertEqual(module.cooperation_history["test_agent"][-1], 1.0)
        
        module.update_cooperation("test_agent", False)
        self.assertEqual(module.cooperation_history["test_agent"][-1], -1.0)

    def test_network_initialization(self):
        """Test that Q-networks are properly initialized."""
        module = ShareModule(config=self.config)
        
        self.assertIsInstance(module.q_network, ShareQNetwork)
        self.assertIsInstance(module.target_network, ShareQNetwork)
        
        # Check that networks have correct output dimensions
        test_input = torch.randn(8).to(module.device)
        q_output = module.q_network(test_input)
        target_output = module.target_network(test_input)
        
        self.assertEqual(q_output.shape, (4,))  # 4 actions
        self.assertEqual(target_output.shape, (4,))  # 4 actions


class TestShareAction(unittest.TestCase):
    """Test cases for share_action function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_environment = Mock()
        self.mock_share_module = Mock()
        
        # Set up agent mock
        self.mock_agent.environment = self.mock_environment
        self.mock_agent.share_module = self.mock_share_module
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.resource_level = 20.0
        self.mock_agent.position = (10.0, 10.0)
        self.mock_agent.current_health = 80.0
        self.mock_agent.starting_health = 100.0
        self.mock_agent.is_defending = False
        self.mock_agent.total_reward = 0.0
        
        # Set up environment mock
        self.mock_environment.time = 100
        self.mock_environment.db = Mock()
        self.mock_environment.agents = [self.mock_agent]  # List of agents
        self.mock_environment.get_nearby_agents.return_value = []  # Empty list by default
        self.mock_environment.resources_shared = 0  # Initialize resources_shared counter
        self.mock_environment.resources_shared_this_step = 0  # Initialize resources_shared_this_step counter
        
        # Set up share module mock
        self.mock_share_module.device = torch.device('cpu')
        self.mock_share_module.get_share_decision.return_value = (0, None, 0)  # NO_SHARE
        self.mock_share_module._get_cooperation_score.return_value = 0.5  # Mock cooperation score

    def test_share_action_no_share(self):
        """Test share action when decision is to not share."""
        self.mock_share_module.get_share_decision.return_value = (0, None, 0)  # NO_SHARE
        
        initial_resources = self.mock_agent.resource_level
        
        with patch('farm.actions.share._get_share_state') as mock_get_state:
            mock_get_state.return_value = [0.5, 0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 0.0]  # Mock state vector
            share_action(self.mock_agent)
        
        # Check that resources were not changed
        self.assertEqual(self.mock_agent.resource_level, initial_resources)
        
        # Check that action was logged
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "share")

    def test_share_action_successful_share(self):
        """Test share action when sharing is successful."""
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.resource_level = 5.0
        mock_target.config = Mock()
        mock_target.config.starvation_threshold = 3.0
        
        self.mock_share_module.get_share_decision.return_value = (2, mock_target, 3)  # SHARE_MEDIUM
        
        initial_resources = self.mock_agent.resource_level
        initial_target_resources = mock_target.resource_level
        
        with patch('farm.actions.share._get_share_state') as mock_get_state:
            mock_get_state.return_value = [0.5, 0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 0.0]  # Mock state vector
            share_action(self.mock_agent)
        
        # Check that resources were transferred
        self.assertLess(self.mock_agent.resource_level, initial_resources)
        self.assertGreater(mock_target.resource_level, initial_target_resources)
        
        # Check that action was logged
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "share")
        self.assertTrue(call_args[1]['details']['success'])

    def test_share_action_insufficient_resources(self):
        """Test share action when agent has insufficient resources."""
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.resource_level = 5.0
        mock_target.config = Mock()
        mock_target.config.starvation_threshold = 3.0
        
        # Set agent to have very low resources
        self.mock_agent.resource_level = 1.0
        
        self.mock_share_module.get_share_decision.return_value = (2, mock_target, 3)  # SHARE_MEDIUM
        
        initial_resources = self.mock_agent.resource_level
        initial_target_resources = mock_target.resource_level
        
        with patch('farm.actions.share._get_share_state') as mock_get_state:
            mock_get_state.return_value = [0.5, 0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 0.0]  # Mock state vector
            share_action(self.mock_agent)
        
        # Check that resources were not transferred
        self.assertEqual(self.mock_agent.resource_level, initial_resources)
        self.assertEqual(mock_target.resource_level, initial_target_resources)
        
        # Check that action was logged as failed
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertFalse(call_args[1]['details']['success'])

    def test_share_action_reward_calculation(self):
        """Test that reward is calculated and added to agent."""
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.resource_level = 5.0
        mock_target.config = Mock()
        mock_target.config.starvation_threshold = 3.0
        
        self.mock_share_module.get_share_decision.return_value = (2, mock_target, 3)  # SHARE_MEDIUM
        
        self.mock_agent.total_reward = 10.0
        
        with patch('farm.actions.share._get_share_state') as mock_get_state, \
             patch('farm.actions.share._calculate_share_reward') as mock_calculate_reward:
            mock_get_state.return_value = [0.5, 0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 0.0]  # Mock state vector
            mock_calculate_reward.return_value = 0.5
            share_action(self.mock_agent)
        
        # Check that reward was calculated and added
        mock_calculate_reward.assert_called_once()
        self.assertEqual(self.mock_agent.total_reward, 10.5)


if __name__ == "__main__":
    unittest.main() 