"""Unit tests for the reproduce module.

This module tests the reproduction functionality including configuration,
neural network architecture, action selection, and reproduction execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from farm.actions.reproduce import (
    ReproduceConfig,
    ReproduceActionSpace,
    ReproduceQNetwork,
    ReproduceModule,
    reproduce_action
)


class TestReproduceConfig(unittest.TestCase):
    """Test cases for ReproduceConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReproduceConfig()
        
        self.assertEqual(config.reproduce_success_reward, 1.0)
        self.assertEqual(config.reproduce_fail_penalty, -0.2)
        self.assertEqual(config.offspring_survival_bonus, 0.5)
        self.assertEqual(config.population_balance_bonus, 0.3)
        self.assertEqual(config.min_health_ratio, 0.5)
        self.assertEqual(config.min_resource_ratio, 0.6)
        self.assertEqual(config.ideal_density_radius, 50.0)
        self.assertEqual(config.max_local_density, 0.7)
        self.assertEqual(config.min_space_required, 20.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReproduceConfig()
        config.reproduce_success_reward = 2.0
        config.reproduce_fail_penalty = -0.5
        config.offspring_survival_bonus = 0.7
        config.population_balance_bonus = 0.4
        config.min_health_ratio = 0.6
        config.min_resource_ratio = 0.7
        config.ideal_density_radius = 60.0
        config.max_local_density = 0.8
        config.min_space_required = 25.0
        
        self.assertEqual(config.reproduce_success_reward, 2.0)
        self.assertEqual(config.reproduce_fail_penalty, -0.5)
        self.assertEqual(config.offspring_survival_bonus, 0.7)
        self.assertEqual(config.population_balance_bonus, 0.4)
        self.assertEqual(config.min_health_ratio, 0.6)
        self.assertEqual(config.min_resource_ratio, 0.7)
        self.assertEqual(config.ideal_density_radius, 60.0)
        self.assertEqual(config.max_local_density, 0.8)
        self.assertEqual(config.min_space_required, 25.0)

    def test_inheritance_from_base_config(self):
        """Test that ReproduceConfig inherits from BaseDQNConfig."""
        config = ReproduceConfig()
        
        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestReproduceActionSpace(unittest.TestCase):
    """Test cases for ReproduceActionSpace class."""

    def test_action_constants(self):
        """Test that all reproduction action constants are defined."""
        self.assertEqual(ReproduceActionSpace.WAIT, 0)
        self.assertEqual(ReproduceActionSpace.REPRODUCE, 1)

    def test_action_count(self):
        """Test that there are exactly 2 reproduction actions."""
        actions = [
            ReproduceActionSpace.WAIT,
            ReproduceActionSpace.REPRODUCE
        ]
        self.assertEqual(len(actions), 2)
        self.assertEqual(len(set(actions)), 2)  # All unique


class TestReproduceQNetwork(unittest.TestCase):
    """Test cases for ReproduceQNetwork class."""

    def test_initialization(self):
        """Test ReproduceQNetwork initialization."""
        network = ReproduceQNetwork(input_dim=8, hidden_size=64)
        
        self.assertEqual(network.network[-1].out_features, 2)  # 2 reproduction actions

    def test_forward_pass(self):
        """Test ReproduceQNetwork forward pass."""
        network = ReproduceQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2, 2))  # Batch size 2, 2 actions

    def test_forward_pass_single_sample(self):
        """Test ReproduceQNetwork forward pass with single sample."""
        network = ReproduceQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2,))  # 2 actions

    def test_initialization_with_shared_encoder(self):
        """Test ReproduceQNetwork initialization with shared encoder."""
        from farm.actions.base_dqn import SharedEncoder
        
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = ReproduceQNetwork(input_dim=8, hidden_size=64, shared_encoder=shared_encoder)
        
        self.assertIs(network.shared_encoder, shared_encoder)


class TestReproduceModule(unittest.TestCase):
    """Test cases for ReproduceModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ReproduceConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test ReproduceModule initialization."""
        module = ReproduceModule(config=self.config)
        
        self.assertEqual(module.output_dim, 2)  # 2 reproduction actions
        self.assertEqual(module.config, self.config)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = ReproduceModule(config=self.config)
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
        module = ReproduceModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        # Move state to same device as network
        state = torch.randn(8).to(module.device)
        
        # Should always return the same action for the same state
        action1 = module.select_action(state)
        action2 = module.select_action(state)
        
        self.assertEqual(action1, action2)

    def test_get_reproduction_decision_wait(self):
        """Test reproduction decision when action is WAIT."""
        module = ReproduceModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return WAIT action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.9, 0.1]).to(module.device)  # WAIT has highest Q-value
            
            should_reproduce, confidence = module.get_reproduction_decision(mock_agent, state)
            
            self.assertFalse(should_reproduce)
            self.assertEqual(confidence, 0.0)

    def test_get_reproduction_decision_reproduce(self):
        """Test reproduction decision when action is REPRODUCE."""
        module = ReproduceModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        mock_agent.current_health = 80.0
        mock_agent.starting_health = 100.0
        mock_agent.resource_level = 60.0
        mock_agent.starting_health = 100.0
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return REPRODUCE action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.9]).to(module.device)  # REPRODUCE has highest Q-value
            
            # Mock the standalone _check_reproduction_conditions function
            with patch('farm.actions.reproduce._check_reproduction_conditions') as mock_check:
                mock_check.return_value = True
                
                # Mock agent reproduction to return offspring
                mock_offspring = Mock()
                mock_agent.reproduce.return_value = mock_offspring
                
                should_reproduce, confidence = module.get_reproduction_decision(mock_agent, state)
                
                self.assertTrue(should_reproduce)
                self.assertGreater(confidence, 0.0)

    def test_get_reproduction_decision_conditions_not_met(self):
        """Test reproduction decision when conditions are not met."""
        module = ReproduceModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
        
        mock_agent = Mock()
        state = torch.randn(8).to(module.device)
        
        # Mock the Q-network to return REPRODUCE action
        with patch.object(module.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([0.1, 0.9]).to(module.device)  # REPRODUCE has highest Q-value
            
            # Mock the standalone _check_reproduction_conditions function
            with patch('farm.actions.reproduce._check_reproduction_conditions') as mock_check:
                mock_check.return_value = False
                
                should_reproduce, confidence = module.get_reproduction_decision(mock_agent, state)
                
                self.assertFalse(should_reproduce)
                self.assertEqual(confidence, 0.0)

    def test_network_initialization(self):
        """Test that Q-networks are properly initialized."""
        module = ReproduceModule(config=self.config)
        
        self.assertIsInstance(module.q_network, ReproduceQNetwork)
        self.assertIsInstance(module.target_network, ReproduceQNetwork)
        
        # Check that networks have correct output dimensions
        test_input = torch.randn(8).to(module.device)
        q_output = module.q_network(test_input)
        target_output = module.target_network(test_input)
        
        self.assertEqual(q_output.shape, (2,))  # 2 actions
        self.assertEqual(target_output.shape, (2,))  # 2 actions


class TestReproduceAction(unittest.TestCase):
    """Test cases for reproduce_action function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_environment = Mock()
        self.mock_reproduce_module = Mock()
        
        # Set up agent mock with proper attributes
        self.mock_agent.environment = self.mock_environment
        self.mock_agent.reproduce_module = self.mock_reproduce_module
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.resource_level = 50.0
        self.mock_agent.current_health = 80.0
        self.mock_agent.starting_health = 100.0
        self.mock_agent.position = [0, 0]
        self.mock_agent.config = Mock()
        self.mock_agent.config.min_reproduction_resources = 10.0
        self.mock_agent.config.gathering_range = 50.0
        self.mock_agent.config.max_population = 100
        self.mock_agent.device = torch.device('cpu')
        self.mock_agent.starvation_threshold = 0.5
        self.mock_agent.max_starvation = 1.0
        self.mock_agent.is_defending = False
        self.mock_agent.generation = 1
        self.mock_agent.total_reward = 10.0
        
        # Set up environment mock with proper iterables
        self.mock_environment.time = 100
        self.mock_environment.db = Mock()
        self.mock_environment.agents = []
        self.mock_environment.resources = []
        
        # Set up reproduce module mock
        self.mock_reproduce_module.get_reproduction_decision.return_value = (False, 0.0)

    def test_reproduce_action_wait(self):
        """Test reproduce action when decision is to wait."""
        self.mock_reproduce_module.get_reproduction_decision.return_value = (False, 0.0)
        
        reproduce_action(self.mock_agent)
        
        # Check that action was logged
        self.mock_environment.db.logger.log_agent_action.assert_called_once()
        call_args = self.mock_environment.db.logger.log_agent_action.call_args
        self.assertEqual(call_args[1]['action_type'], "reproduce")

    def test_reproduce_action_successful(self):
        """Test reproduce action when reproduction is successful."""
        mock_offspring = Mock()
        mock_offspring.agent_id = "offspring_agent"
        
        self.mock_reproduce_module.get_reproduction_decision.return_value = (True, 0.8)
        
        # Mock the _calculate_reproduction_reward function
        with patch('farm.actions.reproduce._calculate_reproduction_reward') as mock_calc:
            mock_calc.return_value = 1.5
            
            reproduce_action(self.mock_agent)
            
            # Check that reward was calculated and added
            mock_calc.assert_called_once()
            self.assertEqual(self.mock_agent.total_reward, 11.5)
            
            # Check that action was logged
            self.mock_environment.db.logger.log_agent_action.assert_called_once()
            call_args = self.mock_environment.db.logger.log_agent_action.call_args
            self.assertEqual(call_args[1]['action_type'], "reproduce")
            self.assertTrue(call_args[1]['details']['success'])

    def test_reproduce_action_failed(self):
        """Test reproduce action when reproduction fails."""
        self.mock_reproduce_module.get_reproduction_decision.return_value = (True, 0.8)
        
        # Mock _check_reproduction_conditions to return False
        with patch('farm.actions.reproduce._check_reproduction_conditions') as mock_check:
            mock_check.return_value = False
            
            reproduce_action(self.mock_agent)
            
            # Check that action was logged as failed
            self.mock_environment.db.logger.log_agent_action.assert_called_once()
            call_args = self.mock_environment.db.logger.log_agent_action.call_args
            self.assertEqual(call_args[1]['action_type'], "reproduce")
            self.assertFalse(call_args[1]['details']['success'])


if __name__ == "__main__":
    unittest.main() 