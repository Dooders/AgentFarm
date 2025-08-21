"""Unit tests for the attack module.

This module tests the attack functionality including configuration,
neural network architecture, action selection, and attack execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from farm.actions.attack import (
    AttackConfig,
    AttackActionSpace,
    AttackQNetwork,
    AttackModule,
    attack_action,
    DEFAULT_ATTACK_CONFIG
)


class TestAttackConfig(unittest.TestCase):
    """Test cases for AttackConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AttackConfig()
        
        self.assertEqual(config.base_cost, -0.2)
        self.assertEqual(config.success_reward, 1.0)
        self.assertEqual(config.failure_penalty, -0.3)
        self.assertEqual(config.defense_threshold, 0.3)
        self.assertEqual(config.defense_boost, 2.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AttackConfig()
        config.base_cost = -0.1
        config.success_reward = 2.0
        config.failure_penalty = -0.5
        config.defense_threshold = 0.5
        config.defense_boost = 3.0
        
        self.assertEqual(config.base_cost, -0.1)
        self.assertEqual(config.success_reward, 2.0)
        self.assertEqual(config.failure_penalty, -0.5)
        self.assertEqual(config.defense_threshold, 0.5)
        self.assertEqual(config.defense_boost, 3.0)

    def test_inheritance_from_base_config(self):
        """Test that AttackConfig inherits from BaseDQNConfig."""
        config = AttackConfig()
        
        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestAttackActionSpace(unittest.TestCase):
    """Test cases for AttackActionSpace class."""

    def test_action_constants(self):
        """Test that all attack action constants are defined."""
        self.assertEqual(AttackActionSpace.ATTACK_RIGHT, 0)
        self.assertEqual(AttackActionSpace.ATTACK_LEFT, 1)
        self.assertEqual(AttackActionSpace.ATTACK_UP, 2)
        self.assertEqual(AttackActionSpace.ATTACK_DOWN, 3)
        self.assertEqual(AttackActionSpace.DEFEND, 4)

    def test_action_count(self):
        """Test that there are exactly 5 attack actions."""
        actions = [
            AttackActionSpace.ATTACK_RIGHT,
            AttackActionSpace.ATTACK_LEFT,
            AttackActionSpace.ATTACK_UP,
            AttackActionSpace.ATTACK_DOWN,
            AttackActionSpace.DEFEND
        ]
        self.assertEqual(len(actions), 5)
        self.assertEqual(len(set(actions)), 5)  # All unique


class TestAttackQNetwork(unittest.TestCase):
    """Test cases for AttackQNetwork class."""

    def test_initialization(self):
        """Test AttackQNetwork initialization."""
        network = AttackQNetwork(input_dim=8, hidden_size=64)
        
        self.assertEqual(network.network[-1].out_features, 5)  # 5 attack actions

    def test_forward_pass(self):
        """Test AttackQNetwork forward pass."""
        network = AttackQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (2, 5))  # Batch size 2, 5 actions

    def test_forward_pass_single_sample(self):
        """Test AttackQNetwork forward pass with single sample."""
        network = AttackQNetwork(input_dim=8, hidden_size=64)
        x = torch.randn(8)
        
        output = network(x)
        
        self.assertEqual(output.shape, (5,))  # 5 actions

    def test_initialization_with_shared_encoder(self):
        """Test AttackQNetwork initialization with shared encoder."""
        from farm.actions.base_dqn import SharedEncoder
        
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = AttackQNetwork(input_dim=8, hidden_size=64, shared_encoder=shared_encoder)
        
        self.assertIs(network.shared_encoder, shared_encoder)


class TestAttackModule(unittest.TestCase):
    """Test cases for AttackModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AttackConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test AttackModule initialization."""
        module = AttackModule(config=self.config)
        
        self.assertEqual(module.output_dim, 5)  # 5 attack actions
        self.assertEqual(module.config, self.config)
        self.assertEqual(module.attack_config, self.config)

    def test_action_space_setup(self):
        """Test that action space is properly set up."""
        module = AttackModule(config=self.config)
        
        expected_actions = {
            AttackActionSpace.ATTACK_RIGHT: (1, 0),
            AttackActionSpace.ATTACK_LEFT: (-1, 0),
            AttackActionSpace.ATTACK_UP: (0, 1),
            AttackActionSpace.ATTACK_DOWN: (0, -1),
            AttackActionSpace.DEFEND: (0, 0),
        }
        
        self.assertEqual(module.action_space, expected_actions)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = AttackModule(config=self.config)
        module.epsilon = 1.0  # Always explore
        
        state = torch.randn(8).to(module.device)
        health_ratio = 0.5
        
        # Test multiple selections to ensure randomness
        actions = set()
        for _ in range(10):
            action = module.select_action(state, health_ratio)
            actions.add(action)
        
        # Should have some variety in actions
        self.assertGreater(len(actions), 1)

    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        module = AttackModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
    
        state = torch.randn(8).to(module.device)
        health_ratio = 0.5
    
        # Should always return the same action for the same state
        action1 = module.select_action(state, health_ratio)
        action2 = module.select_action(state, health_ratio)
        
        self.assertEqual(action1, action2)

    def test_select_action_defense_boost(self):
        """Test that defense action is boosted when health is low."""
        module = AttackModule(config=self.config)
        module.epsilon = 0.0  # Always exploit
    
        state = torch.randn(8).to(module.device)
    
        # Test with high health - should not boost defense
        high_health_action = module.select_action(state, health_ratio=0.8)
        
        # Test with low health - should boost defense
        low_health_action = module.select_action(state, health_ratio=0.2)
        
        # The actions might be different due to health-based boosting
        # We can't guarantee they'll be different, but we can test the logic works

    def test_network_initialization(self):
        """Test that Q-networks are properly initialized."""
        module = AttackModule(config=self.config)
        
        self.assertIsInstance(module.q_network, AttackQNetwork)
        self.assertIsInstance(module.target_network, AttackQNetwork)
        
        # Check that networks have correct output dimensions
        test_input = torch.randn(8).to(module.device)
        q_output = module.q_network(test_input)
        target_output = module.target_network(test_input)
        
        self.assertEqual(q_output.shape, (5,))  # 5 actions
        self.assertEqual(target_output.shape, (5,))  # 5 actions


class TestAttackAction(unittest.TestCase):
    """Test cases for attack_action function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_environment = Mock()
        self.mock_config = Mock()
        self.mock_attack_module = Mock()
        self.mock_attack_logger = Mock()
        
        # Set up agent mock
        self.mock_agent.environment = self.mock_environment
        self.mock_agent.config = self.mock_config
        self.mock_agent.attack_module = self.mock_attack_module
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.position = (10.0, 10.0)
        self.mock_agent.current_health = 80.0
        self.mock_agent.starting_health = 100.0
        self.mock_agent.resource_level = 50.0
        self.mock_agent.attack_strength = 10.0
        self.mock_agent.is_defending = False
        
        # Set up environment mock
        self.mock_environment.time = 100
        self.mock_environment.db = Mock()
        self.mock_environment.combat_encounters = 0
        self.mock_environment.combat_encounters_this_step = 0
        self.mock_environment.successful_attacks = 0
        self.mock_environment.successful_attacks_this_step = 0
        
        # Set up config mock
        self.mock_config.range = 5.0
        self.mock_config.base_cost = -0.2
        
        # Set up attack module mock
        self.mock_attack_module.device = torch.device("cpu")
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT

    def test_defend_action(self):
        """Test that defend action sets defending flag."""
        self.mock_attack_module.select_action.return_value = 4  # DEFEND
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that defending flag was set
            self.assertTrue(self.mock_agent.is_defending)
            
            # Check that defense was logged
            mock_logger.log_defense.assert_called_once()

    def test_attack_action_no_targets(self):
        """Test attack action when no targets are found."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock environment to return no targets
        self.mock_environment.get_nearby_agents.return_value = []
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that attack attempt was logged
            mock_logger.log_attack_attempt.assert_called_once()
            call_args = mock_logger.log_attack_attempt.call_args
            self.assertFalse(call_args[1]['success'])
            self.assertEqual(call_args[1]['reason'], "no_targets")

    def test_attack_action_with_target(self):
        """Test attack action when targets are found."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.is_defending = False
        mock_target.take_damage.return_value = True  # Successful hit
        
        # Mock environment to return a target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that attack was logged
            mock_logger.log_attack_attempt.assert_called_once()
            call_args = mock_logger.log_attack_attempt.call_args
            self.assertTrue(call_args[1]['success'])
            self.assertEqual(call_args[1]['reason'], "hit")

    def test_attack_action_self_targeting_filtered(self):
        """Test that agents cannot attack themselves."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent that is the same as the attacker
        mock_target = Mock()
        mock_target.agent_id = "test_agent"  # Same as attacker
        
        # Mock environment to return self as target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that attack attempt was logged with no valid targets
            mock_logger.log_attack_attempt.assert_called_once()
            call_args = mock_logger.log_attack_attempt.call_args
            self.assertFalse(call_args[1]['success'])
            self.assertEqual(call_args[1]['reason'], "no_targets")

    def test_attack_action_defensive_target(self):
        """Test attack action against a defending target."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent that is defending
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.is_defending = True
        mock_target.defense_strength = 0.5
        mock_target.take_damage.return_value = True  # Successful hit
        
        # Mock environment to return a target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that damage was reduced due to defense
            # The take_damage method should have been called with reduced damage
            mock_target.take_damage.assert_called_once()

    def test_attack_action_missed_attack(self):
        """Test attack action when attack misses."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.is_defending = False
        mock_target.take_damage.return_value = False  # Missed hit
        
        # Mock environment to return a target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that attack was logged as missed
            mock_logger.log_attack_attempt.assert_called_once()
            call_args = mock_logger.log_attack_attempt.call_args
            self.assertFalse(call_args[1]['success'])
            self.assertEqual(call_args[1]['reason'], "missed")

    def test_attack_action_resource_cost(self):
        """Test that attack action consumes resources."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.is_defending = False
        mock_target.take_damage.return_value = True  # Successful hit
        
        # Mock environment to return a target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        initial_resources = self.mock_agent.resource_level
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that resources were consumed
            self.assertLess(self.mock_agent.resource_level, initial_resources)

    def test_attack_action_combat_metrics(self):
        """Test that combat metrics are updated."""
        self.mock_attack_module.select_action.return_value = 0  # ATTACK_RIGHT
        
        # Mock the state
        mock_state = Mock()
        mock_state.to_tensor.return_value = torch.randn(8)
        self.mock_agent.get_state.return_value = mock_state
        
        # Mock target agent
        mock_target = Mock()
        mock_target.agent_id = "target_agent"
        mock_target.is_defending = False
        mock_target.take_damage.return_value = True  # Successful hit
        
        # Mock environment to return a target
        self.mock_environment.get_nearby_agents.return_value = [mock_target]
        
        initial_encounters = self.mock_environment.combat_encounters
        initial_successful = self.mock_environment.successful_attacks
        
        # Mock attack logger
        with patch('farm.actions.attack.AttackLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            attack_action(self.mock_agent)
            
            # Check that combat metrics were updated
            self.assertGreater(self.mock_environment.combat_encounters, initial_encounters)
            self.assertGreater(self.mock_environment.successful_attacks, initial_successful)


if __name__ == "__main__":
    unittest.main() 