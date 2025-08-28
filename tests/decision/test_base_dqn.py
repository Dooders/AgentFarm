"""Unit tests for the base DQN module.

This module tests the core DQN functionality including configuration,
neural network architecture, experience replay, and training mechanisms.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from farm.core.decision.base_dqn import (
    DEVICE,
    BaseDQNConfig,
    BaseDQNModule,
    BaseQNetwork,
    SharedEncoder,
)


class TestBaseDQNConfig(unittest.TestCase):
    """Test cases for BaseDQNConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BaseDQNConfig()

        self.assertEqual(config.target_update_freq, 100)
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.epsilon_start, 1.0)
        self.assertEqual(config.epsilon_min, 0.01)
        self.assertEqual(config.epsilon_decay, 0.995)
        self.assertEqual(config.dqn_hidden_size, 64)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.tau, 0.005)
        self.assertIsNone(config.seed)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BaseDQNConfig()

        # Modify attributes directly
        config.target_update_freq = 50
        config.memory_size = 5000
        config.learning_rate = 0.0005
        config.gamma = 0.95
        config.epsilon_start = 0.8
        config.epsilon_min = 0.05
        config.epsilon_decay = 0.99
        config.dqn_hidden_size = 128
        config.batch_size = 64
        config.tau = 0.01
        config.seed = 42

        self.assertEqual(config.target_update_freq, 50)
        self.assertEqual(config.memory_size, 5000)
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.epsilon_start, 0.8)
        self.assertEqual(config.epsilon_min, 0.05)
        self.assertEqual(config.epsilon_decay, 0.99)
        self.assertEqual(config.dqn_hidden_size, 128)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.tau, 0.01)
        self.assertEqual(config.seed, 42)


class TestSharedEncoder(unittest.TestCase):
    """Test cases for SharedEncoder class."""

    def test_initialization(self):
        """Test SharedEncoder initialization."""
        encoder = SharedEncoder(input_dim=8, hidden_size=64)

        self.assertEqual(encoder.input_dim, 8)
        self.assertEqual(encoder.hidden_size, 64)
        self.assertIsInstance(encoder.fc, torch.nn.Linear)
        self.assertEqual(encoder.fc.in_features, 8)
        self.assertEqual(encoder.fc.out_features, 64)

    def test_forward_pass(self):
        """Test SharedEncoder forward pass."""
        encoder = SharedEncoder(input_dim=8, hidden_size=64)
        x = torch.randn(2, 8)  # Batch size 2, input dim 8

        output = encoder(x)

        self.assertEqual(output.shape, (2, 64))
        self.assertTrue(torch.all(output >= 0))  # ReLU activation

    def test_forward_pass_single_sample(self):
        """Test SharedEncoder forward pass with single sample."""
        encoder = SharedEncoder(input_dim=8, hidden_size=64)
        x = torch.randn(8)  # Single sample

        output = encoder(x)

        self.assertEqual(output.shape, (64,))
        self.assertTrue(torch.all(output >= 0))  # ReLU activation


class TestBaseQNetwork(unittest.TestCase):
    """Test cases for BaseQNetwork class."""

    def test_initialization_without_shared_encoder(self):
        """Test BaseQNetwork initialization without shared encoder."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)

        self.assertIsNone(network.shared_encoder)
        self.assertEqual(
            len(network.network), 9
        )  # 3 layers + 3 LayerNorm + 3 ReLU + 3 Dropout

    def test_initialization_with_shared_encoder(self):
        """Test BaseQNetwork initialization with shared encoder."""
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = BaseQNetwork(
            input_dim=8, output_dim=4, hidden_size=64, shared_encoder=shared_encoder
        )

        self.assertIs(network.shared_encoder, shared_encoder)

    def test_forward_pass_without_shared_encoder(self):
        """Test BaseQNetwork forward pass without shared encoder."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
        x = torch.randn(2, 8)

        output = network(x)

        self.assertEqual(output.shape, (2, 4))

    def test_forward_pass_with_shared_encoder(self):
        """Test BaseQNetwork forward pass with shared encoder."""
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = BaseQNetwork(
            input_dim=8, output_dim=4, hidden_size=64, shared_encoder=shared_encoder
        )
        x = torch.randn(2, 8)

        output = network(x)

        self.assertEqual(output.shape, (2, 4))

    def test_forward_pass_single_sample(self):
        """Test BaseQNetwork forward pass with single sample."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
        x = torch.randn(8)  # Single sample

        output = network(x)

        self.assertEqual(output.shape, (4,))

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)

        # Check that weights are not all zeros
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertFalse(torch.all(module.weight == 0))
                self.assertTrue(
                    torch.all(module.bias == 0)
                )  # Bias should be initialized to 0


class TestBaseDQNModule(unittest.TestCase):
    """Test cases for BaseDQNModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BaseDQNConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9
        self.config.seed = 42

    def test_initialization(self):
        """Test BaseDQNModule initialization."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        self.assertEqual(module.output_dim, 4)
        self.assertEqual(module.config, self.config)
        self.assertEqual(module.epsilon, 1.0)
        self.assertEqual(module.epsilon_min, 0.01)
        self.assertEqual(module.epsilon_decay, 0.9)
        self.assertEqual(module.gamma, 0.99)
        self.assertEqual(module.tau, 0.005)
        self.assertEqual(len(module.memory), 0)
        self.assertEqual(module.memory.maxlen, 100)

    def test_initialization_with_seed(self):
        """Test BaseDQNModule initialization with seed."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Check that networks are initialized
        self.assertIsInstance(module.q_network, BaseQNetwork)
        self.assertIsInstance(module.target_network, BaseQNetwork)
        self.assertIsInstance(module.optimizer, torch.optim.Adam)

    def test_initialization_with_shared_encoder(self):
        """Test BaseDQNModule initialization with shared encoder."""
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Networks should be initialized
        self.assertIsInstance(module.q_network, BaseQNetwork)
        self.assertIsInstance(module.target_network, BaseQNetwork)

    def test_store_experience(self):
        """Test storing experiences in memory."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        state = torch.randn(8).to(module.device)
        action = 2
        reward = 1.0
        next_state = torch.randn(8).to(module.device)
        done = False

        module.store_experience(state, action, reward, next_state, done)

        self.assertEqual(len(module.memory), 1)
        stored_experience = module.memory[0]
        # Use torch.equal for tensor comparison
        self.assertTrue(torch.equal(stored_experience[0], state))
        self.assertEqual(stored_experience[1], action)
        self.assertEqual(stored_experience[2], reward)
        self.assertTrue(torch.equal(stored_experience[3], next_state))
        self.assertEqual(stored_experience[4], done)

    def test_store_experience_with_logging(self):
        """Test storing experiences with database logging."""
        mock_db = Mock()
        mock_logger = Mock()
        mock_db.logger = mock_logger

        module = BaseDQNModule(
            input_dim=8, output_dim=4, config=self.config, db=mock_db
        )

        state = torch.randn(8).to(module.device)
        action = 2
        reward = 1.0
        next_state = torch.randn(8).to(module.device)
        done = False

        module.store_experience(
            state,
            action,
            reward,
            next_state,
            done,
            step_number=10,
            agent_id="agent1",
            module_type="test",
            module_id=123,
            action_taken_mapped=2,
        )

        # Check that logger was called
        mock_logger.log_learning_experience.assert_called_once()

    def test_train_with_small_batch(self):
        """Test training with batch smaller than batch_size."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Add some experiences
        for i in range(2):  # Less than batch_size (4)
            state = torch.randn(8).to(module.device)
            action = i
            reward = 1.0
            next_state = torch.randn(8).to(module.device)
            done = False
            module.store_experience(state, action, reward, next_state, done)

        # Try to train
        loss = module.train(list(module.memory))

        # Should return None for small batch
        self.assertIsNone(loss)

    def test_train_with_sufficient_batch(self):
        """Test training with sufficient batch size."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Add experiences
        for i in range(4):  # Equal to batch_size
            state = torch.randn(8).to(module.device)
            action = i % 4
            reward = 1.0
            next_state = torch.randn(8).to(module.device)
            done = False
            module.store_experience(state, action, reward, next_state, done)

        # Train
        loss = module.train(list(module.memory))

        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        # After the above assertions, loss is guaranteed to be a float
        assert loss is not None  # type: ignore[unreachable]
        self.assertGreater(loss, 0.0)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
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
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit

        state = torch.randn(8).to(module.device)

        # Should always return the same action for the same state
        action1 = module.select_action(state)
        action2 = module.select_action(state)

        self.assertEqual(action1, action2)

    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        initial_epsilon = module.epsilon

        # Train to trigger epsilon decay
        for i in range(4):
            state = torch.randn(8).to(module.device)
            action = i % 4
            reward = 1.0
            next_state = torch.randn(8).to(module.device)
            done = False
            module.store_experience(state, action, reward, next_state, done)

        module.train(list(module.memory))

        # Epsilon should have decreased
        self.assertLess(module.epsilon, initial_epsilon)

    def test_soft_update_target_network(self):
        """Test soft update of target network."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Get initial target network weights
        initial_target_weights = {}
        for name, param in module.target_network.named_parameters():
            initial_target_weights[name] = param.data.clone()

        # Modify main network weights
        for param in module.q_network.parameters():
            param.data += torch.randn_like(param.data) * 0.1

        # Perform soft update
        module._soft_update_target_network()

        # Check that target network weights have changed
        for name, param in module.target_network.named_parameters():
            if name in initial_target_weights:
                self.assertFalse(
                    torch.allclose(param.data, initial_target_weights[name])
                )

    def test_get_state_dict(self):
        """Test getting state dictionary."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        state_dict = module.get_state_dict()

        required_keys = [
            "q_network_state",
            "target_network_state",
            "optimizer_state",
            "epsilon",
            "steps",
            "losses",
            "episode_rewards",
            "seed",
        ]

        for key in required_keys:
            self.assertIn(key, state_dict)

    def test_load_state_dict(self):
        """Test loading state dictionary."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Get current state
        original_state = module.get_state_dict()

        # Modify some values
        original_state["epsilon"] = 0.5
        original_state["steps"] = 100

        # Load the modified state
        module.load_state_dict(original_state)

        # Check that values were updated
        self.assertEqual(module.epsilon, 0.5)
        self.assertEqual(module.steps, 100)

    def test_cleanup(self):
        """Test cleanup method."""
        mock_db = Mock()
        module = BaseDQNModule(
            input_dim=8, output_dim=4, config=self.config, db=mock_db
        )

        # Add some pending experiences
        module.pending_experiences = [Mock(), Mock()]

        # Cleanup should not raise an exception
        module.cleanup()

    def test_state_caching(self):
        """Test state caching in select_action."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit for consistent results

        state = torch.randn(8).to(module.device)

        # First call should cache the result
        action1 = module.select_action(state)

        # Second call should use cached result
        action2 = module.select_action(state)

        self.assertEqual(action1, action2)

    def test_memory_overflow(self):
        """Test that memory respects maxlen."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

        # Add more experiences than memory size
        for i in range(150):  # More than memory_size (100)
            state = torch.randn(8).to(module.device)
            action = i % 4
            reward = 1.0
            next_state = torch.randn(8).to(module.device)
            done = False
            module.store_experience(state, action, reward, next_state, done)

        # Memory should not exceed maxlen
        self.assertLessEqual(len(module.memory), 100)


if __name__ == "__main__":
    unittest.main()
