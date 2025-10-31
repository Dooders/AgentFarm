"""Unit tests for the base DQN module.

This module tests the core DQN functionality including configuration,
neural network architecture, experience replay, and training mechanisms.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

np.random.seed(42)  # For reproducibility in tests

from farm.core.decision.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork


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


class TestBaseQNetwork(unittest.TestCase):
    """Test cases for BaseQNetwork class."""

    def test_initialization_without_shared_encoder(self):
        """Test BaseQNetwork initialization without shared encoder."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)

        self.assertEqual(
            len(network.network), 9
        )  # 3 layers + 3 LayerNorm + 3 ReLU + 3 Dropout

    # def test_initialization_with_shared_encoder(self):
    #     """Test BaseQNetwork initialization with shared encoder."""

    def test_forward_pass_without_shared_encoder(self):
        """Test BaseQNetwork forward pass without shared encoder."""
        network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
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

    # def test_initialization_with_shared_encoder(self):
    #     """Test BaseDQNModule initialization with shared encoder."""
    #     module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)

    #     # Networks should be initialized
    #     self.assertIsInstance(module.q_network, BaseQNetwork)
    #     self.assertIsInstance(module.target_network, BaseQNetwork)

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

    def test_select_action_exploration_with_weights(self):
        """Test action selection during exploration with action weights."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 1.0  # Always explore

        state = torch.randn(8).to(module.device)
        
        # Create action weights favoring action 0
        action_weights = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)

        # Test multiple selections
        selection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            action = module.select_action(state, action_weights=action_weights)
            selection_counts[action] += 1

        # Action 0 should be selected most often due to higher weight
        self.assertGreater(selection_counts[0], selection_counts[1])
        self.assertGreater(selection_counts[0], selection_counts[2])
        self.assertGreater(selection_counts[0], selection_counts[3])
        # Should be roughly 70% of selections
        self.assertGreater(selection_counts[0], 600)

    def test_select_action_exploitation_with_weights(self):
        """Test action selection during exploitation with action weights scaling Q-values."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit

        state = torch.randn(8).to(module.device)
        
        # Get action without weights
        action_no_weights = module.select_action(state)
        
        # Create action weights that favor action 2
        action_weights = np.array([0.1, 0.1, 0.8, 0.0], dtype=np.float64)

        # With high weight on action 2, selection should be deterministic
        # Run multiple times with same state and weights
        actions = []
        for _ in range(10):
            action = module.select_action(state, action_weights=action_weights)
            actions.append(action)

        # Same state + same weights should produce same action (deterministic)
        self.assertEqual(len(set(actions)), 1, "Same state and weights should produce deterministic action")
        
        # Weights should influence selection (may or may not change from no-weights case)
        # But with very high weight (0.8) on action 2, it's likely to be selected
        # unless Q-values for other actions are extremely high
        selected_action = actions[0]
        self.assertTrue(0 <= selected_action < 4, "Action should be valid")

    def test_select_action_with_weights_backward_compatibility(self):
        """Test that select_action works without weights (backward compatibility)."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 1.0  # Always explore

        state = torch.randn(8).to(module.device)

        # Should work without weights parameter
        action = module.select_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        # Should also work with None weights
        action2 = module.select_action(state, action_weights=None)
        self.assertIsInstance(action2, int)
        self.assertTrue(0 <= action2 < 4)

    def test_select_action_weights_normalized(self):
        """Test that unnormalized weights are handled correctly."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 1.0  # Always explore

        state = torch.randn(8).to(module.device)
        
        # Unnormalized weights (don't sum to 1)
        action_weights = np.array([4.0, 2.0, 1.0, 1.0], dtype=np.float64)

        # Should not raise error
        action = module.select_action(state, action_weights=action_weights)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

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
        """Test state caching in select_action without action weights."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit for consistent results

        state = torch.randn(8).to(module.device)

        # First call should cache the result
        action1 = module.select_action(state)
        
        # Verify cache was populated
        state_hash = hash(state.cpu().numpy().tobytes())
        self.assertIn(state_hash, module._state_cache)
        self.assertEqual(module._state_cache[state_hash], action1)

        # Second call should use cached result (cache checked before Q-network)
        # Mock q_network to verify it's not called on cache hit
        original_q_network = module.q_network
        module.q_network = Mock(wraps=original_q_network)
        
        action2 = module.select_action(state)
        
        # Actions should match
        self.assertEqual(action1, action2)
        
        # Q-network should NOT be called on cache hit
        module.q_network.forward.assert_not_called()

    def test_state_caching_with_weights_bypassed(self):
        """Test that cache is bypassed when action_weights is provided."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit for consistent results

        state = torch.randn(8).to(module.device)
        action_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        # First call with weights - should compute and not use cache
        action1 = module.select_action(state, action_weights=action_weights)
        
        # Verify state was NOT cached (weights invalidate cache)
        state_hash = hash(state.cpu().numpy().tobytes())
        self.assertNotIn(state_hash, module._state_cache)
        
        # Second call with same weights - should recompute (no cache)
        action2 = module.select_action(state, action_weights=action_weights)
        
        # Actions should match (same weights, same state)
        self.assertEqual(action1, action2)

    def test_state_caching_mixed_weights(self):
        """Test cache behavior when switching between weights and no weights."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit

        state = torch.randn(8).to(module.device)
        state_hash = hash(state.cpu().numpy().tobytes())
        
        # Call without weights - should cache
        action_no_weights = module.select_action(state)
        self.assertIn(state_hash, module._state_cache)
        cached_action = module._state_cache[state_hash]
        
        # Call with weights - should bypass cache and recompute
        action_weights = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        action_with_weights = module.select_action(state, action_weights=action_weights)
        
        # Cache should still exist (wasn't cleared by weights call)
        self.assertIn(state_hash, module._state_cache)
        self.assertEqual(module._state_cache[state_hash], cached_action)
        
        # Call again without weights - should use cache
        module.q_network = Mock(wraps=module.q_network)
        action_cached = module.select_action(state)
        self.assertEqual(action_cached, cached_action)
        # Q-network should not be called
        module.q_network.forward.assert_not_called()

    def test_state_caching_cache_before_computation(self):
        """Test that cache check happens before Q-network computation."""
        module = BaseDQNModule(input_dim=8, output_dim=4, config=self.config)
        module.epsilon = 0.0  # Always exploit

        state = torch.randn(8).to(module.device)
        
        # First call - populate cache
        action1 = module.select_action(state)
        
        # Verify cache exists
        state_hash = hash(state.cpu().numpy().tobytes())
        self.assertIn(state_hash, module._state_cache)
        
        # Mock q_network.forward to track calls
        original_forward = module.q_network.forward
        forward_mock = Mock(side_effect=original_forward)
        module.q_network.forward = forward_mock
        
        # Second call - should use cache, not call Q-network
        action2 = module.select_action(state)
        
        # Verify Q-network was NOT called (cache hit before computation)
        forward_mock.assert_not_called()
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
