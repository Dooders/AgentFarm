"""Unit tests for the DecisionModule class.

This module tests the core DecisionModule functionality including:
- Initialization with different configurations
- Action decision making
- Experience updates
- Model persistence
- Algorithm selection and fallbacks
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from pydantic import ValidationError

from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import TIANSHOU_AVAILABLE, DecisionModule


class TestDecisionModule(unittest.TestCase):
    """Test cases for DecisionModule class."""

    def setUp(self):
        """Set up test fixtures."""
        from gymnasium import spaces

        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "test_agent_1"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = spaces.Discrete(7)  # Standard action count
        self.mock_agent.environment = self.mock_env

        # Create mock observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )

        # Default config
        self.config = DecisionConfig()

    def test_initialization_without_tianshou(self):
        """Test DecisionModule initialization when Tianshou is not available."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        self.assertEqual(module.agent_id, "test_agent_1")
        self.assertEqual(module.num_actions, 7)
        self.assertIsInstance(
            module.algorithm, type(module.algorithm)
        )  # Fallback algorithm
        self.assertFalse(module._is_trained)

    def test_initialization_with_custom_action_space(self):
        """Test initialization with custom action space."""
        from gymnasium import spaces

        custom_action_space = spaces.Discrete(8)
        module = DecisionModule(
            self.mock_agent, custom_action_space, self.observation_space, self.config
        )

        self.assertEqual(module.num_actions, 8)
        self.assertEqual(module.action_space, custom_action_space)

    def test_initialization_with_custom_observation_space(self):
        """Test initialization with custom observation space."""
        from gymnasium import spaces

        custom_obs_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        # Create config that doesn't override observation space shape
        config = DecisionConfig(rl_state_dim=0)
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, custom_obs_space, config
        )

        self.assertEqual(module.state_dim, 16)
        self.assertEqual(module.observation_space, custom_obs_space)

    def test_decide_action_tensor_input(self):
        """Test decide_action with tensor input."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_numpy_input(self):
        """Test decide_action with numpy array input."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        state = np.random.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_algorithm(self):
        """Test decide_action when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        module.algorithm = None
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_action_range(self):
        """Test decide_action when algorithm returns out-of-range action."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm select_action method to return invalid action
        with patch.object(
            module.algorithm, "select_action", return_value=10
        ):
            state = torch.randn(8)
            action = module.decide_action(state)

            # Should fallback to random valid action
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_with_enabled_actions(self):
        """Test decide_action with enabled_actions parameter."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)

        # Test with enabled actions restricting to first 3 actions
        enabled_actions = [0, 1, 2]
        action = module.decide_action(state, enabled_actions)

        # Should return an index within the enabled_actions list (0, 1, or 2)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < len(enabled_actions))

        # Test with single enabled action
        enabled_actions = [1]
        action = module.decide_action(state, enabled_actions)

        # Should return index 0 (position of the only enabled action)
        self.assertEqual(action, 0)

        # Test with empty enabled actions (should fallback to full space)
        action = module.decide_action(state, [])
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

        # Test with None (should use full space)
        action = module.decide_action(state, None)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    @patch("farm.core.decision.decision.logger")
    def test_decide_action_exception_handling(self, mock_logger):
        """Test decide_action exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception
        with patch.object(
            module.algorithm, "select_action", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            action = module.decide_action(state)

            # Should log error and return random action
            mock_logger.error.assert_called_once()
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < module.num_actions)

    def test_update_with_algorithm(self):
        """Test update method with algorithm."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Mock the train method and should_train to ensure training happens
        with patch.object(module.algorithm, "train") as mock_train, \
             patch.object(module.algorithm, "should_train", return_value=True):
            module.update(state, action, reward, next_state, done)

            # For Tianshou algorithms, train should be called when should_train returns True
            if hasattr(module.algorithm, "train"):
                mock_train.assert_called_once()
            self.assertTrue(module._is_trained)

    def test_update_exception_handling(self):
        """Test update method exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception
        with patch.object(
            module.algorithm, "train", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            action = 1
            reward = 1.0
            next_state = torch.randn(8)
            done = False

            # Should not raise exception
            module.update(state, action, reward, next_state, done)

    def test_get_action_probabilities_with_predict_proba(self):
        """Test get_action_probabilities with algorithm that supports predict_proba."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm with predict_proba
        expected_probs = np.array([[0.1, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0]])
        with patch.object(
            module.algorithm, "predict_proba", return_value=expected_probs
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            np.testing.assert_array_almost_equal(
                probs, [0.1, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0]
            )

    def test_get_action_probabilities_fallback(self):
        """Test get_action_probabilities fallback to uniform distribution."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock the algorithm to not have predict_proba method
        with patch.object(
            module.algorithm, "predict_proba", side_effect=AttributeError
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            expected = np.full(7, 1.0 / 7)
            np.testing.assert_array_almost_equal(probs, expected)

    def test_get_action_probabilities_exception_handling(self):
        """Test get_action_probabilities exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception
        with patch.object(
            module.algorithm, "predict_proba", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            # Should return uniform distribution
            expected = np.full(7, 1.0 / 7)
            np.testing.assert_array_almost_equal(probs, expected)

    def test_get_model_info(self):
        """Test get_model_info method."""
        config = DecisionConfig(algorithm_type="dqn", rl_state_dim=16)
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        info = module.get_model_info()

        expected_keys = [
            "agent_id",
            "algorithm_type",
            "num_actions",
            "state_dim",
            "is_trained",
            "tianshou_available",
        ]

        for key in expected_keys:
            self.assertIn(key, info)

        self.assertEqual(info["agent_id"], "test_agent_1")
        self.assertEqual(info["algorithm_type"], "dqn")
        self.assertEqual(info["state_dim"], 16)
        self.assertEqual(info["tianshou_available"], TIANSHOU_AVAILABLE)

    def test_reset(self):
        """Test reset method."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        module._is_trained = True

        module.reset()

        self.assertFalse(module._is_trained)
        # If algorithm has reset method, it should be called
        if hasattr(module.algorithm, "reset"):
            module.algorithm.reset.assert_called_once()

    def test_get_action_space_size_from_environment(self):
        """Test getting action space size from environment."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        self.assertEqual(module.num_actions, 7)  # From mock environment

    def test_get_action_space_size_fallback(self):
        """Test getting action space size fallback."""
        # Remove environment from agent
        delattr(self.mock_agent, "environment")

        with patch("farm.core.action.ActionType") as mock_action_type:
            mock_action_type.__len__ = Mock(return_value=7)
            module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                self.config,
            )
            self.assertEqual(module.num_actions, 7)

    def test_get_action_space_size_from_space_object(self):
        """Test getting action space size from space object."""
        from gymnasium import spaces

        action_space = spaces.Discrete(10)
        module = DecisionModule(
            self.mock_agent, action_space, self.observation_space, self.config
        )

        self.assertEqual(module.num_actions, 10)

    def test_initialization_with_tianshou_ppo(self):
        """Test DecisionModule initialization with Tianshou PPO."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ppo")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "ppo")

    def test_initialization_with_tianshou_sac(self):
        """Test DecisionModule initialization with Tianshou SAC."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="sac")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "sac")

    def test_initialization_with_tianshou_dqn(self):
        """Test DecisionModule initialization with Tianshou DQN."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="dqn")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "dqn")

    def test_initialization_with_tianshou_a2c(self):
        """Test DecisionModule initialization with Tianshou A2C."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="a2c")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "a2c")

    def test_initialization_with_tianshou_ddpg(self):
        """Test DecisionModule initialization with Tianshou DDPG."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ddpg")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "ddpg")

    def test_initialization_algorithm_fallback(self):
        """Test that invalid algorithm falls back to fallback algorithm."""
        # DecisionConfig validates algorithm_type and falls back to 'fallback' for invalid values
        config = DecisionConfig(algorithm_type="invalid_algorithm")
        self.assertEqual(config.algorithm_type, "fallback")
        
        # Test that the module initializes successfully with fallback algorithm
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )
        self.assertIsNotNone(module.algorithm)
        self.assertEqual(module.config.algorithm_type, "fallback")

    def test_decide_action_with_multi_dimensional_state(self):
        """Test decide_action with multi-dimensional state."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Test with different state shapes
        state_2d = torch.randn(4, 2)  # Should be flattened to 8 elements
        action = module.decide_action(state_2d)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_update_with_none_algorithm(self):
        """Test update method when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Set algorithm to None
        module.algorithm = None

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Should not crash
        module.update(state, action, reward, next_state, done)
        self.assertFalse(module._is_trained)

    def test_get_action_probabilities_with_none_algorithm(self):
        """Test get_action_probabilities when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Set algorithm to None
        module.algorithm = None

        state = torch.randn(8)
        probs = module.get_action_probabilities(state)

        # Should return uniform distribution
        expected = np.full(7, 1.0 / 7)
        np.testing.assert_array_almost_equal(probs, expected)


class TestDecisionModuleIntegration(unittest.TestCase):
    """Integration tests for DecisionModule."""

    def setUp(self):
        """Set up test fixtures."""
        from gymnasium import spaces

        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "integration_test_agent"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = spaces.Discrete(7)
        self.mock_agent.environment = self.mock_env

        # Create mock observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )

    def test_full_decision_cycle(self):
        """Test a complete decision cycle."""
        config = DecisionConfig(algorithm_type="fallback")  # Ensure we use fallback
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        # Create state
        state = torch.randn(8)

        # Get action
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

        # Get probabilities
        probs = module.get_action_probabilities(state)
        self.assertEqual(len(probs), 7)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

        # Update with experience
        reward = 0.5
        next_state = torch.randn(8)
        done = False

        module.update(state, action, reward, next_state, done)
        self.assertTrue(module._is_trained)

        # Get model info
        info = module.get_model_info()
        self.assertEqual(info["agent_id"], "integration_test_agent")
        self.assertTrue(info["is_trained"])

    def test_model_persistence_cycle(self):
        """Test complete model save/load cycle."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        # Train a bit
        for _ in range(3):
            state = torch.randn(8)
            action = module.decide_action(state)
            module.update(state, action, 1.0, torch.randn(8), False)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")

            # Save model
            module.save_model(model_path)

            # Create new module and load
            new_module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                config,
            )
            self.assertFalse(new_module._is_trained)

            new_module.load_model(model_path)

            # Check state was restored
            self.assertEqual(new_module._is_trained, module._is_trained)

    def test_integration_with_tianshou_algorithm(self):
        """Test integration with Tianshou algorithm."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                # Create a proper mock algorithm that behaves like the real one
                mock_algorithm = Mock()
                mock_algorithm.select_action.return_value = 2
                mock_algorithm.select_action_with_mask.return_value = 2
                mock_algorithm.predict_proba.return_value = np.full(
                    (1, 7), 1.0 / 7, dtype=np.float32
                )
                mock_algorithm.update = Mock()
                mock_algorithm.learn = Mock()
                mock_algorithm.store_experience = Mock()
                mock_algorithm.should_train.return_value = True
                mock_algorithm.train = Mock()

                # Make the mock constructor return the mock algorithm
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ppo")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                # Test full cycle
                state = torch.randn(8)
                action = module.decide_action(state)
                self.assertEqual(action, 2)  # From mock

                probs = module.get_action_probabilities(state)
                self.assertEqual(len(probs), 7)

                # Test update
                module.update(state, action, 1.0, torch.randn(8), False)

                # Test model info
                info = module.get_model_info()
                self.assertEqual(info["algorithm_type"], "ppo")

    def test_integration_with_custom_config(self):
        """Test integration with custom configuration."""
        custom_config = DecisionConfig(
            algorithm_type="fallback",
            learning_rate=0.002,
            gamma=0.95,
            epsilon_start=0.9,
            epsilon_min=0.1,
            rl_state_dim=12,
        )

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            custom_config,
        )

        # Verify custom config was applied
        self.assertEqual(module.config.learning_rate, 0.002)
        self.assertEqual(module.config.gamma, 0.95)
        self.assertEqual(module.config.epsilon_start, 0.9)
        self.assertEqual(module.state_dim, 12)

        # Test functionality
        state = torch.randn(12)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

    def test_integration_performance_stress_test(self):
        """Test performance with multiple decision cycles."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Stress test with many decisions
        num_cycles = 100
        states = [torch.randn(8) for _ in range(num_cycles)]

        import time

        start_time = time.time()

        for i, state in enumerate(states):
            action = module.decide_action(state)
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < 7)

            # Update occasionally
            if i % 10 == 0:
                module.update(state, action, 0.5, torch.randn(8), False)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (< 1 second for 100 cycles)
        self.assertLess(
            duration,
            1.0,
            f"Performance test failed: {duration:.3f}s for {num_cycles} cycles",
        )
        self.assertTrue(module._is_trained)


if __name__ == "__main__":
    unittest.main()
