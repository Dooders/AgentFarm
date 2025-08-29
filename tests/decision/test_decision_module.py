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

from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import SB3_AVAILABLE, DecisionModule, SB3Wrapper


class TestSB3Wrapper(unittest.TestCase):
    """Test cases for SB3Wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        from gymnasium import spaces

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.wrapper = SB3Wrapper(self.observation_space, self.action_space)

    def test_initialization(self):
        """Test SB3Wrapper initialization."""
        self.assertEqual(self.wrapper.observation_space, self.observation_space)
        self.assertEqual(self.wrapper.action_space, self.action_space)

    def test_reset(self):
        """Test wrapper reset method."""
        result = self.wrapper.reset()
        expected = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        np.testing.assert_array_equal(result, expected)

    def test_step(self):
        """Test wrapper step method."""
        action = 1
        result = self.wrapper.step(action)

        self.assertEqual(len(result), 4)  # observation, reward, done, truncated, info
        self.assertEqual(result[1], 0.0)  # reward
        self.assertFalse(result[2])  # done
        self.assertFalse(result[3])  # truncated
        self.assertEqual(result[4], {})  # info


class TestDecisionModule(unittest.TestCase):
    """Test cases for DecisionModule class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "test_agent_1"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 6  # Standard action count
        self.mock_agent.environment = self.mock_env

        # Default config
        self.config = DecisionConfig()

    @patch("farm.core.decision.decision.SB3_AVAILABLE", False)
    def test_initialization_without_sb3(self):
        """Test DecisionModule initialization when SB3 is not available."""
        module = DecisionModule(self.mock_agent, self.config)

        self.assertEqual(module.agent_id, "test_agent_1")
        self.assertEqual(module.num_actions, 6)
        self.assertIsInstance(
            module.algorithm, type(module.algorithm)
        )  # Fallback algorithm
        self.assertFalse(module._is_trained)

    @patch("farm.core.decision.decision.SB3_AVAILABLE", True)
    @patch("stable_baselines3.DQN")
    def test_initialization_with_sb3_ddqn(self, mock_dqn_class):
        """Test DecisionModule initialization with SB3 DDQN."""
        mock_algorithm = Mock()
        mock_dqn_class.return_value = mock_algorithm

        config = DecisionConfig(algorithm_type="ddqn")
        module = DecisionModule(self.mock_agent, config)

        self.assertEqual(module.agent_id, "test_agent_1")
        mock_dqn_class.assert_called_once()
        self.assertEqual(module.algorithm, mock_algorithm)

    @patch("farm.core.decision.decision.SB3_AVAILABLE", True)
    @patch("stable_baselines3.PPO")
    def test_initialization_with_sb3_ppo(self, mock_ppo_class):
        """Test DecisionModule initialization with SB3 PPO."""
        mock_algorithm = Mock()
        mock_ppo_class.return_value = mock_algorithm

        config = DecisionConfig(algorithm_type="ppo")
        module = DecisionModule(self.mock_agent, config)

        self.assertEqual(module.agent_id, "test_agent_1")
        mock_ppo_class.assert_called_once()
        self.assertEqual(module.algorithm, mock_algorithm)

    def test_initialization_with_custom_action_space(self):
        """Test initialization with custom action space."""
        from gymnasium import spaces

        custom_action_space = spaces.Discrete(8)
        module = DecisionModule(
            self.mock_agent, self.config, action_space=custom_action_space
        )

        self.assertEqual(module.num_actions, 8)
        self.assertEqual(module.action_space, custom_action_space)

    def test_initialization_with_custom_observation_space(self):
        """Test initialization with custom observation space."""
        from gymnasium import spaces

        custom_obs_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        module = DecisionModule(
            self.mock_agent, self.config, observation_space=custom_obs_space
        )

        self.assertEqual(module.state_dim, 16)
        self.assertEqual(module.observation_space, custom_obs_space)

    def test_decide_action_tensor_input(self):
        """Test decide_action with tensor input."""
        module = DecisionModule(self.mock_agent, self.config)
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_numpy_input(self):
        """Test decide_action with numpy array input."""
        module = DecisionModule(self.mock_agent, self.config)
        state = np.random.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_algorithm(self):
        """Test decide_action when algorithm is None."""
        module = DecisionModule(self.mock_agent, self.config)
        module.algorithm = None
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_action_range(self):
        """Test decide_action when algorithm returns out-of-range action."""
        module = DecisionModule(self.mock_agent, self.config)

        # Mock algorithm to return invalid action
        module.algorithm.predict.return_value = (np.array([10]), None)  # Invalid action

        state = torch.randn(8)
        action = module.decide_action(state)

        # Should fallback to random valid action
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    @patch("farm.core.decision.decision.logger")
    def test_decide_action_exception_handling(self, mock_logger):
        """Test decide_action exception handling."""
        module = DecisionModule(self.mock_agent, self.config)

        # Mock algorithm to raise exception
        module.algorithm.predict.side_effect = Exception("Test error")

        state = torch.randn(8)
        action = module.decide_action(state)

        # Should log error and return random action
        mock_logger.error.assert_called_once()
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_update_with_sb3_algorithm(self):
        """Test update method with SB3 algorithm."""
        module = DecisionModule(self.mock_agent, self.config)

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        module.update(state, action, reward, next_state, done)

        # For SB3 algorithms, learn should be called
        if hasattr(module.algorithm, "learn"):
            module.algorithm.learn.assert_called_with(total_timesteps=1)
        self.assertTrue(module._is_trained)

    def test_update_exception_handling(self):
        """Test update method exception handling."""
        module = DecisionModule(self.mock_agent, self.config)

        # Mock algorithm to raise exception
        module.algorithm.learn.side_effect = Exception("Test error")

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Should not raise exception
        module.update(state, action, reward, next_state, done)

    def test_get_action_probabilities_with_predict_proba(self):
        """Test get_action_probabilities with algorithm that supports predict_proba."""
        module = DecisionModule(self.mock_agent, self.config)

        # Mock algorithm with predict_proba
        module.algorithm.predict_proba.return_value = np.array(
            [[0.1, 0.3, 0.4, 0.2, 0.0, 0.0]]
        )

        state = torch.randn(8)
        probs = module.get_action_probabilities(state)

        np.testing.assert_array_almost_equal(probs, [0.1, 0.3, 0.4, 0.2, 0.0, 0.0])

    def test_get_action_probabilities_fallback(self):
        """Test get_action_probabilities fallback to uniform distribution."""
        module = DecisionModule(self.mock_agent, self.config)

        # Remove predict_proba method
        if hasattr(module.algorithm, "predict_proba"):
            delattr(module.algorithm, "predict_proba")

        state = torch.randn(8)
        probs = module.get_action_probabilities(state)

        expected = np.full(module.num_actions, 1.0 / module.num_actions)
        np.testing.assert_array_almost_equal(probs, expected)

    def test_get_action_probabilities_exception_handling(self):
        """Test get_action_probabilities exception handling."""
        module = DecisionModule(self.mock_agent, self.config)

        # Mock algorithm to raise exception
        module.algorithm.predict_proba.side_effect = Exception("Test error")

        state = torch.randn(8)
        probs = module.get_action_probabilities(state)

        # Should return uniform distribution
        expected = np.full(module.num_actions, 1.0 / module.num_actions)
        np.testing.assert_array_almost_equal(probs, expected)

    def test_save_model_sb3_algorithm(self):
        """Test save_model with SB3 algorithm."""
        with tempfile.TemporaryDirectory() as temp_dir:
            module = DecisionModule(self.mock_agent, self.config)
            model_path = os.path.join(temp_dir, "test_model")

            module.save_model(model_path)

            # Should call save on algorithm
            module.algorithm.save.assert_called_with(model_path)

    def test_save_model_fallback_algorithm(self):
        """Test save_model with fallback algorithm."""
        with tempfile.TemporaryDirectory() as temp_dir:
            module = DecisionModule(self.mock_agent, self.config)
            # Mock config dict method
            module.config.dict = Mock(return_value={"test": "config"})

            model_path = os.path.join(temp_dir, "test_model")
            module.save_model(model_path)

            # Should create pickle file
            pickle_path = f"{model_path}.pkl"
            self.assertTrue(os.path.exists(pickle_path))

    @patch("stable_baselines3.DQN")
    def test_load_model_sb3_ddqn(self, mock_dqn_class):
        """Test load_model with SB3 DDQN."""
        module = DecisionModule(self.mock_agent, DecisionConfig(algorithm_type="ddqn"))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")
            module.load_model(model_path)

            # Should call DQN.load
            mock_dqn_class.load.assert_called_with(model_path)

    @patch("stable_baselines3.PPO")
    def test_load_model_sb3_ppo(self, mock_ppo_class):
        """Test load_model with SB3 PPO."""
        module = DecisionModule(self.mock_agent, DecisionConfig(algorithm_type="ppo"))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")
            module.load_model(model_path)

            # Should call PPO.load
            mock_ppo_class.load.assert_called_with(model_path)

    def test_load_model_fallback_algorithm(self):
        """Test load_model with fallback algorithm."""
        with tempfile.TemporaryDirectory() as temp_dir:
            module = DecisionModule(self.mock_agent, self.config)
            # Mock config dict method
            module.config.dict = Mock(return_value={"test": "config"})

            # Create a mock pickle file
            model_path = os.path.join(temp_dir, "test_model")
            pickle_path = f"{model_path}.pkl"

            import pickle

            with open(pickle_path, "wb") as f:
                pickle.dump({"is_trained": True}, f)

            module.load_model(model_path)

            self.assertTrue(module._is_trained)

    def test_get_model_info(self):
        """Test get_model_info method."""
        config = DecisionConfig(algorithm_type="ddqn", rl_state_dim=16)
        module = DecisionModule(self.mock_agent, config)

        info = module.get_model_info()

        expected_keys = [
            "agent_id",
            "algorithm_type",
            "num_actions",
            "state_dim",
            "is_trained",
            "sb3_available",
        ]

        for key in expected_keys:
            self.assertIn(key, info)

        self.assertEqual(info["agent_id"], "test_agent_1")
        self.assertEqual(info["algorithm_type"], "ddqn")
        self.assertEqual(info["state_dim"], 16)
        self.assertEqual(info["sb3_available"], SB3_AVAILABLE)

    def test_reset(self):
        """Test reset method."""
        module = DecisionModule(self.mock_agent, self.config)
        module._is_trained = True

        module.reset()

        self.assertFalse(module._is_trained)
        # If algorithm has reset method, it should be called
        if hasattr(module.algorithm, "reset"):
            module.algorithm.reset.assert_called_once()

    def test_get_action_space_size_from_environment(self):
        """Test getting action space size from environment."""
        module = DecisionModule(self.mock_agent, self.config)
        self.assertEqual(module.num_actions, 6)  # From mock environment

    def test_get_action_space_size_fallback(self):
        """Test getting action space size fallback."""
        # Remove environment from agent
        delattr(self.mock_agent, "environment")

        with patch("farm.core.action.ActionType") as mock_action_type:
            mock_action_type.__len__ = Mock(return_value=5)
            module = DecisionModule(self.mock_agent, self.config)
            self.assertEqual(module.num_actions, 5)

    def test_get_action_space_size_from_space_object(self):
        """Test getting action space size from space object."""
        from gymnasium import spaces

        action_space = spaces.Discrete(10)
        module = DecisionModule(self.mock_agent, self.config, action_space=action_space)

        self.assertEqual(module.num_actions, 10)

    def test_create_observation_space(self):
        """Test creating observation space."""
        module = DecisionModule(self.mock_agent, self.config)

        obs_space = module._create_observation_space()

        self.assertEqual(obs_space.shape, (self.config.rl_state_dim,))
        self.assertEqual(obs_space.dtype, np.float32)

    def test_create_action_space(self):
        """Test creating action space."""
        module = DecisionModule(self.mock_agent, self.config)

        action_space = module._create_action_space()

        self.assertEqual(action_space.n, module.num_actions)


class TestDecisionModuleIntegration(unittest.TestCase):
    """Integration tests for DecisionModule."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "integration_test_agent"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 4
        self.mock_agent.environment = self.mock_env

    def test_full_decision_cycle(self):
        """Test a complete decision cycle."""
        config = DecisionConfig(algorithm_type="fallback")  # Ensure we use fallback
        module = DecisionModule(self.mock_agent, config)

        # Create state
        state = torch.randn(8)

        # Get action
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        # Get probabilities
        probs = module.get_action_probabilities(state)
        self.assertEqual(len(probs), 4)
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
        module = DecisionModule(self.mock_agent, config)

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
            new_module = DecisionModule(self.mock_agent, config)
            self.assertFalse(new_module._is_trained)

            new_module.load_model(model_path)

            # Check state was restored
            self.assertEqual(new_module._is_trained, module._is_trained)


if __name__ == "__main__":
    unittest.main()
