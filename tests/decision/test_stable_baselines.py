import unittest
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from farm.core.decision.algorithms.base import ActionAlgorithm
from farm.core.decision.algorithms.stable_baselines import (
    A2CWrapper,
    PPOWrapper,
    SACWrapper,
    StableBaselinesWrapper,
    TD3Wrapper,
)


class TestStableBaselinesWrapper(unittest.TestCase):
    """Test StableBaselinesWrapper base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 3
        self.state_dim = 4

        # Mock stable baselines components
        self.mock_algorithm_class = Mock()
        self.mock_algorithm = Mock()
        self.mock_algorithm_class.return_value = self.mock_algorithm

        # Mock gymnasium
        self.mock_env = Mock()
        with patch("gymnasium.make", return_value=self.mock_env):
            self.wrapper = StableBaselinesWrapper(
                num_actions=self.num_actions,
                algorithm_class=cast(type[BaseAlgorithm], self.mock_algorithm_class),
                state_dim=self.state_dim,
                algorithm_kwargs={"learning_rate": 0.001},
            )

    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.num_actions, self.num_actions)
        self.assertEqual(self.wrapper.state_dim, self.state_dim)
        self.assertEqual(self.wrapper.algorithm_class, self.mock_algorithm_class)
        self.assertIsNotNone(self.wrapper.algorithm)
        self.assertIsNotNone(self.wrapper.replay_buffer)

    def test_select_action_numpy_state(self):
        """Test action selection with numpy state."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        self.mock_algorithm.predict.return_value = (np.array([1]), None)

        action = self.wrapper.select_action(state)

        self.assertIsInstance(action, int)
        self.assertEqual(action, 1)
        self.mock_algorithm.predict.assert_called_once()

    def test_select_action_tensor_state(self):
        """Test action selection with PyTorch tensor state."""
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.mock_algorithm.predict.return_value = (np.array([2]), None)

        action = self.wrapper.select_action(state.numpy())

        self.assertIsInstance(action, int)
        self.assertEqual(action, 2)

    def test_select_action_continuous_to_discrete(self):
        """Test conversion of continuous actions to discrete."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        # Simulate continuous action output
        self.mock_algorithm.predict.return_value = (np.array([2.7]), None)

        action = self.wrapper.select_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    def test_predict_proba(self):
        """Test probability prediction."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        self.mock_algorithm.predict.return_value = (np.array([1]), None)

        probs = self.wrapper.predict_proba(state)

        self.assertEqual(len(probs), self.num_actions)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        # Action 1 should have highest probability (0.8)
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[1], probs[2])

    def test_store_experience(self):
        """Test experience storage."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        next_state = np.array([1.1, 2.1, 3.1, 4.1])
        action = 1
        reward = 0.5
        done = False

        self.wrapper.store_experience(state, action, reward, next_state, done)

        # Check that experience was added to replay buffer
        self.assertEqual(len(self.wrapper.replay_buffer), 1)

    def test_store_experience_tensor_conversion(self):
        """Test experience storage with tensor conversion."""
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        next_state = torch.tensor([1.1, 2.1, 3.1, 4.1])
        action = 1
        reward = 0.5
        done = False

        self.wrapper.store_experience(state, action, reward, next_state, done)

        # Check that tensors were converted to numpy
        self.assertEqual(len(self.wrapper.replay_buffer), 1)
        experience = self.wrapper.replay_buffer.buffer[0]
        self.assertIsInstance(experience["state"], np.ndarray)
        self.assertIsInstance(experience["next_state"], np.ndarray)

    def test_train_on_batch(self):
        """Test batch training."""
        # Add some experiences
        for i in range(5):
            state = np.random.randn(4)
            self.wrapper.store_experience(state, i % 3, float(i), state + 0.1, False)

        batch = self.wrapper.replay_buffer.sample(3)

        metrics = self.wrapper.train_on_batch(batch)

        # Should return some metrics (even if mocked)
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)

    def test_should_train(self):
        """Test training condition check."""
        # Initially should not train (empty buffer)
        self.assertFalse(self.wrapper.should_train())

        # Add enough experiences
        for i in range(self.wrapper.batch_size + 1):
            state = np.random.randn(4)
            self.wrapper.store_experience(state, i % 3, float(i), state + 0.1, False)

        # Should still not train (step count condition)
        self.assertFalse(self.wrapper.should_train())

        # Advance step count to training point
        self.wrapper._step_count = self.wrapper.train_freq

        # Should train now
        self.assertTrue(self.wrapper.should_train())

    def test_get_model_state(self):
        """Test getting model state."""
        # Add some data
        self.wrapper._step_count = 42
        for i in range(5):
            state = np.random.randn(4)
            self.wrapper.store_experience(state, i % 3, float(i), state + 0.1, False)

        state = self.wrapper.get_model_state()

        self.assertIn("algorithm_state", state)
        self.assertIn("step_count", state)
        self.assertIn("buffer_size", state)
        self.assertIn("algorithm_class", state)
        self.assertEqual(state["step_count"], 42)
        self.assertEqual(state["buffer_size"], 5)

    def test_load_model_state(self):
        """Test loading model state."""
        state = {"algorithm_state": {"mock": "data"}, "step_count": 100}

        self.wrapper.load_model_state(state)

        self.assertEqual(self.wrapper.step_count, 100)
        self.mock_algorithm.set_parameters.assert_called_once_with({"mock": "data"})

    def test_train_method(self):
        """Test the train method required by ActionAlgorithm."""
        # Add enough experiences
        for i in range(self.wrapper.batch_size + 1):
            state = np.random.randn(4)
            self.wrapper.store_experience(state, i % 3, float(i), state + 0.1, False)

        # Advance to training step
        self.wrapper._step_count = self.wrapper.train_freq

        # Should not raise an error
        self.wrapper.train(None)


class TestPPOWrapper(unittest.TestCase):
    """Test PPO wrapper."""

    @patch("gymnasium.make")
    @patch("stable_baselines3.PPO")
    def test_initialization(self, mock_ppo_class, mock_gym_make):
        """Test PPO wrapper initialization."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = PPOWrapper(
            num_actions=3,
            state_dim=4,
            algorithm_kwargs={"learning_rate": 0.001, "n_steps": 1024},
        )

        self.assertEqual(wrapper.num_actions, 3)
        self.assertEqual(wrapper.state_dim, 4)

        # Check that PPO was initialized with correct parameters
        mock_ppo_class.assert_called_once()
        call_args = mock_ppo_class.call_args
        # The wrapper creates its own dummy environment, not the mock env
        self.assertIsNotNone(call_args[1]["env"])
        self.assertEqual(call_args[1]["learning_rate"], 0.001)
        self.assertEqual(call_args[1]["n_steps"], 1024)

    @patch("gymnasium.make")
    @patch("stable_baselines3.PPO")
    def test_default_parameters(self, mock_ppo_class, mock_gym_make):
        """Test PPO wrapper with default parameters."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = PPOWrapper(num_actions=3, state_dim=4)

        # Check default parameters were set
        call_args = mock_ppo_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 3e-4)
        self.assertEqual(call_args[1]["n_steps"], 2048)
        self.assertEqual(call_args[1]["batch_size"], 64)


class TestSACWrapper(unittest.TestCase):
    """Test SAC wrapper."""

    @patch("gymnasium.make")
    @patch("stable_baselines3.SAC")
    def test_initialization(self, mock_sac_class, mock_gym_make):
        """Test SAC wrapper initialization."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = SACWrapper(
            num_actions=3,
            state_dim=4,
            algorithm_kwargs={"learning_rate": 0.001, "buffer_size": 10000},
        )

        self.assertEqual(wrapper.num_actions, 3)

        # Check SAC initialization
        mock_sac_class.assert_called_once()
        call_args = mock_sac_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 0.001)
        self.assertEqual(call_args[1]["buffer_size"], 10000)

    @patch("gymnasium.make")
    @patch("stable_baselines3.SAC")
    def test_default_parameters(self, mock_sac_class, mock_gym_make):
        """Test SAC wrapper with default parameters."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = SACWrapper(num_actions=3, state_dim=4)

        call_args = mock_sac_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 3e-4)
        self.assertEqual(call_args[1]["buffer_size"], 1000000)


class TestA2CWrapper(unittest.TestCase):
    """Test A2C wrapper."""

    @patch("gymnasium.make")
    @patch("stable_baselines3.A2C")
    def test_initialization(self, mock_a2c_class, mock_gym_make):
        """Test A2C wrapper initialization."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = A2CWrapper(
            num_actions=3,
            state_dim=4,
            algorithm_kwargs={"learning_rate": 0.001, "n_steps": 10},
        )

        # Check A2C initialization
        mock_a2c_class.assert_called_once()
        call_args = mock_a2c_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 0.001)
        self.assertEqual(call_args[1]["n_steps"], 10)

    @patch("gymnasium.make")
    @patch("stable_baselines3.A2C")
    def test_default_parameters(self, mock_a2c_class, mock_gym_make):
        """Test A2C wrapper with default parameters."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = A2CWrapper(num_actions=3, state_dim=4)

        call_args = mock_a2c_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 7e-4)
        self.assertEqual(call_args[1]["n_steps"], 5)


class TestTD3Wrapper(unittest.TestCase):
    """Test TD3 wrapper."""

    @patch("gymnasium.make")
    @patch("stable_baselines3.TD3")
    def test_initialization(self, mock_td3_class, mock_gym_make):
        """Test TD3 wrapper initialization."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = TD3Wrapper(
            num_actions=3,
            state_dim=4,
            algorithm_kwargs={"learning_rate": 0.001, "buffer_size": 50000},
        )

        # Check TD3 initialization
        mock_td3_class.assert_called_once()
        call_args = mock_td3_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 0.001)
        self.assertEqual(call_args[1]["buffer_size"], 50000)

    @patch("gymnasium.make")
    @patch("stable_baselines3.TD3")
    def test_default_parameters(self, mock_td3_class, mock_gym_make):
        """Test TD3 wrapper with default parameters."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        wrapper = TD3Wrapper(num_actions=3, state_dim=4)

        call_args = mock_td3_class.call_args
        self.assertEqual(call_args[1]["learning_rate"], 1e-3)
        self.assertEqual(call_args[1]["buffer_size"], 1000000)


class TestStableBaselinesIntegration(unittest.TestCase):
    """Test integration scenarios for stable baselines wrappers."""

    @patch("gymnasium.make")
    @patch("stable_baselines3.PPO")
    def test_full_training_cycle(self, mock_ppo_class, mock_gym_make):
        """Test a full training cycle with PPO."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env
        mock_algorithm = Mock()
        # Mock the predict method to return a tuple (action, state)
        mock_algorithm.predict.return_value = (np.array([1]), None)
        mock_ppo_class.return_value = mock_algorithm

        wrapper = PPOWrapper(num_actions=3, state_dim=4)

        # Simulate training cycle
        state = np.random.randn(4)

        # Store experiences
        for i in range(50):
            action = wrapper.select_action(state)
            next_state = state + 0.1 * np.random.randn(4)
            reward = np.random.normal(0, 1)
            done = i >= 49

            wrapper.store_experience(state, action, reward, next_state, done)
            wrapper.train(None)

            if not done:
                state = next_state

        # Check that experiences were stored
        self.assertGreater(len(wrapper.replay_buffer), 0)

        # Check that algorithm was used for action selection
        self.assertGreater(mock_algorithm.predict.call_count, 0)

    @patch("gymnasium.make")
    @patch("stable_baselines3.SAC")
    def test_probability_prediction(self, mock_sac_class, mock_gym_make):
        """Test probability prediction for SAC."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env
        mock_algorithm = Mock()
        mock_sac_class.return_value = mock_algorithm
        mock_algorithm.predict.return_value = (np.array([1]), None)

        wrapper = SACWrapper(num_actions=4, state_dim=6)

        state = np.random.randn(6)
        probs = wrapper.predict_proba(state)

        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

        # Action 1 should have highest probability
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[1], probs[2])
        self.assertGreater(probs[1], probs[3])


if __name__ == "__main__":
    unittest.main()
