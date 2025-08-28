import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from farm.core.decision.algorithms.base import ActionAlgorithm
from farm.core.decision.algorithms.rl_base import (
    ExperienceReplayBuffer,
    RLAlgorithm,
    SimpleReplayBuffer,
)


class TestRLAlgorithm(unittest.TestCase):
    """Test the RLAlgorithm abstract base class."""

    def test_abstract_methods(self):
        """Test that RLAlgorithm is properly abstract."""
        with self.assertRaises(TypeError):
            RLAlgorithm(num_actions=4)  # type: ignore

    def test_initialization(self):
        """Test RLAlgorithm initialization with concrete implementation."""

        class ConcreteRLAlgorithm(RLAlgorithm):
            def __init__(self, num_actions, **kwargs):
                super().__init__(num_actions=num_actions, **kwargs)
                self.replay_buffer = SimpleReplayBuffer(max_size=100)
                self.batch_size = 32

            def select_action(self, state):
                return 0

            def train(self, states, actions, rewards=None):
                pass

            def predict_proba(self, state):
                return np.full(self.num_actions, 1.0 / self.num_actions)

            def store_experience(
                self, state, action, reward, next_state, done, **kwargs
            ):
                self.replay_buffer.append(
                    state, action, reward, next_state, done, **kwargs
                )

            def train_on_batch(self, batch, **kwargs):
                return {"loss": 0.5, "value_loss": 0.25}

            def should_train(self):
                return len(self.replay_buffer) >= self.batch_size

            def get_model_state(self):
                return {
                    "step_count": self.step_count,
                    "buffer_size": len(self.replay_buffer),
                }

            def load_model_state(self, state):
                self._step_count = state.get("step_count", 0)

        algo = ConcreteRLAlgorithm(num_actions=3)
        self.assertEqual(algo.num_actions, 3)
        self.assertEqual(algo.step_count, 0)
        self.assertIsInstance(algo.replay_buffer, SimpleReplayBuffer)
        self.assertEqual(algo.batch_size, 32)

    def test_step_count_management(self):
        """Test step count increment functionality."""

        class ConcreteRLAlgorithm(RLAlgorithm):
            def select_action(self, state):
                return 0

            def train(self, states, actions, rewards=None):
                pass

            def predict_proba(self, state):
                return np.full(self.num_actions, 1.0 / self.num_actions)

            def store_experience(
                self, state, action, reward, next_state, done, **kwargs
            ):
                pass

            def train_on_batch(self, batch, **kwargs):
                return {}

            def should_train(self):
                return False

            def get_model_state(self):
                return {}

            def load_model_state(self, state):
                pass

        algo = ConcreteRLAlgorithm(num_actions=2)
        self.assertEqual(algo.step_count, 0)

        algo.update_step_count()
        self.assertEqual(algo.step_count, 1)

        algo.update_step_count()
        self.assertEqual(algo.step_count, 2)


class TestSimpleReplayBuffer(unittest.TestCase):
    """Test SimpleReplayBuffer implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = SimpleReplayBuffer(max_size=5)

    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.max_size, 5)
        self.assertEqual(self.buffer.position, 0)
        self.assertEqual(self.buffer.buffer, [])

    def test_append_experience(self):
        """Test appending experiences to buffer."""
        state = np.array([1.0, 2.0])
        action = 1
        reward = 0.5
        next_state = np.array([1.1, 2.1])
        done = False

        self.buffer.append(state, action, reward, next_state, done)

        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.position, 1)

    def test_append_with_kwargs(self):
        """Test appending experiences with additional kwargs."""
        state = np.array([1.0, 2.0])
        action = 1
        reward = 0.5
        next_state = np.array([1.1, 2.1])
        done = False

        self.buffer.append(
            state, action, reward, next_state, done, log_prob=-0.5, value=0.8
        )

        self.assertEqual(len(self.buffer), 1)
        experience = self.buffer.buffer[0]
        self.assertEqual(experience["log_prob"], -0.5)
        self.assertEqual(experience["value"], 0.8)

    def test_buffer_capacity(self):
        """Test buffer capacity management."""
        # Fill buffer to capacity
        for i in range(6):  # max_size is 5, so this should overwrite
            state = np.array([float(i), float(i + 1)])
            self.buffer.append(state, i % 3, float(i), state + 0.1, False)

        self.assertEqual(len(self.buffer), 5)
        self.assertEqual(self.buffer.position, 1)  # Should wrap around

    def test_sample_batch(self):
        """Test sampling a batch of experiences."""
        # Add some experiences
        for i in range(4):
            state = np.array([float(i), float(i + 1)])
            self.buffer.append(state, i % 3, float(i), state + 0.1, i == 3)

        batch = self.buffer.sample(batch_size=2)

        self.assertIn("state", batch)
        self.assertIn("action", batch)
        self.assertIn("reward", batch)
        self.assertIn("next_state", batch)
        self.assertIn("done", batch)

        # Check shapes - state and next_state should be numpy arrays
        state_batch = np.asarray(batch["state"])
        next_state_batch = np.asarray(batch["next_state"])
        self.assertEqual(state_batch.shape, (2, 2))  # 2 samples, 2 state dims
        self.assertEqual(next_state_batch.shape, (2, 2))
        # Other fields may be lists or arrays depending on implementation
        self.assertEqual(len(batch["action"]), 2)
        self.assertEqual(len(batch["reward"]), 2)
        self.assertEqual(len(batch["done"]), 2)

    def test_sample_insufficient_data(self):
        """Test sampling when buffer has insufficient data."""
        # Add only 2 experiences
        for i in range(2):
            state = np.array([float(i), float(i + 1)])
            self.buffer.append(state, i % 3, float(i), state + 0.1, False)

        with self.assertRaises(ValueError) as context:
            self.buffer.sample(batch_size=5)

        self.assertIn("Not enough experiences", str(context.exception))

    def test_sample_with_tensors(self):
        """Test sampling with PyTorch tensors."""
        # Add experiences with tensor states
        for i in range(3):
            state = torch.tensor([float(i), float(i + 1)])
            next_state = torch.tensor([float(i + 1), float(i + 2)])
            self.buffer.append(state, i % 3, float(i), next_state, False)

        batch = self.buffer.sample(batch_size=2)

        # States should be converted to numpy arrays
        self.assertIsInstance(batch["state"], np.ndarray)
        self.assertIsInstance(batch["next_state"], np.ndarray)

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        # Add some experiences
        for i in range(3):
            state = np.array([float(i), float(i + 1)])
            self.buffer.append(state, i % 3, float(i), state + 0.1, False)

        self.assertEqual(len(self.buffer), 3)

        self.buffer.clear()

        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.position, 0)
        self.assertEqual(self.buffer.buffer, [])

    def test_sample_data_types(self):
        """Test that sample returns correct data types."""
        # Add experience
        state = np.array([1.0, 2.0])
        next_state = np.array([1.1, 2.1])
        self.buffer.append(state, 1, 0.5, next_state, False, extra_data="test")

        batch = self.buffer.sample(batch_size=1)

        # Check data types
        self.assertIsInstance(batch["state"], np.ndarray)
        self.assertIsInstance(batch["next_state"], np.ndarray)
        self.assertIsInstance(batch["action"], np.ndarray)
        self.assertIsInstance(batch["reward"], np.ndarray)
        self.assertIsInstance(batch["done"], np.ndarray)
        self.assertEqual(batch["extra_data"], ["test"])  # List for non-standard types

    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer."""
        with self.assertRaises(ValueError):
            self.buffer.sample(batch_size=1)


class TestExperienceReplayBuffer(unittest.TestCase):
    """Test the ExperienceReplayBuffer abstract base class."""

    def test_abstract_methods(self):
        """Test that ExperienceReplayBuffer is properly abstract."""
        with self.assertRaises(TypeError):
            ExperienceReplayBuffer()  # type: ignore


class TestRLIntegration(unittest.TestCase):
    """Test integration of RL components."""

    def setUp(self):
        """Set up test fixtures."""

        class TestRLAlgorithm(RLAlgorithm):
            def __init__(self, num_actions, **kwargs):
                super().__init__(num_actions=num_actions, **kwargs)
                self.replay_buffer = SimpleReplayBuffer(max_size=100)
                self.batch_size = 32
                self.training_calls = []

            def select_action(self, state):
                return np.random.choice(self.num_actions)

            def train(self, states, actions, rewards=None):
                if self.should_train():
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.train_on_batch(batch)
                    self.training_calls.append(metrics)

            def predict_proba(self, state):
                return np.full(self.num_actions, 1.0 / self.num_actions)

            def store_experience(
                self, state, action, reward, next_state, done, **kwargs
            ):
                self.replay_buffer.append(
                    state, action, reward, next_state, done, **kwargs
                )
                self.update_step_count()

            def train_on_batch(self, batch, **kwargs):
                return {
                    "loss": np.random.uniform(0.1, 1.0),
                    "value_loss": np.random.uniform(0.05, 0.5),
                    "policy_loss": np.random.uniform(0.01, 0.3),
                }

            def should_train(self):
                return (
                    len(self.replay_buffer) >= self.batch_size
                    and self.step_count % 4 == 0  # Train every 4 steps
                )

            def get_model_state(self):
                return {
                    "step_count": self.step_count,
                    "buffer_size": len(self.replay_buffer),
                    "training_calls": len(self.training_calls),
                }

            def load_model_state(self, state):
                self._step_count = state.get("step_count", 0)

        self.algorithm = TestRLAlgorithm(num_actions=3)

    def test_experience_collection_and_training(self):
        """Test the full experience collection and training cycle."""
        state_dim = 4
        num_steps = 50

        # Simulate an episode
        current_state = np.random.randn(state_dim)

        for step in range(num_steps):
            # Select action
            action = self.algorithm.select_action(current_state)

            # Simulate environment step
            next_state = current_state + 0.1 * np.random.randn(state_dim)
            reward = np.random.normal(0, 1)
            done = step >= num_steps - 1

            # Store experience
            self.algorithm.store_experience(
                current_state, action, reward, next_state, done
            )

            # Train if needed
            self.algorithm.train(np.array([current_state]), np.array([action]))

            if not done:
                current_state = next_state

        # Check that experiences were stored
        self.assertGreater(len(self.algorithm.replay_buffer), 0)
        self.assertEqual(self.algorithm.step_count, num_steps)

        # Check that training occurred
        self.assertGreater(len(self.algorithm.training_calls), 0)

    def test_model_state_save_load(self):
        """Test saving and loading model state."""
        # Add some experiences and training
        for i in range(10):
            state = np.random.randn(4)
            self.algorithm.store_experience(state, i % 3, float(i), state + 0.1, False)

        # Save state
        state = self.algorithm.get_model_state()
        self.assertIn("step_count", state)
        self.assertIn("buffer_size", state)

        # Create new algorithm and load state
        new_algorithm = type(self.algorithm)(num_actions=3)
        new_algorithm.load_model_state(state)

        self.assertEqual(new_algorithm.step_count, self.algorithm.step_count)

    def test_training_frequency(self):
        """Test that training occurs at the correct frequency."""
        # Add enough experiences to trigger training
        for i in range(self.algorithm.batch_size + 10):
            state = np.random.randn(4)
            self.algorithm.store_experience(state, i % 3, float(i), state + 0.1, False)

        initial_training_calls = len(self.algorithm.training_calls)

        # Force a training step
        self.algorithm.train(np.array([[0.0, 0.0, 0.0, 0.0]]), np.array([0]))

        # Training should have occurred
        self.assertGreater(len(self.algorithm.training_calls), initial_training_calls)


if __name__ == "__main__":
    unittest.main()
