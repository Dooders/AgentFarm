import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from farm.core.decision.algorithms.base import ActionAlgorithm
from farm.core.decision.algorithms.rl_base import (
    ExperienceReplayBuffer,
    PrioritizedReplayBuffer,
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

        # Ensure we're at a training step (step_count % 4 == 0)
        while self.algorithm.step_count % 4 != 0:
            state = np.random.randn(4)
            self.algorithm.store_experience(state, 0, 0.0, state + 0.1, False)

        # Force a training step
        self.algorithm.train(np.array([[0.0, 0.0, 0.0, 0.0]]), np.array([0]))

        # Training should have occurred
        self.assertGreater(len(self.algorithm.training_calls), initial_training_calls)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests for PrioritizedReplayBuffer (PER)."""

    def _make_buffer(self, **kwargs):
        """Helper: create a small PER buffer."""
        defaults = dict(max_size=20, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=10, epsilon=1e-6)
        defaults.update(kwargs)
        return PrioritizedReplayBuffer(**defaults)

    def _fill(self, buf, n=10):
        """Helper: add *n* dummy experiences to *buf*."""
        for i in range(n):
            state = np.array([float(i), float(i + 1)])
            buf.append(state, i % 3, float(i) * 0.1, state + 0.1, i == n - 1)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        buf = PrioritizedReplayBuffer(max_size=100)
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.max_size, 100)
        self.assertEqual(buf.position, 0)
        self.assertEqual(buf.alpha, 0.6)
        self.assertEqual(buf.beta_start, 0.4)
        self.assertAlmostEqual(buf.beta, 0.4)
        self.assertEqual(buf.replay_strategy, "prioritized")

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            PrioritizedReplayBuffer(alpha=1.5)

    def test_invalid_beta_start_raises(self):
        with self.assertRaises(ValueError):
            PrioritizedReplayBuffer(beta_start=-0.1)

    def test_invalid_epsilon_raises(self):
        with self.assertRaises(ValueError):
            PrioritizedReplayBuffer(epsilon=0.0)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            PrioritizedReplayBuffer(replay_strategy="unknown")  # type: ignore

    # ------------------------------------------------------------------
    # Append & capacity
    # ------------------------------------------------------------------

    def test_append_increments_length(self):
        buf = self._make_buffer()
        self._fill(buf, 5)
        self.assertEqual(len(buf), 5)

    def test_append_wraps_at_capacity(self):
        buf = self._make_buffer(max_size=5)
        self._fill(buf, 7)
        self.assertEqual(len(buf), 5)
        self.assertEqual(buf.position, 2)

    def test_new_transition_gets_max_priority(self):
        buf = self._make_buffer()
        self._fill(buf, 3)
        # Update priorities so they differ
        buf.update_priorities(np.array([0, 1, 2]), np.array([0.5, 2.0, 0.1]))
        max_priority = float(buf.priorities[:3].max())
        # Append a new transition
        state = np.zeros(2)
        buf.append(state, 0, 0.0, state, False)
        self.assertAlmostEqual(buf.priorities[3], max_priority)

    # ------------------------------------------------------------------
    # Sampling — prioritized
    # ------------------------------------------------------------------

    def test_sample_returns_required_keys(self):
        buf = self._make_buffer()
        self._fill(buf, 10)
        batch = buf.sample(4)
        for key in ("state", "action", "reward", "next_state", "done", "indices", "is_weights"):
            self.assertIn(key, batch)

    def test_sample_correct_batch_size(self):
        buf = self._make_buffer()
        self._fill(buf, 10)
        batch = buf.sample(4)
        self.assertEqual(len(batch["indices"]), 4)
        self.assertEqual(len(batch["is_weights"]), 4)
        self.assertEqual(batch["state"].shape[0], 4)

    def test_is_weights_range(self):
        """IS weights must be in (0, 1]."""
        buf = self._make_buffer()
        self._fill(buf, 10)
        batch = buf.sample(5)
        weights = batch["is_weights"]
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1.0 + 1e-6))

    def test_sampling_biased_toward_high_priority(self):
        """Transitions with higher priority should be sampled more often."""
        np.random.seed(42)
        buf = self._make_buffer(max_size=100, alpha=1.0, beta_start=0.0, beta_end=0.0, beta_steps=1)
        self._fill(buf, 10)
        # Give index 0 a very high priority
        buf.update_priorities(np.arange(10), np.full(10, 0.01))
        buf.update_priorities(np.array([0]), np.array([100.0]))

        counts = np.zeros(10, dtype=int)
        for _ in range(5000):
            batch = buf.sample(1)
            counts[batch["indices"][0]] += 1

        # Index 0 should be sampled far more than any other
        self.assertGreater(counts[0], counts[1:].max() * 5)

    def test_sampling_biased_toward_high_priority_nonzero_beta(self):
        """Sampling bias toward high-priority transitions should hold with non-zero beta."""
        np.random.seed(0)
        buf = self._make_buffer(max_size=100, alpha=1.0, beta_start=0.5, beta_end=1.0, beta_steps=100)
        self._fill(buf, 10)
        buf.update_priorities(np.arange(10), np.full(10, 0.01))
        buf.update_priorities(np.array([0]), np.array([100.0]))

        counts = np.zeros(10, dtype=int)
        for _ in range(5000):
            batch = buf.sample(1)
            counts[batch["indices"][0]] += 1

        # Even with non-zero beta, index 0 should be sampled far more often
        self.assertGreater(counts[0], counts[1:].max() * 5)

    def test_insufficient_data_raises(self):
        buf = self._make_buffer()
        self._fill(buf, 2)
        with self.assertRaises(ValueError):
            buf.sample(5)

    # ------------------------------------------------------------------
    # Sampling — uniform fallback
    # ------------------------------------------------------------------

    def test_uniform_strategy_no_indices_bias(self):
        """Uniform strategy: IS weights should all be 1.0."""
        buf = self._make_buffer(replay_strategy="uniform")
        self._fill(buf, 10)
        batch = buf.sample(5)
        np.testing.assert_array_equal(batch["is_weights"], np.ones(5, dtype=np.float32))

    def test_uniform_strategy_still_has_indices_key(self):
        buf = self._make_buffer(replay_strategy="uniform")
        self._fill(buf, 10)
        batch = buf.sample(5)
        self.assertIn("indices", batch)
        self.assertEqual(len(batch["indices"]), 5)

    # ------------------------------------------------------------------
    # update_priorities
    # ------------------------------------------------------------------

    def test_update_priorities_correctness(self):
        buf = self._make_buffer()
        self._fill(buf, 5)
        indices = np.array([0, 2, 4])
        td_errors = np.array([1.0, 2.0, 0.5])
        buf.update_priorities(indices, td_errors)
        expected = td_errors + buf.epsilon
        for idx, exp in zip(indices, expected):
            self.assertAlmostEqual(buf.priorities[idx], exp, places=10)

    def test_update_priorities_uses_abs(self):
        """Negative TD errors should be treated as their absolute value."""
        buf = self._make_buffer()
        self._fill(buf, 3)
        buf.update_priorities(np.array([0]), np.array([-3.0]))
        self.assertAlmostEqual(buf.priorities[0], 3.0 + buf.epsilon, places=10)

    def test_update_priorities_scalar_broadcast(self):
        """Scalar TD error should broadcast across all provided indices."""
        buf = self._make_buffer()
        self._fill(buf, 4)
        indices = np.array([0, 2, 3])
        buf.update_priorities(indices, np.array(1.25))
        for idx in indices:
            self.assertAlmostEqual(buf.priorities[idx], 1.25 + buf.epsilon, places=10)

    def test_update_priorities_mismatched_shape_raises(self):
        """Non-broadcastable TD error shapes should raise a clear error."""
        buf = self._make_buffer()
        self._fill(buf, 4)
        with self.assertRaises(ValueError):
            buf.update_priorities(np.array([0, 1, 2]), np.array([0.1, 0.2]))

    # ------------------------------------------------------------------
    # Beta annealing
    # ------------------------------------------------------------------

    def test_beta_anneals_toward_beta_end(self):
        buf = self._make_buffer(beta_start=0.4, beta_end=1.0, beta_steps=10)
        for _ in range(10):
            buf.update_beta()
        self.assertAlmostEqual(buf.beta, 1.0)

    def test_beta_does_not_exceed_beta_end(self):
        buf = self._make_buffer(beta_start=0.4, beta_end=1.0, beta_steps=5)
        for _ in range(20):  # more steps than beta_steps
            buf.update_beta()
        self.assertLessEqual(buf.beta, 1.0)

    def test_update_beta_returns_current_beta(self):
        buf = self._make_buffer(beta_start=0.4, beta_end=1.0, beta_steps=10)
        returned = buf.update_beta()
        self.assertAlmostEqual(returned, buf.beta)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def test_diagnostics_empty_buffer(self):
        buf = self._make_buffer()
        diag = buf.diagnostics()
        self.assertEqual(diag["buffer_size"], 0)
        self.assertEqual(diag["priority_max"], 0.0)

    def test_diagnostics_keys(self):
        buf = self._make_buffer()
        self._fill(buf, 5)
        diag = buf.diagnostics()
        for key in ("priority_min", "priority_max", "priority_mean", "beta", "buffer_size"):
            self.assertIn(key, diag)
        self.assertEqual(diag["buffer_size"], 5)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def test_clear_resets_buffer(self):
        buf = self._make_buffer()
        self._fill(buf, 5)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.position, 0)
        self.assertAlmostEqual(buf.beta, buf.beta_start)
        self.assertTrue(np.all(buf.priorities == 0.0))


class TestDecisionConfigPER(unittest.TestCase):
    """Tests for PER-related fields in DecisionConfig."""

    def test_default_replay_strategy(self):
        from farm.core.decision.config import DecisionConfig
        cfg = DecisionConfig()
        self.assertEqual(cfg.replay_strategy, "uniform")

    def test_prioritized_strategy(self):
        from farm.core.decision.config import DecisionConfig
        cfg = DecisionConfig(replay_strategy="prioritized")
        self.assertEqual(cfg.replay_strategy, "prioritized")

    def test_invalid_strategy_raises(self):
        from pydantic import ValidationError
        from farm.core.decision.config import DecisionConfig
        with self.assertRaises(ValidationError):
            DecisionConfig(replay_strategy="bad")

    def test_per_alpha_defaults(self):
        from farm.core.decision.config import DecisionConfig
        cfg = DecisionConfig()
        self.assertAlmostEqual(cfg.per_alpha, 0.6)

    def test_per_alpha_out_of_range_raises(self):
        from pydantic import ValidationError
        from farm.core.decision.config import DecisionConfig
        with self.assertRaises(ValidationError):
            DecisionConfig(per_alpha=1.5)

    def test_per_epsilon_must_be_positive(self):
        from pydantic import ValidationError
        from farm.core.decision.config import DecisionConfig
        with self.assertRaises(ValidationError):
            DecisionConfig(per_epsilon=0.0)

    def test_per_beta_steps_must_be_positive(self):
        from pydantic import ValidationError
        from farm.core.decision.config import DecisionConfig
        with self.assertRaises(ValidationError):
            DecisionConfig(per_beta_steps=0)


if __name__ == "__main__":
    unittest.main()
