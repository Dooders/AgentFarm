"""Tests for farm/core/decision/training/trainer.py – AlgorithmTrainer."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from farm.core.decision.training.trainer import AlgorithmTrainer


def _make_data(n=10, state_dim=4, num_actions=3):
    """Return a list of (state, action, reward) tuples."""
    rng = np.random.default_rng(42)
    return [
        (rng.random(state_dim), rng.integers(0, num_actions), float(rng.random()))
        for _ in range(n)
    ]


class TestAlgorithmTrainer:
    def test_train_calls_algorithm_train(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()
        data = _make_data(n=5)

        trainer.train_algorithm(algo, data)

        algo.train.assert_called_once()
        X, y_actions = algo.train.call_args[0][:2]
        assert X.shape == (5, 4)
        assert y_actions.shape == (5,)

    def test_train_passes_rewards_kwarg(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()
        data = _make_data(n=3)

        trainer.train_algorithm(algo, data)

        _, kwargs = algo.train.call_args
        assert "rewards" in kwargs
        assert kwargs["rewards"].shape == (3,)

    def test_empty_data_does_not_call_train(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()

        trainer.train_algorithm(algo, [])

        algo.train.assert_not_called()

    def test_train_stacks_states_correctly(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()

        state1 = np.array([1.0, 2.0])
        state2 = np.array([3.0, 4.0])
        data = [(state1, 0, 1.0), (state2, 1, 0.5)]

        trainer.train_algorithm(algo, data)

        X = algo.train.call_args[0][0]
        np.testing.assert_array_equal(X[0], state1)
        np.testing.assert_array_equal(X[1], state2)

    def test_train_converts_actions_to_int_array(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()
        data = [(np.zeros(2), 2, 0.3)]

        trainer.train_algorithm(algo, data)

        y_actions = algo.train.call_args[0][1]
        assert np.issubdtype(y_actions.dtype, np.integer)

    def test_train_converts_rewards_to_float_array(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()
        data = [(np.zeros(2), 0, 1)]  # integer reward

        trainer.train_algorithm(algo, data)

        rewards = algo.train.call_args[1]["rewards"]
        assert np.issubdtype(rewards.dtype, np.floating)

    def test_accepts_generator_as_training_data(self):
        trainer = AlgorithmTrainer()
        algo = MagicMock()

        def _gen():
            for i in range(4):
                yield np.array([float(i)]), i % 2, float(i * 0.1)

        trainer.train_algorithm(algo, _gen())
        algo.train.assert_called_once()
        X = algo.train.call_args[0][0]
        assert X.shape == (4, 1)
