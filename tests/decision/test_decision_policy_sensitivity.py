"""Regression tests ensuring policy weights affect action selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from gymnasium import spaces

from farm.core.action import get_action_count
from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule
from farm.core.policy_inheritance import apply_lamarckian_policy_warmstart

try:
    import tianshou  # noqa: F401

    TIANSHOU_AVAILABLE = True
except ImportError:
    TIANSHOU_AVAILABLE = False


def _make_decision_module(*, seed: int = 0) -> DecisionModule:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mock_agent = Mock()
    mock_agent.agent_id = f"agent_{seed}"
    mock_env = Mock()
    mock_env.action_space = spaces.Discrete(get_action_count())
    mock_agent.environment = mock_env
    observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    return DecisionModule(
        mock_agent,
        mock_env.action_space,
        observation_space,
        DecisionConfig(),
    )


def _action_histogram(module: DecisionModule, state: torch.Tensor, *, trials: int = 2000) -> np.ndarray:
    counts = np.zeros(module.num_actions, dtype=np.int64)
    for _ in range(trials):
        action = module.decide_action(state)
        counts[action] += 1
    return counts


@pytest.mark.skipif(not TIANSHOU_AVAILABLE, reason="Tianshou not available")
class TestDecisionPolicySensitivity(unittest.TestCase):
    """Policy state must change sampled actions when chromosome weights are fixed."""

    def test_decide_action_responds_to_policy_weights(self):
        state = torch.randn(8)
        uniform_weights = np.ones(get_action_count(), dtype=np.float64) / get_action_count()

        module_a = _make_decision_module(seed=11)
        module_b = _make_decision_module(seed=22)

        np.random.seed(123)
        hist_a = _action_histogram(module_a, state)
        np.random.seed(123)
        hist_b = _action_histogram(module_b, state, trials=2000)

        self.assertFalse(np.array_equal(hist_a, hist_b))

        np.random.seed(456)
        weighted_a = np.zeros(get_action_count(), dtype=np.int64)
        for _ in range(2000):
            weighted_a[module_a.decide_action(state, action_weights=uniform_weights)] += 1
        np.random.seed(456)
        weighted_b = np.zeros(get_action_count(), dtype=np.int64)
        for _ in range(2000):
            weighted_b[module_b.decide_action(state, action_weights=uniform_weights)] += 1
        self.assertFalse(np.array_equal(weighted_a, weighted_b))

    def test_lamarckian_warmstart_matches_parent_distribution(self):
        parent_module = _make_decision_module(seed=101)
        cold_module = _make_decision_module(seed=202)
        warm_module = _make_decision_module(seed=303)

        parent = SimpleNamespace(
            agent_id="parent",
            behavior=SimpleNamespace(decision_module=parent_module),
        )
        cold_child = SimpleNamespace(
            agent_id="cold",
            behavior=SimpleNamespace(decision_module=cold_module),
        )
        warm_child = SimpleNamespace(
            agent_id="warm",
            behavior=SimpleNamespace(decision_module=warm_module),
        )

        state = torch.randn(8)
        uniform_weights = np.ones(get_action_count(), dtype=np.float64) / get_action_count()

        for _ in range(32):
            parent_module.update(state, 0, 1.0, state, False, train_now=False)
        parent_module.train_if_ready()

        self.assertIsNone(apply_lamarckian_policy_warmstart(parent, warm_child))

        np.random.seed(999)
        parent_hist = _action_histogram(parent_module, state, trials=1500)
        np.random.seed(999)
        warm_hist = _action_histogram(warm_module, state, trials=1500)
        np.random.seed(999)
        cold_hist = _action_histogram(cold_module, state, trials=1500)

        self.assertTrue(np.array_equal(parent_hist, warm_hist))
        self.assertFalse(np.array_equal(parent_hist, cold_hist))

        np.random.seed(1001)
        parent_weighted = np.zeros(get_action_count(), dtype=np.int64)
        for _ in range(1500):
            parent_weighted[
                parent_module.decide_action(state, action_weights=uniform_weights)
            ] += 1
        np.random.seed(1001)
        warm_weighted = np.zeros(get_action_count(), dtype=np.int64)
        for _ in range(1500):
            warm_weighted[
                warm_module.decide_action(state, action_weights=uniform_weights)
            ] += 1
        np.random.seed(1001)
        cold_weighted = np.zeros(get_action_count(), dtype=np.int64)
        for _ in range(1500):
            cold_weighted[
                cold_module.decide_action(state, action_weights=uniform_weights)
            ] += 1

        self.assertTrue(np.array_equal(parent_weighted, warm_weighted))
        self.assertFalse(np.array_equal(parent_weighted, cold_weighted))


@pytest.mark.skipif(not TIANSHOU_AVAILABLE, reason="Tianshou not available")
class TestTianshouPredictProbaShapes(unittest.TestCase):
    """predict_proba must work for common observation shapes on each wrapper."""

    def _assert_predict_proba(self, wrapper, state: np.ndarray) -> None:
        probs = wrapper.predict_proba(state)
        self.assertEqual(probs.shape, (wrapper.num_actions,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))

    def test_predict_proba_state_shapes(self):
        from farm.core.decision.algorithms.tianshou import (
            A2CWrapper,
            DQNWrapper,
            PPOWrapper,
            SACWrapper,
        )

        flat_dim = 8
        flat_1d = np.random.randn(flat_dim).astype(np.float32)
        flat_2d = flat_1d.reshape(1, -1)

        dqn = DQNWrapper(
            num_actions=4,
            state_dim=flat_dim,
            observation_shape=(flat_dim,),
        )
        for state in (flat_1d, flat_2d):
            self._assert_predict_proba(dqn, state)

        spatial_shape = (13, 13, 13)
        spatial_state = np.random.randn(*spatial_shape).astype(np.float32)
        spatial_wrappers = [
            PPOWrapper(num_actions=4, state_dim=int(np.prod(spatial_shape)), observation_shape=spatial_shape),
            A2CWrapper(num_actions=4, state_dim=int(np.prod(spatial_shape)), observation_shape=spatial_shape),
            SACWrapper(num_actions=4, state_dim=int(np.prod(spatial_shape)), observation_shape=spatial_shape),
        ]
        for wrapper in spatial_wrappers:
            self._assert_predict_proba(wrapper, spatial_state)

    def test_select_action_with_mask_respects_mask(self):
        from farm.core.decision.algorithms.tianshou import DQNWrapper

        wrapper = DQNWrapper(
            num_actions=4,
            state_dim=8,
            observation_shape=(8,),
        )
        state = np.random.randn(8).astype(np.float32)
        mask = np.array([True, True, False, False])

        for _ in range(50):
            action = wrapper.select_action_with_mask(state, mask)
            self.assertIn(action, [0, 1])
