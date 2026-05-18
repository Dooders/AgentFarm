"""Wiring tests for the previously-latent Tier 1 chromosome genes.

Each test exercises one of the four follow-up wiring changes and asserts the
gene actually controls a runtime artifact rather than just being declared on
``DecisionConfig``.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium

from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule, TIANSHOU_AVAILABLE


class _FakeAgent:
    """Minimal stand-in accepted by :class:`DecisionModule`."""

    def __init__(self, agent_id: str = "test_agent") -> None:
        self.agent_id = agent_id


def _make_module(config: DecisionConfig) -> DecisionModule:
    return DecisionModule(
        agent=_FakeAgent(),
        action_space=gymnasium.spaces.Discrete(4),
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(8,)),
        config=config,
    )


@unittest.skipUnless(TIANSHOU_AVAILABLE, "Tianshou required for DQN target-update wiring tests")
class TestTargetUpdateFreqWiring(unittest.TestCase):
    """`DecisionConfig.target_update_freq` must reach the live DQN policy."""

    def _read_freq(self, module: DecisionModule) -> int:
        # Tianshou stores target_update_freq on _freq (versioned attr name).
        policy = module.algorithm.policy
        for attr in ("_freq", "target_update_freq"):
            if hasattr(policy, attr):
                return int(getattr(policy, attr))
        raise AssertionError(
            f"Could not locate target_update_freq on {type(policy).__name__}"
        )

    def test_default_target_update_freq_propagates(self):
        cfg = DecisionConfig(algorithm_type="dqn")
        module = _make_module(cfg)
        self.assertEqual(self._read_freq(module), int(cfg.target_update_freq))

    def test_overridden_target_update_freq_propagates(self):
        cfg = DecisionConfig(algorithm_type="dqn", target_update_freq=250)
        module = _make_module(cfg)
        self.assertEqual(self._read_freq(module), 250)

    def test_distinct_values_yield_distinct_policies(self):
        m1 = _make_module(DecisionConfig(algorithm_type="dqn", target_update_freq=50))
        m2 = _make_module(DecisionConfig(algorithm_type="dqn", target_update_freq=750))
        self.assertNotEqual(self._read_freq(m1), self._read_freq(m2))


class TestEnsembleSizeWiring(unittest.TestCase):
    """`DecisionConfig.ensemble_size` must control RandomForest n_estimators."""

    def test_random_forest_uses_ensemble_size(self):
        cfg = DecisionConfig(algorithm_type="random_forest", ensemble_size=7, seed=0)
        module = _make_module(cfg)
        self.assertEqual(type(module.algorithm).__name__, "RandomForestActionSelector")
        self.assertEqual(module.algorithm.model.n_estimators, 7)

    def test_random_forest_ensemble_size_is_evolvable(self):
        m1 = _make_module(DecisionConfig(algorithm_type="random_forest", ensemble_size=5, seed=0))
        m2 = _make_module(DecisionConfig(algorithm_type="random_forest", ensemble_size=12, seed=0))
        self.assertEqual(m1.algorithm.model.n_estimators, 5)
        self.assertEqual(m2.algorithm.model.n_estimators, 12)

    def test_explicit_algorithm_params_n_estimators_wins_over_ensemble_size(self):
        cfg = DecisionConfig(
            algorithm_type="random_forest",
            ensemble_size=10,
            algorithm_params={"n_estimators": 3},
            seed=0,
        )
        module = _make_module(cfg)
        self.assertEqual(module.algorithm.model.n_estimators, 3)

    def test_naive_bayes_does_not_get_n_estimators(self):
        # naive_bayes is reachable through the new traditional-ML route but
        # is not in the ensemble whitelist, so n_estimators must NOT be
        # injected into its constructor.
        cfg = DecisionConfig(algorithm_type="naive_bayes", ensemble_size=99)
        module = _make_module(cfg)
        self.assertEqual(type(module.algorithm).__name__, "NaiveBayesActionSelector")
        self.assertFalse(hasattr(module.algorithm.model, "n_estimators"))


@unittest.skipUnless(TIANSHOU_AVAILABLE, "Tianshou required for DQN epsilon-greedy wiring tests")
class TestEpsilonGreedyWiring(unittest.TestCase):
    """`DecisionConfig.epsilon_*` must drive the live Tianshou DQN policy."""

    def test_initial_eps_matches_epsilon_start(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.7,
            epsilon_min=0.05,
            epsilon_decay=0.99,
        )
        module = _make_module(cfg)
        # Before any action selection, ``policy.eps`` should reflect
        # ``epsilon_start``; previously it stayed at the Tianshou default 0.0.
        self.assertAlmostEqual(float(module.algorithm.policy.eps), 0.7, places=5)

    def test_eps_decays_per_action_call(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.7,
            epsilon_min=0.05,
            epsilon_decay=0.5,
        )
        module = _make_module(cfg)
        algo = module.algorithm

        import numpy as np

        state = np.zeros((1, 8), dtype=np.float32)
        mask = np.ones(4, dtype=bool)

        algo.select_action_with_mask(state, mask)
        first = float(algo.policy.eps)
        algo.select_action_with_mask(state, mask)
        second = float(algo.policy.eps)

        self.assertAlmostEqual(first, 0.7 * 0.5, places=5)
        self.assertAlmostEqual(second, 0.7 * 0.5 * 0.5, places=5)

    def test_first_action_uses_epsilon_start_before_decay(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.7,
            epsilon_min=0.05,
            epsilon_decay=0.5,
        )
        module = _make_module(cfg)
        algo = module.algorithm

        import numpy as np
        import torch

        class _SpyPolicy:
            def __init__(self):
                self.eps = 0.0
                self.eps_seen: list[float] = []

            def set_eps(self, value: float) -> None:
                self.eps = float(value)

            def __call__(self, *_args, **_kwargs):
                self.eps_seen.append(float(self.eps))
                return torch.tensor([0]), None

        spy = _SpyPolicy()
        algo.policy = spy
        algo._apply_eps_to_policy(initial=True)

        state = np.zeros((1, 8), dtype=np.float32)
        mask = np.ones(4, dtype=bool)

        algo.select_action_with_mask(state, mask)

        self.assertEqual(len(spy.eps_seen), 1)
        self.assertAlmostEqual(spy.eps_seen[0], 0.7, places=5)
        self.assertAlmostEqual(float(spy.eps), 0.35, places=5)

    def test_eps_floor_at_epsilon_min(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.2,
            epsilon_min=0.1,
            epsilon_decay=0.5,
        )
        module = _make_module(cfg)
        algo = module.algorithm

        import numpy as np

        state = np.zeros((1, 8), dtype=np.float32)
        mask = np.ones(4, dtype=bool)
        for _ in range(20):
            algo.select_action_with_mask(state, mask)

        self.assertAlmostEqual(float(algo.policy.eps), 0.1, places=5)

    def test_eval_mode_uses_eps_test(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.9,
            epsilon_min=0.02,
        )
        module = _make_module(cfg)
        module.algorithm.set_train_mode(False)
        self.assertAlmostEqual(float(module.algorithm.policy.eps), 0.02, places=5)

    def test_eps_decays_on_weighted_decision_fallback_path(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.5,
        )
        module = _make_module(cfg)

        import numpy as np

        state = np.zeros(8, dtype=np.float32)
        action_weights = np.ones(4, dtype=np.float64) / 4.0

        # Simulate a probability-prediction failure so DecisionModule falls
        # back to weighted random action selection. Epsilon should still decay
        # once for this real decision.
        with patch.object(module.algorithm, "predict_proba", side_effect=RuntimeError("boom")):
            module.decide_action(state, action_weights=action_weights)

        self.assertAlmostEqual(float(module.algorithm.policy.eps), 0.5, places=5)

    def test_weighted_decision_predict_proba_sees_initial_eps(self):
        cfg = DecisionConfig(
            algorithm_type="dqn",
            epsilon_start=0.8,
            epsilon_min=0.1,
            epsilon_decay=0.5,
        )
        module = _make_module(cfg)

        import numpy as np

        observed_eps: list[float] = []

        def _predict_proba(_state):
            observed_eps.append(float(module.algorithm.policy.eps))
            return np.ones(4, dtype=np.float64) / 4.0

        state = np.zeros(8, dtype=np.float32)
        action_weights = np.ones(4, dtype=np.float64) / 4.0
        with patch.object(module.algorithm, "predict_proba", side_effect=_predict_proba):
            module.decide_action(state, action_weights=action_weights)

        self.assertEqual(len(observed_eps), 1)
        self.assertAlmostEqual(observed_eps[0], 0.8, places=5)
        self.assertAlmostEqual(float(module.algorithm.policy.eps), 0.4, places=5)


@unittest.skipUnless(TIANSHOU_AVAILABLE, "Tianshou required for DQN hidden-size wiring tests")
class TestDqnHiddenSizeWiring(unittest.TestCase):
    """`DecisionConfig.dqn_hidden_size` must control the Q-network width."""

    def _hidden_widths(self, module: DecisionModule):
        import torch.nn as nn

        layers = list(module.algorithm.policy.model.q_layers)
        widths = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                widths.append((layer.in_features, layer.out_features))
        return widths

    def test_hidden_size_changes_layer_widths(self):
        m_small = _make_module(DecisionConfig(algorithm_type="dqn", dqn_hidden_size=16))
        m_large = _make_module(DecisionConfig(algorithm_type="dqn", dqn_hidden_size=64))
        widths_small = self._hidden_widths(m_small)
        widths_large = self._hidden_widths(m_large)
        self.assertNotEqual(widths_small, widths_large)
        # The narrowest hidden width should equal ``dqn_hidden_size`` (last
        # FC layer's input dimension before the action head).
        last_in_small = widths_small[-1][0]
        last_in_large = widths_large[-1][0]
        self.assertEqual(last_in_small, 16)
        self.assertEqual(last_in_large, 64)


class TestReplayBufferWiring(unittest.TestCase):
    """YAML ``learning.memory_size`` / ``batch_size`` must reach the Tianshou
    DQN replay buffer (previously they were silently dropped)."""

    def test_yaml_memory_size_drives_rl_buffer_size(self):
        from farm.config import SimulationConfig
        from farm.core.agent.config.component_configs import AgentComponentConfig

        cfg = SimulationConfig.from_centralized_config(environment="development")
        cfg.learning.memory_size = 1234
        cfg.learning.batch_size = 17
        decision_cfg = AgentComponentConfig.from_simulation_config(cfg).decision
        self.assertEqual(decision_cfg.rl_buffer_size, 1234)
        self.assertEqual(decision_cfg.rl_batch_size, 17)
        # Legacy fields still receive the value too, for the legacy DQN module.
        self.assertEqual(decision_cfg.memory_size, 1234)
        self.assertEqual(decision_cfg.batch_size, 17)


if __name__ == "__main__":
    unittest.main()
