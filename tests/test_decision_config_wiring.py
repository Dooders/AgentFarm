"""Wiring tests for the previously-latent Tier 1 chromosome genes.

Each test exercises one of the four follow-up wiring changes and asserts the
gene actually controls a runtime artifact rather than just being declared on
``DecisionConfig``.
"""

from __future__ import annotations

import unittest

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
        self.fail(f"Could not locate target_update_freq on {type(policy).__name__}")

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


if __name__ == "__main__":
    unittest.main()
