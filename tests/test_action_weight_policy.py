"""Unit tests for the state-aware action re-weighter.

Verifies that each Chromosome B multiplier / threshold gene actually changes
the per-action weight vector under the appropriate state conditions.
"""

from __future__ import annotations

import unittest

import numpy as np

from farm.core.decision.action_weight_policy import (
    ActionStateSignals,
    compute_action_weights,
    extract_signals,
)
from farm.core.decision.config import DecisionConfig

ACTION_NAMES = (
    "move",
    "gather",
    "share",
    "attack",
    "reproduce",
    "defend",
    "pass",
    "communicate",
)
BASE_WEIGHTS = (0.30, 0.30, 0.15, 0.10, 0.15, 0.0, 0.0, 0.0)


def _weight(weights: np.ndarray, name: str) -> float:
    return float(weights[ACTION_NAMES.index(name)])


_NEUTRAL_KWARGS = dict(
    move_mult_no_resources=1.0,
    gather_mult_low_resources=1.0,
    share_mult_wealthy=1.0,
    share_mult_poor=1.0,
    attack_mult_desperate=1.0,
    attack_mult_stable=1.0,
    reproduce_mult_wealthy=1.0,
    reproduce_mult_poor=1.0,
)


def _neutral_cfg(**overrides) -> DecisionConfig:
    """Build a DecisionConfig where every multiplier is the identity (1.0).

    Multiplier overrides supplied by individual tests still win, so they can
    isolate the effect of a single gene under arbitrary state signals.
    """
    kwargs = dict(_NEUTRAL_KWARGS)
    kwargs.update(overrides)
    return DecisionConfig(**kwargs)


def _baseline(cfg: DecisionConfig) -> np.ndarray:
    """Reference scaled vector for *cfg* under fully-neutral state signals."""
    sig = ActionStateSignals(
        resource_ratio=0.5,
        health_ratio=0.5,
        starvation_risk=0.0,
        nearby_resources=True,
    )
    return compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg, sig)


class TestComputeActionWeights(unittest.TestCase):
    def test_normalizes_to_unit_sum(self):
        cfg = DecisionConfig()
        sig = ActionStateSignals()
        weights = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg, sig)
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)
        self.assertEqual(weights.shape, (len(ACTION_NAMES),))

    def test_baseline_with_identity_multipliers_matches_normalized_base(self):
        # When every multiplier is the identity (1.0), the vector is just the
        # base weights renormalized -- regardless of state signals.
        cfg = _neutral_cfg()
        weights = _baseline(cfg)
        expected = np.asarray(BASE_WEIGHTS) / sum(BASE_WEIGHTS)
        np.testing.assert_allclose(weights, expected, atol=1e-12)

    def test_default_decision_config_applies_reproduce_poor_at_mid_resource(self):
        # With the platform default (reproduce_mult_poor=0.3,
        # reproduce_resource_threshold=0.7), a half-resource agent has
        # reproduce dampened relative to the identity-multiplier baseline.
        # Note: ``share_mult_poor`` only fires below ratio 0.3, so it is
        # untouched at ratio=0.5 -- intentional.
        cfg_default = DecisionConfig()
        cfg_neutral = _neutral_cfg()
        sig = ActionStateSignals(
            resource_ratio=0.5,
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=True,
        )
        weights_default = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_default, sig)
        weights_neutral = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_neutral, sig)
        self.assertLess(_weight(weights_default, "reproduce"), _weight(weights_neutral, "reproduce"))

    def test_no_nearby_resources_boosts_move(self):
        cfg = _neutral_cfg(move_mult_no_resources=3.0)
        baseline = _baseline(cfg)
        sig = ActionStateSignals(
            resource_ratio=0.5,
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=False,
        )
        scaled = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg, sig)
        self.assertGreater(_weight(scaled, "move"), _weight(baseline, "move"))
        # Other actions should shrink (mass conservation under normalization)
        self.assertLess(_weight(scaled, "gather"), _weight(baseline, "gather"))

    def test_low_resources_boosts_gather(self):
        cfg = _neutral_cfg(gather_mult_low_resources=2.5)
        sig = ActionStateSignals(
            resource_ratio=0.2,
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=True,
        )
        scaled = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg, sig)
        self.assertGreater(_weight(scaled, "gather"), _weight(_baseline(cfg), "gather"))

    def test_wealthy_boosts_share_and_reproduce(self):
        cfg = _neutral_cfg(
            share_mult_wealthy=2.0,
            reproduce_mult_wealthy=2.0,
            reproduce_resource_threshold=0.7,
        )
        sig = ActionStateSignals(
            resource_ratio=0.9,
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=True,
        )
        scaled = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg, sig)
        baseline = _baseline(cfg)
        self.assertGreater(_weight(scaled, "share"), _weight(baseline, "share"))
        self.assertGreater(_weight(scaled, "reproduce"), _weight(baseline, "reproduce"))

    def test_poor_throttles_share_and_reproduce(self):
        # Compare two configs that differ only in share/reproduce poor
        # multipliers, under identical poor-state signals.  The throttled
        # config must redistribute mass away from share+reproduce.
        sig = ActionStateSignals(
            resource_ratio=0.1,  # below 0.3 (share-poor) and below 0.7 (reproduce-poor)
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=True,
        )
        cfg_neutral = _neutral_cfg(reproduce_resource_threshold=0.7)
        cfg_throttled = _neutral_cfg(
            share_mult_poor=0.1,
            reproduce_mult_poor=0.1,
            reproduce_resource_threshold=0.7,
        )
        weights_neutral = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_neutral, sig)
        weights_throttled = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_throttled, sig)
        self.assertLess(_weight(weights_throttled, "share"), _weight(weights_neutral, "share"))
        self.assertLess(_weight(weights_throttled, "reproduce"), _weight(weights_neutral, "reproduce"))

    def test_starvation_threshold_gates_attack_desperate(self):
        cfg_off = _neutral_cfg(
            attack_mult_desperate=3.0,
            attack_starvation_threshold=0.9,
        )
        cfg_on = _neutral_cfg(
            attack_mult_desperate=3.0,
            attack_starvation_threshold=0.1,
        )
        sig = ActionStateSignals(
            resource_ratio=0.5,
            health_ratio=0.5,
            starvation_risk=0.5,
            nearby_resources=True,
        )
        weights_off = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_off, sig)
        weights_on = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_on, sig)
        # threshold above starvation_risk -> multiplier does NOT fire
        # threshold below starvation_risk -> multiplier fires
        self.assertGreater(_weight(weights_on, "attack"), _weight(weights_off, "attack"))

    def test_defense_threshold_gates_attack_stable(self):
        cfg_low = _neutral_cfg(
            attack_mult_stable=0.1,
            attack_defense_threshold=0.9,  # health_ratio>=1-0.9=0.1 -> fires almost always
        )
        cfg_high = _neutral_cfg(
            attack_mult_stable=0.1,
            attack_defense_threshold=0.05,  # health_ratio>=0.95 -> fires only when very healthy
        )
        sig = ActionStateSignals(
            resource_ratio=0.5,
            health_ratio=0.5,
            starvation_risk=0.0,
            nearby_resources=True,
        )
        weights_low = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_low, sig)
        weights_high = compute_action_weights(BASE_WEIGHTS, ACTION_NAMES, cfg_high, sig)
        # cfg_low fires the down-scaling multiplier; cfg_high does not.
        self.assertLess(_weight(weights_low, "attack"), _weight(weights_high, "attack"))

    def test_enabled_actions_zero_out_disabled(self):
        cfg = _neutral_cfg()
        sig = ActionStateSignals(resource_ratio=0.5, health_ratio=0.5)
        # Only enable move (idx 0) and gather (idx 1).
        weights = compute_action_weights(
            BASE_WEIGHTS, ACTION_NAMES, cfg, sig, enabled_actions=[0, 1]
        )
        self.assertGreater(weights[0], 0.0)
        self.assertGreater(weights[1], 0.0)
        for idx in range(2, len(weights)):
            self.assertEqual(weights[idx], 0.0)
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)

    def test_zero_base_weights_falls_back_to_uniform(self):
        cfg = _neutral_cfg()
        sig = ActionStateSignals(resource_ratio=0.5)
        weights = compute_action_weights(
            tuple(0.0 for _ in BASE_WEIGHTS), ACTION_NAMES, cfg, sig
        )
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)
        np.testing.assert_allclose(weights, np.full(len(weights), 1.0 / len(weights)))

    def test_zero_base_with_enabled_uniform_over_enabled(self):
        cfg = _neutral_cfg()
        sig = ActionStateSignals()
        weights = compute_action_weights(
            tuple(0.0 for _ in BASE_WEIGHTS),
            ACTION_NAMES,
            cfg,
            sig,
            enabled_actions=[2, 5],
        )
        self.assertAlmostEqual(weights.sum(), 1.0, places=12)
        self.assertAlmostEqual(weights[2], 0.5)
        self.assertAlmostEqual(weights[5], 0.5)

    def test_length_mismatch_raises(self):
        cfg = _neutral_cfg()
        sig = ActionStateSignals(resource_ratio=0.5, nearby_resources=False)
        with self.assertRaises(ValueError):
            compute_action_weights(
                BASE_WEIGHTS, ACTION_NAMES[:-1], cfg, sig
            )


class TestExtractSignals(unittest.TestCase):
    def test_extracts_from_minimal_object(self):
        class _Stub:
            agent_id = "x"
            resource_level = 25.0
            current_health = 50.0
            starting_health = 100.0
            config = type("C", (), {"reward": type("R", (), {"max_resource_level": 100.0})()})()

            def get_component(self, _name):
                return None

        sig = extract_signals(_Stub())
        self.assertAlmostEqual(sig.resource_ratio, 0.25)
        self.assertAlmostEqual(sig.health_ratio, 0.5)
        self.assertEqual(sig.starvation_risk, 0.0)
        self.assertIsNone(sig.nearby_resources)


if __name__ == "__main__":
    unittest.main()
