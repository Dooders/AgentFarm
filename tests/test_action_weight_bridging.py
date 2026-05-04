"""Tests for the Chromosome B action-weight gene bridging.

These tests exercise the path that lets evolved
``DecisionConfig.{move,gather,share,attack,reproduce}_weight`` values flow
into ``core.actions[i].weight`` so they actually shape the policy.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from farm.core.action import (
    Action,
    attack_action,
    defend_action,
    gather_action,
    move_action,
    pass_action,
    reproduce_action,
    share_action,
)
from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.decision.config import DecisionConfig


def _build_default_actions() -> list[Action]:
    """Reproduce the platform's default action set with neutral starting weights."""
    return [
        Action("attack", 0.10, attack_action),
        Action("move", 0.40, move_action),
        Action("gather", 0.30, gather_action),
        Action("reproduce", 0.15, reproduce_action),
        Action("share", 0.20, share_action),
        Action("defend", 0.25, defend_action),
        Action("pass", 0.05, pass_action),
    ]


def _make_agent_with_decision_config(decision_config: DecisionConfig) -> AgentCore:
    """Build a partial :class:`AgentCore` exposing only the surface used here."""
    agent = object.__new__(AgentCore)
    agent.agent_id = "test"
    agent.agent_type = "system"
    config = AgentComponentConfig.default()
    config.decision = decision_config
    agent.config = config
    return agent


def _weight_for(actions, name: str) -> float:
    for action in actions:
        if action.name == name:
            return action.weight
    raise KeyError(name)


class TestCustomizeActionWeightsFromDecisionConfig(unittest.TestCase):
    """``_customize_action_weights`` should pick up evolved DecisionConfig weights."""

    def test_default_decision_config_does_not_change_legacy_behaviour(self):
        """A pristine DecisionConfig() must leave registry weights normalized only."""
        agent = _make_agent_with_decision_config(DecisionConfig())
        base = _build_default_actions()
        customized = AgentCore._customize_action_weights(agent, base, "system", environment=None)
        # Customized weights are normalized relative to one another.
        total = sum(a.weight for a in customized)
        self.assertAlmostEqual(total, 1.0, places=12)
        # Order matches input order.
        self.assertEqual([a.name for a in customized], [a.name for a in base])
        # Each action's relative ranking should mirror the registry defaults
        # since none of the genes diverge from the Pydantic defaults.
        baseline_total = sum(a.weight for a in base)
        for original, custom in zip(base, customized):
            expected = original.weight / baseline_total
            self.assertAlmostEqual(custom.weight, expected, places=12)

    def test_attack_weight_gene_overrides_registry_default(self):
        cfg = DecisionConfig(attack_weight=1.5)
        agent = _make_agent_with_decision_config(cfg)
        customized = AgentCore._customize_action_weights(
            agent, _build_default_actions(), "system", environment=None
        )
        attack_share = _weight_for(customized, "attack")
        # With attack_weight elevated to 1.5, attack should dominate the
        # normalized vector relative to the other actions.
        for name in ("move", "gather", "share", "reproduce", "defend", "pass"):
            self.assertGreater(attack_share, _weight_for(customized, name))

    def test_chromosome_value_wins_over_environment_agent_parameters(self):
        cfg = DecisionConfig(share_weight=0.95)  # near max
        agent = _make_agent_with_decision_config(cfg)
        env = Mock()
        env.config = Mock()
        env.config.agent_parameters = {"SystemAgent": {"share_weight": 0.01}}
        customized = AgentCore._customize_action_weights(
            agent, _build_default_actions(), "system", environment=env
        )
        # Compute what the gene-driven share would normalize to.
        # base after override = registry weights with share=0.95 instead of 0.20
        base_after_override = [
            ("attack", 0.10),
            ("move", 0.40),
            ("gather", 0.30),
            ("reproduce", 0.15),
            ("share", 0.95),  # gene wins
            ("defend", 0.25),
            ("pass", 0.05),
        ]
        total = sum(w for _, w in base_after_override)
        expected_share = 0.95 / total
        self.assertAlmostEqual(_weight_for(customized, "share"), expected_share, places=12)

    def test_unmodified_gene_falls_through_to_agent_parameters(self):
        # share_weight is left at its Pydantic default → DecisionConfig has no
        # effective opinion on it, so the legacy environment override wins.
        cfg = DecisionConfig()
        agent = _make_agent_with_decision_config(cfg)
        env = Mock()
        env.config = Mock()
        env.config.agent_parameters = {"SystemAgent": {"share_weight": 0.95}}
        customized = AgentCore._customize_action_weights(
            agent, _build_default_actions(), "system", environment=env
        )
        # The gene didn't override, so share weight = 0.95 from agent_parameters.
        base_after_override = [
            ("attack", 0.10),
            ("move", 0.40),
            ("gather", 0.30),
            ("reproduce", 0.15),
            ("share", 0.95),
            ("defend", 0.25),
            ("pass", 0.05),
        ]
        total = sum(w for _, w in base_after_override)
        expected_share = 0.95 / total
        self.assertAlmostEqual(_weight_for(customized, "share"), expected_share, places=12)

    def test_normalization_falls_back_to_uniform_when_all_zero(self):
        """If every gene zeroes out a weight, the helper falls back to uniform."""
        cfg = DecisionConfig(
            move_weight=0.0,
            gather_weight=0.0,
            share_weight=0.0,
            attack_weight=0.0,
            reproduce_weight=0.0,
        )
        agent = _make_agent_with_decision_config(cfg)
        # Build actions list with everything zero so legacy registry doesn't add weight.
        base = [
            Action(name, 0.0, func)
            for name, func in (
                ("attack", attack_action),
                ("move", move_action),
                ("gather", gather_action),
                ("reproduce", reproduce_action),
                ("share", share_action),
            )
        ]
        customized = AgentCore._customize_action_weights(agent, base, "system", environment=None)
        # All five get equal share.
        for action in customized:
            self.assertAlmostEqual(action.weight, 1.0 / len(customized), places=12)


class TestRefreshActionWeights(unittest.TestCase):
    """`refresh_action_weights_from_decision_config` updates live actions."""

    def test_refresh_picks_up_new_decision_config_values(self):
        agent = _make_agent_with_decision_config(DecisionConfig())
        agent.actions = _build_default_actions()
        # Initial vector reflects registry defaults only -- attack weight is small.
        initial_attack = _weight_for(agent.actions, "attack")

        # Mutate the DecisionConfig to push attack_weight near the max bound.
        agent.config.decision = DecisionConfig(attack_weight=1.9)
        AgentCore.refresh_action_weights_from_decision_config(agent)

        new_attack = _weight_for(agent.actions, "attack")
        self.assertGreater(new_attack, initial_attack)
        # Weights remain normalized after refresh.
        self.assertAlmostEqual(sum(a.weight for a in agent.actions), 1.0, places=12)

    def test_refresh_is_a_noop_when_no_genes_diverge_from_defaults(self):
        agent = _make_agent_with_decision_config(DecisionConfig())
        # Pre-existing custom action weights must not be wiped by a refresh
        # whose source decision config is unchanged.
        agent.actions = [
            Action("attack", 0.5, attack_action),
            Action("move", 0.5, move_action),
        ]
        AgentCore.refresh_action_weights_from_decision_config(agent)
        self.assertEqual(_weight_for(agent.actions, "attack"), 0.5)
        self.assertEqual(_weight_for(agent.actions, "move"), 0.5)


if __name__ == "__main__":
    unittest.main()
