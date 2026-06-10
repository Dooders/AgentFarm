"""Tests for the intrinsic-goals experiment runner."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    INTRINSIC_REWARD_GENE_NAMES,
    default_hyperparameter_chromosome,
)
from farm.runners.intrinsic_goals_experiment import (
    ArmResult,
    IntrinsicGoalsExperiment,
    IntrinsicGoalsExperimentConfig,
    ReplicateResult,
    TRACKED_ACTIONS,
    assign_unique_goals,
    sample_goal_chromosome,
)


def test_sample_goal_chromosome_randomizes_only_goal_genes():
    base = default_hyperparameter_chromosome()
    rng = random.Random(0)
    sampled = sample_goal_chromosome(base, rng)

    # Non-goal genes untouched.
    assert sampled.get_value("learning_rate") == base.get_value("learning_rate")
    assert sampled.get_value("move_weight") == base.get_value("move_weight")

    # At least one goal gene changed and all stay within bounds.
    changed = False
    for name in INTRINSIC_REWARD_GENE_NAMES:
        gene = sampled.get_gene(name)
        assert gene.min_value <= gene.value <= gene.max_value
        if gene.value != base.get_value(name):
            changed = True
    assert changed


def test_sample_goal_chromosome_is_deterministic_for_seed():
    base = default_hyperparameter_chromosome()
    a = sample_goal_chromosome(base, random.Random(123))
    b = sample_goal_chromosome(base, random.Random(123))
    for name in INTRINSIC_REWARD_GENE_NAMES:
        assert a.get_value(name) == b.get_value(name)


def test_assign_unique_goals_gives_each_agent_a_distinct_goal():
    agents = [
        SimpleNamespace(
            agent_id=f"a{i}",
            hyperparameter_chromosome=default_hyperparameter_chromosome(),
        )
        for i in range(8)
    ]
    env = SimpleNamespace(alive_agent_objects=agents)

    n = assign_unique_goals(env, random.Random(7))
    assert n == 8

    # Collect each agent's share-bonus gene; with random sampling across [0,2]
    # they should not all be identical.
    share_values = {
        a.hyperparameter_chromosome.get_value("reward_share_bonus") for a in agents
    }
    assert len(share_values) > 1


class _FakeAgent:
    def __init__(self, agent_id: str, *, last_action: str | None = "move") -> None:
        self.agent_id = agent_id
        self.alive = True
        self.hyperparameter_chromosome = default_hyperparameter_chromosome()
        self.last_action_name = last_action


class _FakeEnvironment:
    def __init__(self, agents: list[_FakeAgent]) -> None:
        self._agents = list(agents)
        self.intrinsic_evolution_policy = None
        self.intrinsic_evolution_rng = None

    @property
    def alive_agent_objects(self) -> list[_FakeAgent]:
        return [a for a in self._agents if a.alive]


class TestIntrinsicGoalsExperimentConfig(unittest.TestCase):
    def test_rejects_non_positive_steps(self) -> None:
        with self.assertRaises(ValueError):
            IntrinsicGoalsExperimentConfig(num_steps=0)

    def test_rejects_non_positive_record_interval(self) -> None:
        with self.assertRaises(ValueError):
            IntrinsicGoalsExperimentConfig(record_interval=0)

    def test_rejects_non_positive_replicates(self) -> None:
        with self.assertRaises(ValueError):
            IntrinsicGoalsExperimentConfig(num_replicates=0)


class TestArmResult(unittest.TestCase):
    def test_summary_serializes_telemetry(self) -> None:
        arm = ArmResult(arm="unique")
        arm.population = [4, 6]
        arm.births = [1, 0]
        arm.deaths = [0, 1]
        arm.action_mix = {"move": [0.5, 0.25]}
        arm.gene_means = {"reward_share_bonus": [0.1, 0.2]}
        arm.final_population = 6
        arm.goal_diversity_start = {"reward_share_bonus": 0.3}
        arm.goal_diversity_end = {"reward_share_bonus": 0.1}

        payload = arm.summary()
        self.assertEqual(payload["arm"], "unique")
        self.assertEqual(payload["final_population"], 6)
        self.assertEqual(payload["mean_population"], 5.0)
        self.assertEqual(payload["peak_population"], 6)
        self.assertEqual(payload["total_births"], 1)
        self.assertEqual(payload["total_deaths"], 1)
        self.assertAlmostEqual(payload["mean_action_share"]["move"], 0.375)
        self.assertEqual(payload["goal_gene_mean_start"]["reward_share_bonus"], 0.1)
        self.assertEqual(payload["goal_gene_mean_end"]["reward_share_bonus"], 0.2)


class TestIntrinsicGoalsRunnerHelpers(unittest.TestCase):
    def test_record_action_mix_ignores_untracked_actions(self) -> None:
        arm = ArmResult(arm="unique")
        for action in TRACKED_ACTIONS:
            arm.action_mix[action] = []
        alive = [
            _FakeAgent("a0", last_action="move"),
            _FakeAgent("a1", last_action="custom_action"),
            _FakeAgent("a2", last_action=None),
        ]
        IntrinsicGoalsExperiment._record_action_mix(arm, alive)
        # Only agents with a tracked last_action_name count toward the mix.
        self.assertEqual(arm.action_mix["move"][-1], 0.5)
        self.assertEqual(arm.action_mix["gather"][-1], 0.0)

    def test_goal_diversity_is_zero_for_monoculture(self) -> None:
        agents = [_FakeAgent(f"a{i}") for i in range(3)]
        env = _FakeEnvironment(agents)
        experiment = IntrinsicGoalsExperiment(SimulationConfig())
        diversity = experiment._goal_diversity(env)
        for gene in INTRINSIC_REWARD_GENE_NAMES:
            self.assertEqual(diversity[gene], 0.0)

    def test_build_comparison_reports_unique_minus_uniform_deltas(self) -> None:
        uniform = ArmResult(arm="uniform")
        unique = ArmResult(arm="unique")
        uniform.population = [5, 5]
        unique.population = [5, 8]
        uniform.action_mix = {"move": [1.0, 1.0]}
        unique.action_mix = {"move": [0.5, 0.5]}
        uniform.goal_diversity_start = {"reward_share_bonus": 0.0}
        unique.goal_diversity_start = {"reward_share_bonus": 0.4}
        uniform.goal_diversity_end = {"reward_share_bonus": 0.0}
        unique.goal_diversity_end = {"reward_share_bonus": 0.2}
        uniform.final_population = 5
        unique.final_population = 8

        comparison = IntrinsicGoalsExperiment(SimulationConfig())._build_comparison(
            uniform, unique
        )
        self.assertEqual(comparison["final_population_delta"], 3)
        self.assertAlmostEqual(comparison["mean_population_delta"], 1.5)
        self.assertAlmostEqual(
            comparison["action_share_delta_unique_minus_uniform"]["move"], -0.5
        )
        self.assertEqual(
            comparison["start_goal_diversity"]["unique"]["reward_share_bonus"], 0.4
        )


class TestIntrinsicGoalsRunnerOrchestration(unittest.TestCase):
    def _stub_run_simulation(self, *, num_agents: int = 3, num_steps: int = 4):
        agents = [_FakeAgent(f"a{i}", last_action=TRACKED_ACTIONS[i % len(TRACKED_ACTIONS)]) for i in range(num_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*_args, **_kwargs):
            on_ready = _kwargs.get("on_environment_ready")
            on_step_end = _kwargs.get("on_step_end")
            if on_ready is not None:
                on_ready(env)
            for step in range(num_steps):
                if on_step_end is not None:
                    on_step_end(env, step)
            return env

        return _side_effect, env

    def test_run_writes_summary_and_comparison_for_single_replicate(self) -> None:
        side_effect, _env = self._stub_run_simulation(num_agents=2, num_steps=3)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_goals_experiment.run_simulation",
            side_effect=side_effect,
        ) as run_mock:
            cfg = IntrinsicGoalsExperimentConfig(
                num_steps=3,
                seed=9,
                output_dir=output_dir,
                record_interval=1,
                selection_pressure="none",
            )
            result = IntrinsicGoalsExperiment(SimulationConfig(), cfg).run()

            self.assertEqual(run_mock.call_count, 2)
            self.assertTrue(os.path.exists(result.summary_path))
            with open(result.summary_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIn("uniform", payload)
            self.assertIn("unique", payload)
            self.assertIn("comparison", payload)
            self.assertIsNone(result.aggregate)
            self.assertEqual(len(result.replicates), 1)
            self.assertTrue(result.figure_path and os.path.exists(result.figure_path))

    def test_unique_arm_assigns_distinct_goals_on_environment_ready(self) -> None:
        captured: dict[str, list[float]] = {"share_values": []}

        def _side_effect(*_args, **_kwargs):
            on_ready = _kwargs.get("on_environment_ready")
            on_step_end = _kwargs.get("on_step_end")
            agents = [_FakeAgent(f"a{i}") for i in range(4)]
            env = _FakeEnvironment(agents)
            if on_ready is not None:
                on_ready(env)
            captured["share_values"] = [
                a.hyperparameter_chromosome.get_value("reward_share_bonus")
                for a in env.alive_agent_objects
            ]
            if on_step_end is not None:
                on_step_end(env, 0)
            return env

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_goals_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            cfg = IntrinsicGoalsExperimentConfig(
                num_steps=1,
                seed=3,
                output_dir=output_dir,
                selection_pressure="none",
            )
            IntrinsicGoalsExperiment(SimulationConfig(), cfg).run()

        self.assertGreater(len(set(captured["share_values"])), 1)

    def test_multi_replicate_run_populates_aggregate_statistics(self) -> None:
        side_effect, _env = self._stub_run_simulation(num_agents=2, num_steps=2)

        def _varying_side_effect(*_args, **_kwargs):
            on_ready = _kwargs.get("on_environment_ready")
            on_step_end = _kwargs.get("on_step_end")
            seed = _kwargs.get("seed", 0)
            agents = [
                _FakeAgent(
                    f"a{i}-s{seed}",
                    last_action=TRACKED_ACTIONS[(i + seed) % len(TRACKED_ACTIONS)],
                )
                for i in range(2)
            ]
            env = _FakeEnvironment(agents)
            if on_ready is not None:
                on_ready(env)
            for step in range(2):
                if on_step_end is not None:
                    on_step_end(env, step)
            return env

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_goals_experiment.run_simulation",
            side_effect=_varying_side_effect,
        ):
            cfg = IntrinsicGoalsExperimentConfig(
                num_steps=2,
                seed=11,
                num_replicates=3,
                output_dir=output_dir,
                selection_pressure="none",
            )
            result = IntrinsicGoalsExperiment(SimulationConfig(), cfg).run()

            self.assertIsNotNone(result.aggregate)
            assert result.aggregate is not None
            self.assertEqual(result.aggregate["num_replicates"], 3)
            self.assertIn("paired_deltas", result.aggregate)
            self.assertIn("mean_population", result.aggregate["paired_deltas"])
            self.assertTrue(
                result.figure_path
                and result.figure_path.endswith("intrinsic_goals_aggregate.png")
            )

    def test_build_run_config_disables_startup_diversity(self) -> None:
        experiment = IntrinsicGoalsExperiment(SimulationConfig())
        run_config = experiment._build_run_config()
        self.assertEqual(run_config.initial_diversity.mode.value, "none")
        self.assertEqual(run_config.agent_behavior.initial_resource_level, 12.0)
        self.assertEqual(run_config.resources.initial_resources, 60)


@pytest.mark.integration
def test_short_experiment_runs_and_writes_artifacts(tmp_path):
    base_config = SimulationConfig.from_centralized_config(environment="development")
    # Keep the smoke run tiny but non-trivial.
    base_config.population.system_agents = 4
    base_config.population.independent_agents = 4
    base_config.population.control_agents = 0

    config = IntrinsicGoalsExperimentConfig(
        num_steps=12,
        seed=5,
        output_dir=str(tmp_path / "goals"),
        selection_pressure="none",
    )
    result = IntrinsicGoalsExperiment(base_config, config).run()

    assert os.path.exists(result.summary_path)
    with open(result.summary_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "uniform" in payload and "unique" in payload
    assert "comparison" in payload

    # The unique arm should start with strictly more goal diversity than the
    # uniform arm (which starts as a goal monoculture).
    start_div_unique = sum(
        result.unique.goal_diversity_start.get(g, 0.0)
        for g in INTRINSIC_REWARD_GENE_NAMES
    )
    start_div_uniform = sum(
        result.uniform.goal_diversity_start.get(g, 0.0)
        for g in INTRINSIC_REWARD_GENE_NAMES
    )
    assert start_div_uniform == pytest.approx(0.0)
    assert start_div_unique > 0.0

    # Per-step telemetry was recorded.
    assert len(result.uniform.population) > 0
    assert len(result.unique.population) > 0
