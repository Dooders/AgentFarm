"""Tests for the intrinsic evolution runner.

The runner drives a single simulation and patches `run_simulation` so we can
exercise the orchestration (policy attachment, seed diversity, per-step
logger snapshots, artifact persistence) without the cost of a full sim.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    CrossoverMode,
    MutationMode,
    chromosome_from_learning_config,
)
from farm.runners.intrinsic_evolution_experiment import (
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
    seed_population_diversity,
)


def _make_fake_agent(learning_rate: float = 0.01):
    """Lightweight agent stand-in carrying the attributes the runner reads."""
    config = SimpleNamespace(decision=SimpleNamespace(learning_rate=learning_rate))
    chromosome = chromosome_from_learning_config(config.decision)
    state_inner = SimpleNamespace(parent_ids=["seed"])
    return SimpleNamespace(
        agent_id=f"a_{learning_rate}",
        agent_type="system",
        generation=0,
        alive=True,
        config=config,
        hyperparameter_chromosome=chromosome,
        state=SimpleNamespace(_state=state_inner),
    )


class _FakeEnvironment:
    """Minimal environment compatible with the runner / logger contracts."""

    def __init__(self, agents):
        self._agents = list(agents)
        self.time = 0
        # Allow runner to attach policy / rng:
        self.intrinsic_evolution_policy = None
        self.intrinsic_evolution_rng = None

    @property
    def agents(self):
        return [a.agent_id for a in self._agents if a.alive]

    @property
    def agent_objects(self):
        return list(self._agents)

    @property
    def alive_agent_objects(self):
        return [a for a in self._agents if a.alive]


class TestIntrinsicEvolutionPolicy(unittest.TestCase):
    def test_defaults_construct_cleanly(self):
        policy = IntrinsicEvolutionPolicy()
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.mutation_mode, MutationMode.GAUSSIAN)
        self.assertEqual(policy.boundary_mode, BoundaryMode.CLAMP)
        self.assertEqual(policy.crossover_mode, CrossoverMode.UNIFORM)
        self.assertFalse(policy.crossover_enabled)

    def test_rejects_invalid_mutation_rate(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(mutation_rate=1.5)

    def test_rejects_negative_mutation_scale(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(mutation_scale=-0.1)

    def test_rejects_negative_radius(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(coparent_max_radius=-1.0)

    def test_rejects_unknown_coparent_strategy(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(coparent_strategy="bogus")  # type: ignore[arg-type]


class TestIntrinsicEvolutionExperimentConfig(unittest.TestCase):
    def test_rejects_zero_steps(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionExperimentConfig(num_steps=0)

    def test_rejects_zero_snapshot_interval(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionExperimentConfig(snapshot_interval=0)


class TestSeedPopulationDiversity(unittest.TestCase):
    def test_no_op_when_disabled(self):
        agents = [_make_fake_agent(0.01) for _ in range(3)]
        env = _FakeEnvironment(agents)
        original_lrs = [a.hyperparameter_chromosome.get_value("learning_rate") for a in agents]
        policy = IntrinsicEvolutionPolicy(seed_initial_diversity=False)
        seed_population_diversity(env, policy, random.Random(0))
        new_lrs = [a.hyperparameter_chromosome.get_value("learning_rate") for a in agents]
        self.assertEqual(original_lrs, new_lrs)

    def test_seeds_each_agent_with_distinct_chromosome(self):
        agents = [_make_fake_agent(0.01) for _ in range(5)]
        env = _FakeEnvironment(agents)
        policy = IntrinsicEvolutionPolicy(
            seed_initial_diversity=True,
            seed_mutation_rate=1.0,
            seed_mutation_scale=0.3,
        )
        seed_population_diversity(env, policy, random.Random(0))
        new_lrs = [a.hyperparameter_chromosome.get_value("learning_rate") for a in agents]
        # With mutation_rate=1.0 and a non-zero scale, at least one value should differ.
        self.assertTrue(any(lr != 0.01 for lr in new_lrs))
        # Decision config is updated alongside the chromosome.
        for agent, lr in zip(agents, new_lrs):
            self.assertEqual(agent.config.decision.learning_rate, lr)


class TestRunnerOrchestration(unittest.TestCase):
    def _stub_run_simulation(self, num_agents: int = 4, num_steps: int = 3):
        """Return a side-effect that mimics run_simulation's hook contract."""
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            if on_environment_ready is not None:
                on_environment_ready(env)
            for step in range(num_steps):
                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)
            return env

        return _side_effect, env

    def test_runner_attaches_policy_and_drives_loop(self):
        side_effect, env = self._stub_run_simulation(num_agents=3, num_steps=4)
        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ) as run_mock:
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=4, snapshot_interval=2, seed=7)
            base_config = SimulationConfig()
            result = IntrinsicEvolutionExperiment(base_config, cfg).run()

        self.assertEqual(run_mock.call_count, 1)
        # Policy is attached to the env exactly once during on_environment_ready.
        self.assertIsInstance(env.intrinsic_evolution_policy, IntrinsicEvolutionPolicy)
        self.assertIsInstance(env.intrinsic_evolution_rng, random.Random)
        self.assertEqual(result.final_population, 3)
        self.assertEqual(result.num_steps_completed, 4)
        self.assertIn("learning_rate", result.final_gene_statistics)

    def test_runner_persists_artifacts(self):
        side_effect, _env = self._stub_run_simulation(num_agents=2, num_steps=5)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=5,
                snapshot_interval=2,
                output_dir=output_dir,
                seed=11,
            )
            base_config = SimulationConfig()
            IntrinsicEvolutionExperiment(base_config, cfg).run()

            traj_path = os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl")
            snap_path = os.path.join(output_dir, "intrinsic_gene_snapshots.jsonl")
            meta_path = os.path.join(output_dir, "intrinsic_evolution_metadata.json")

            self.assertTrue(os.path.exists(traj_path))
            self.assertTrue(os.path.exists(snap_path))
            self.assertTrue(os.path.exists(meta_path))

            with open(traj_path, encoding="utf-8") as fh:
                trajectory_lines = [json.loads(line) for line in fh if line.strip()]
            with open(snap_path, encoding="utf-8") as fh:
                snapshot_lines = [json.loads(line) for line in fh if line.strip()]
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)

            # Trajectory: one record per snapshot call (env_ready + 5 step_end = 6).
            self.assertEqual(len(trajectory_lines), 6)
            for record in trajectory_lines:
                self.assertIn("step", record)
                self.assertIn("gene_stats", record)
                self.assertIn("learning_rate", record["gene_stats"])

            # Snapshot interval = 2: steps 0, 2, 4 -> 3 snapshots.
            self.assertEqual(len(snapshot_lines), 3)
            self.assertEqual([rec["step"] for rec in snapshot_lines], [0, 2, 4])

            self.assertEqual(metadata["num_steps_configured"], 5)
            self.assertEqual(metadata["snapshot_interval"], 2)
            self.assertIn("policy", metadata)
            # Enums in the policy must serialize to plain string values.
            self.assertEqual(metadata["policy"]["mutation_mode"], "gaussian")


if __name__ == "__main__":
    unittest.main()
