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

    def test_string_enums_coerced_to_enum_instances(self):
        """String values passed for enum fields must be normalized to enum instances."""
        policy = IntrinsicEvolutionPolicy(
            mutation_mode="gaussian",  # type: ignore[arg-type]
            boundary_mode="clamp",  # type: ignore[arg-type]
            crossover_mode="uniform",  # type: ignore[arg-type]
        )
        self.assertIsInstance(policy.mutation_mode, MutationMode)
        self.assertIsInstance(policy.boundary_mode, BoundaryMode)
        self.assertIsInstance(policy.crossover_mode, CrossoverMode)
        self.assertEqual(policy.mutation_mode, MutationMode.GAUSSIAN)
        self.assertEqual(policy.boundary_mode, BoundaryMode.CLAMP)
        self.assertEqual(policy.crossover_mode, CrossoverMode.UNIFORM)

    def test_invalid_string_enum_raises(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(mutation_mode="not_a_mode")  # type: ignore[arg-type]


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

    def test_reinitializes_decision_module_when_behavior_present(self):
        """DecisionModule config and algorithm are updated when already constructed."""
        from unittest.mock import MagicMock

        from farm.core.agent.behaviors.learning import LearningAgentBehavior

        dm = MagicMock()
        behavior = LearningAgentBehavior(dm)
        config = SimpleNamespace(decision=SimpleNamespace(learning_rate=0.01))
        chromosome = chromosome_from_learning_config(config.decision)
        agent = SimpleNamespace(
            agent_id="dm_agent",
            agent_type="system",
            alive=True,
            config=config,
            hyperparameter_chromosome=chromosome,
            behavior=behavior,
        )
        env = _FakeEnvironment([agent])
        policy = IntrinsicEvolutionPolicy(
            seed_initial_diversity=True,
            seed_mutation_rate=1.0,
            seed_mutation_scale=0.5,
        )
        seed_population_diversity(env, policy, random.Random(42))

        # DecisionModule must be reinitialized via the public reinitialize_algorithm method.
        dm.reinitialize_algorithm.assert_called_once_with(agent.config.decision)

    def test_reinitializes_any_behavior_with_compatible_decision_module(self):
        """Reinitialization is capability-based, not tied to one behavior class."""
        from unittest.mock import MagicMock

        dm = MagicMock()
        behavior = SimpleNamespace(decision_module=dm)
        config = SimpleNamespace(decision=SimpleNamespace(learning_rate=0.01))
        chromosome = chromosome_from_learning_config(config.decision)
        agent = SimpleNamespace(
            agent_id="generic_behavior_agent",
            agent_type="system",
            alive=True,
            config=config,
            hyperparameter_chromosome=chromosome,
            behavior=behavior,
        )
        env = _FakeEnvironment([agent])
        policy = IntrinsicEvolutionPolicy(
            seed_initial_diversity=True,
            seed_mutation_rate=1.0,
            seed_mutation_scale=0.5,
        )
        seed_population_diversity(env, policy, random.Random(42))

        dm.reinitialize_algorithm.assert_called_once_with(agent.config.decision)


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
            # Match run_simulation: one extra environment.update() after the loop.
            env.time += 1
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

    def test_final_result_uses_last_hooked_state(self):
        """Final metadata aligns with callback telemetry, not post-loop finalization."""
        agents = [_make_fake_agent(0.01), _make_fake_agent(0.02)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            if on_environment_ready is not None:
                on_environment_ready(env)
            # Run exactly one logical step and report it via hook.
            env.time = 1
            if on_step_end is not None:
                on_step_end(env, 0)
            # Simulate a post-loop finalization update that changes live state.
            env._agents[1].alive = False
            env.time = 2
            return env

        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=1, snapshot_interval=1, seed=7)
            base_config = SimulationConfig()
            result = IntrinsicEvolutionExperiment(base_config, cfg).run()

        # Last hooked state still had both agents alive.
        self.assertEqual(result.final_population, 2)
        self.assertEqual(result.num_steps_completed, 1)


class TestReproductionPressureConfig(unittest.TestCase):
    """Tests for the ReproductionPressureConfig dataclass."""

    def test_defaults_are_all_zero(self):
        from farm.core.agent.config.component_configs import ReproductionPressureConfig

        cfg = ReproductionPressureConfig()
        self.assertEqual(cfg.local_density_coefficient, 0.0)
        self.assertEqual(cfg.global_carrying_capacity, 0)
        self.assertEqual(cfg.global_carrying_capacity_coefficient, 0.0)

    def test_custom_values_round_trip(self):
        from farm.core.agent.config.component_configs import ReproductionPressureConfig

        cfg = ReproductionPressureConfig(
            local_density_radius=10.0,
            local_density_coefficient=1.5,
            global_carrying_capacity=200,
            global_carrying_capacity_coefficient=0.8,
        )
        self.assertEqual(cfg.local_density_radius, 10.0)
        self.assertEqual(cfg.local_density_coefficient, 1.5)
        self.assertEqual(cfg.global_carrying_capacity, 200)
        self.assertEqual(cfg.global_carrying_capacity_coefficient, 0.8)


class TestSelectionPressurePresets(unittest.TestCase):
    """Tests for the selection_pressure preset knob on IntrinsicEvolutionPolicy."""

    def test_none_preset_string_sets_zero_coefficients(self):
        policy = IntrinsicEvolutionPolicy(selection_pressure="none")
        self.assertEqual(policy.reproduction_pressure.local_density_coefficient, 0.0)
        self.assertEqual(policy.reproduction_pressure.global_carrying_capacity_coefficient, 0.0)

    def test_low_preset_string(self):
        policy = IntrinsicEvolutionPolicy(selection_pressure="low")
        self.assertGreater(policy.reproduction_pressure.local_density_coefficient, 0.0)

    def test_medium_preset_string(self):
        policy = IntrinsicEvolutionPolicy(selection_pressure="medium")
        p = policy.reproduction_pressure
        self.assertGreater(p.local_density_coefficient, 0.0)
        self.assertGreater(p.global_carrying_capacity, 0)

    def test_high_preset_string(self):
        policy = IntrinsicEvolutionPolicy(selection_pressure="high")
        p = policy.reproduction_pressure
        self.assertGreater(p.local_density_coefficient, 0.0)
        self.assertGreater(p.global_carrying_capacity, 0)

    def test_low_pressure_less_than_high(self):
        low = IntrinsicEvolutionPolicy(selection_pressure="low")
        high = IntrinsicEvolutionPolicy(selection_pressure="high")
        self.assertLess(
            low.reproduction_pressure.local_density_coefficient,
            high.reproduction_pressure.local_density_coefficient,
        )

    def test_float_zero_equals_none_preset(self):
        p_float = IntrinsicEvolutionPolicy(selection_pressure=0.0)
        p_none = IntrinsicEvolutionPolicy(selection_pressure="none")
        self.assertEqual(
            p_float.reproduction_pressure.local_density_coefficient,
            p_none.reproduction_pressure.local_density_coefficient,
        )

    def test_float_one_equals_high_preset(self):
        p_float = IntrinsicEvolutionPolicy(selection_pressure=1.0)
        p_high = IntrinsicEvolutionPolicy(selection_pressure="high")
        self.assertEqual(
            p_float.reproduction_pressure.local_density_coefficient,
            p_high.reproduction_pressure.local_density_coefficient,
        )

    def test_float_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(selection_pressure=1.5)
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(selection_pressure=-0.1)

    def test_unknown_preset_string_raises(self):
        with self.assertRaises(ValueError):
            IntrinsicEvolutionPolicy(selection_pressure="ultra")

    def test_default_no_selection_pressure(self):
        """Default policy has zero density-dependent cost."""
        policy = IntrinsicEvolutionPolicy()
        self.assertIsNone(policy.selection_pressure)
        self.assertEqual(policy.reproduction_pressure.local_density_coefficient, 0.0)

    def test_explicit_reproduction_pressure_respected_when_no_preset(self):
        """When selection_pressure=None, explicit reproduction_pressure is used."""
        from farm.core.agent.config.component_configs import ReproductionPressureConfig

        custom = ReproductionPressureConfig(local_density_coefficient=3.0)
        policy = IntrinsicEvolutionPolicy(reproduction_pressure=custom)
        self.assertEqual(policy.reproduction_pressure.local_density_coefficient, 3.0)


class TestEffectiveReproductionCost(unittest.TestCase):
    """Tests for _compute_effective_reproduction_cost."""

    def _make_agent_with_env(self, n_nearby: int = 0, pop: int = 10):
        """Build a minimal agent with a fake environment for cost computation."""
        from farm.core.agent.config.component_configs import ReproductionPressureConfig
        from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy

        pressure = ReproductionPressureConfig(
            local_density_radius=5.0,
            local_density_coefficient=1.0,
            global_carrying_capacity=100,
            global_carrying_capacity_coefficient=0.5,
        )
        policy = IntrinsicEvolutionPolicy(reproduction_pressure=pressure)

        agent = SimpleNamespace(position=(0.0, 0.0))
        # Build fake nearby agents (excluding self via 'a is not agent')
        nearby_agents = [SimpleNamespace() for _ in range(n_nearby + 1)]  # +1 self
        nearby_agents[0] = agent  # first is self

        alive = [SimpleNamespace() for _ in range(pop)]

        env = SimpleNamespace(
            intrinsic_evolution_policy=policy,
            get_nearby_agents=lambda pos, radius: nearby_agents,
            alive_agent_objects=alive,
        )
        agent.environment = env
        return agent

    def test_zero_pressure_returns_base_cost(self):
        from farm.core.agent.core import _compute_effective_reproduction_cost

        policy = IntrinsicEvolutionPolicy(selection_pressure="none")
        agent = SimpleNamespace(
            position=(0.0, 0.0),
            environment=SimpleNamespace(
                intrinsic_evolution_policy=policy,
                get_nearby_agents=lambda pos, r: [],
                alive_agent_objects=[],
            ),
        )
        cost = _compute_effective_reproduction_cost(agent, base_cost=5.0)
        self.assertEqual(cost, 5.0)

    def test_local_density_increases_cost(self):
        from farm.core.agent.core import _compute_effective_reproduction_cost

        agent = self._make_agent_with_env(n_nearby=3, pop=10)
        cost = _compute_effective_reproduction_cost(agent, base_cost=5.0)
        # 5.0 + 1.0*3 + 0.5*5.0*(10/100) = 5.0 + 3.0 + 0.25 = 8.25
        self.assertGreater(cost, 5.0)
        self.assertAlmostEqual(cost, 5.0 + 1.0 * 3 + 0.5 * 5.0 * (10 / 100), places=6)

    def test_no_environment_returns_base_cost(self):
        from farm.core.agent.core import _compute_effective_reproduction_cost

        agent = SimpleNamespace(environment=None)
        self.assertEqual(_compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_no_policy_returns_base_cost(self):
        from farm.core.agent.core import _compute_effective_reproduction_cost

        agent = SimpleNamespace(
            environment=SimpleNamespace(intrinsic_evolution_policy=None)
        )
        self.assertEqual(_compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_disabled_policy_returns_base_cost(self):
        from farm.core.agent.core import _compute_effective_reproduction_cost

        policy = IntrinsicEvolutionPolicy(enabled=False)
        agent = SimpleNamespace(
            environment=SimpleNamespace(intrinsic_evolution_policy=policy)
        )
        self.assertEqual(_compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_carrying_capacity_term_alone(self):
        from farm.core.agent.config.component_configs import ReproductionPressureConfig
        from farm.core.agent.core import _compute_effective_reproduction_cost

        pressure = ReproductionPressureConfig(
            local_density_coefficient=0.0,
            global_carrying_capacity=50,
            global_carrying_capacity_coefficient=1.0,
        )
        policy = IntrinsicEvolutionPolicy(reproduction_pressure=pressure)
        pop_count = [SimpleNamespace()] * 25  # pop / K = 0.5
        agent = SimpleNamespace(
            position=(0.0, 0.0),
            environment=SimpleNamespace(
                intrinsic_evolution_policy=policy,
                get_nearby_agents=lambda pos, r: [],
                alive_agent_objects=pop_count,
            ),
        )
        cost = _compute_effective_reproduction_cost(agent, base_cost=10.0)
        # 10.0 + 1.0 * 10.0 * (25/50) = 10.0 + 5.0 = 15.0
        self.assertAlmostEqual(cost, 15.0, places=6)


class TestTrajectoryTelemetryFields(unittest.TestCase):
    """Tests for the new selection-pressure telemetry in trajectory records."""

    def _stub_run_simulation(self, num_agents: int = 4, num_steps: int = 3):
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]

        # Add get_component stub so telemetry computation works.
        for agent in agents:
            repro_cfg = SimpleNamespace(offspring_cost=5.0)
            repro_comp = SimpleNamespace(config=repro_cfg)
            agent.get_component = lambda name, rc=repro_comp: rc if name == "reproduction" else None
            agent.position = (0.0, 0.0)
            agent.environment = None  # No density adjustment for simple test

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
            env.time += 1
            return env

        return _side_effect, env

    def test_trajectory_records_contain_telemetry_fields(self):
        side_effect, _env = self._stub_run_simulation(num_agents=3, num_steps=3)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=3,
                snapshot_interval=3,
                output_dir=output_dir,
                seed=5,
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            traj_path = os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl")
            with open(traj_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]

        # Step 0 record (from on_environment_ready) has no extra fields.
        step0 = lines[0]
        self.assertIn("step", step0)
        # Step 1+ records must carry the new telemetry fields.
        for record in lines[1:]:
            self.assertIn("mean_reproduction_cost", record)
            self.assertIn("realized_birth_rate", record)
            self.assertIn("realized_death_rate", record)
            self.assertIn("effective_selection_strength", record)

    def test_stable_population_has_zero_birth_and_death_rates(self):
        """When population is unchanged between steps, rates should be 0."""
        side_effect, _env = self._stub_run_simulation(num_agents=3, num_steps=2)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=5,
                output_dir=output_dir,
                seed=9,
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            traj_path = os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl")
            with open(traj_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]

        for record in lines[1:]:  # skip step 0
            self.assertAlmostEqual(record["realized_birth_rate"], 0.0)
            self.assertAlmostEqual(record["realized_death_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
