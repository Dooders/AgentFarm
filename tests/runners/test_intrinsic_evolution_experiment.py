"""Tests for the intrinsic evolution runner.

The runner drives a single simulation and patches `run_simulation` so we can
exercise the orchestration (policy attachment, default initial-diversity
config installation, per-step logger snapshots, artifact persistence) without
the cost of a full sim.
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
from farm.core.initial_diversity import (
    InitialDiversityConfig,
    InitialDiversityMetrics,
    SeedingMode,
)
from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger as RealGeneTrajectoryLogger
from farm.runners.intrinsic_evolution_experiment import (
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
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


class TestRunnerInitialDiversityIntegration(unittest.TestCase):
    """The runner must install its own initial-diversity defaults when the
    base config has not opted in, so the starting population is not a
    monoculture by default.  Detailed behavioral tests for the seeding
    primitive itself live in tests/core/test_initial_diversity.py.
    """

    def test_runner_installs_independent_mutation_default_when_unset(self):
        base_config = SimulationConfig()
        self.assertIs(base_config.initial_diversity.mode, SeedingMode.NONE)

        with patch("farm.runners.intrinsic_evolution_experiment.run_simulation") as run_mock:
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=1, snapshot_interval=1, seed=99)
            IntrinsicEvolutionExperiment(base_config, cfg).run()

        # Caller-owned config remains unchanged; runner operates on a per-run copy.
        self.assertIs(base_config.initial_diversity.mode, SeedingMode.NONE)
        passed_config = run_mock.call_args.kwargs["config"]
        self.assertIsNot(passed_config, base_config)
        self.assertIs(passed_config.initial_diversity.mode, SeedingMode.INDEPENDENT_MUTATION)
        self.assertEqual(passed_config.initial_diversity.mutation_rate, 1.0)
        self.assertEqual(passed_config.initial_diversity.mutation_scale, 0.2)
        self.assertEqual(passed_config.initial_diversity.seed, 99)

    def test_runner_respects_caller_supplied_initial_diversity(self):
        custom = InitialDiversityConfig(
            mode=SeedingMode.UNIQUE,
            mutation_rate=0.5,
            mutation_scale=0.05,
            seed=777,
        )
        base_config = SimulationConfig()
        base_config.initial_diversity = custom

        with patch("farm.runners.intrinsic_evolution_experiment.run_simulation") as run_mock:
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=1, snapshot_interval=1, seed=42)
            IntrinsicEvolutionExperiment(base_config, cfg).run()

        # Should be left untouched because the caller already opted in.
        self.assertIs(base_config.initial_diversity, custom)
        self.assertIs(base_config.initial_diversity.mode, SeedingMode.UNIQUE)
        self.assertEqual(run_mock.call_args.kwargs["config"].initial_diversity, custom)

    def test_runner_can_preserve_explicit_none_when_default_install_disabled(self):
        base_config = SimulationConfig()
        self.assertIs(base_config.initial_diversity.mode, SeedingMode.NONE)

        with patch("farm.runners.intrinsic_evolution_experiment.run_simulation") as run_mock:
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=1,
                snapshot_interval=1,
                seed=21,
                install_default_initial_diversity=False,
            )
            IntrinsicEvolutionExperiment(base_config, cfg).run()

        passed_config = run_mock.call_args.kwargs["config"]
        self.assertIs(passed_config.initial_diversity.mode, SeedingMode.NONE)


class TestRunnerOrchestration(unittest.TestCase):
    def _stub_run_simulation(self, num_agents: int = 4, num_steps: int = 3):
        """Return a side-effect that mimics run_simulation's hook contract."""
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            # Real run_simulation attaches initial_diversity_metrics to the env
            # before invoking the hook; mimic that contract for the runner's
            # _on_environment_ready capture path.
            env.initial_diversity_metrics = InitialDiversityMetrics(
                mode=SeedingMode.INDEPENDENT_MUTATION,
                agents_processed=num_agents,
                unique_count=num_agents,
            )
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
            self.assertIn("speciation", metadata)
            self.assertEqual(metadata["speciation"]["enabled"], False)
            self.assertEqual(metadata["speciation"]["algorithm"], "gmm")
            self.assertEqual(metadata["speciation"]["max_k"], 5)
            self.assertEqual(metadata["speciation"]["seed"], 0)
            self.assertEqual(metadata["speciation"]["scaler"], "none")
            # Enums in the policy must serialize to plain string values.
            self.assertEqual(metadata["policy"]["mutation_mode"], "gaussian")
            # Initial-diversity defaults installed by the runner are
            # surfaced in the metadata for reproducibility.
            self.assertIn("initial_diversity", metadata)
            self.assertEqual(metadata["initial_diversity"]["mode"], "independent_mutation")
            self.assertIn("initial_diversity_metrics", metadata)
            self.assertEqual(
                metadata["initial_diversity_metrics"]["mode"], "independent_mutation"
            )

    def test_runner_persists_speciation_settings_when_logger_enables_it(self):
        side_effect, _env = self._stub_run_simulation(num_agents=2, num_steps=2)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ), patch(
            "farm.runners.intrinsic_evolution_experiment.GeneTrajectoryLogger"
        ) as logger_cls:
            def _make_logger(*args, **kwargs):
                return RealGeneTrajectoryLogger(
                    *args,
                    enable_speciation=True,
                    speciation_algorithm="gmm",
                    speciation_max_k=7,
                    speciation_seed=123,
                    speciation_scaler="robust",
                    **kwargs,
                )

            logger_cls.side_effect = _make_logger

            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=123,
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            meta_path = os.path.join(output_dir, "intrinsic_evolution_metadata.json")
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)

            self.assertEqual(metadata["speciation"]["enabled"], True)
            self.assertEqual(metadata["speciation"]["algorithm"], "gmm")
            self.assertEqual(metadata["speciation"]["max_k"], 7)
            self.assertEqual(metadata["speciation"]["seed"], 123)
            self.assertEqual(metadata["speciation"]["scaler"], "robust")

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
    """Tests for compute_effective_reproduction_cost."""

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
        # Build fake nearby agents (includes self as the first item so that the
        # 'a is not agent' filter in compute_effective_reproduction_cost
        # correctly yields n_nearby neighbours).
        nearby_agents = [SimpleNamespace() for _ in range(n_nearby + 1)]
        nearby_agents[0] = agent  # first slot is self

        alive = [SimpleNamespace() for _ in range(pop)]

        env = SimpleNamespace(
            intrinsic_evolution_policy=policy,
            get_nearby_agents=lambda pos, radius: nearby_agents,
            alive_agent_objects=alive,
        )
        agent.environment = env
        return agent

    def test_zero_pressure_returns_base_cost(self):
        from farm.core.agent.core import compute_effective_reproduction_cost

        policy = IntrinsicEvolutionPolicy(selection_pressure="none")
        agent = SimpleNamespace(
            position=(0.0, 0.0),
            environment=SimpleNamespace(
                intrinsic_evolution_policy=policy,
                get_nearby_agents=lambda pos, r: [],
                alive_agent_objects=[],
            ),
        )
        cost = compute_effective_reproduction_cost(agent, base_cost=5.0)
        self.assertEqual(cost, 5.0)

    def test_local_density_increases_cost(self):
        from farm.core.agent.core import compute_effective_reproduction_cost

        agent = self._make_agent_with_env(n_nearby=3, pop=10)
        cost = compute_effective_reproduction_cost(agent, base_cost=5.0)
        # 5.0 + 1.0*3 + 0.5*5.0*(10/100) = 5.0 + 3.0 + 0.25 = 8.25
        self.assertGreater(cost, 5.0)
        self.assertAlmostEqual(cost, 5.0 + 1.0 * 3 + 0.5 * 5.0 * (10 / 100), places=6)

    def test_no_environment_returns_base_cost(self):
        from farm.core.agent.core import compute_effective_reproduction_cost

        agent = SimpleNamespace(environment=None)
        self.assertEqual(compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_no_policy_returns_base_cost(self):
        from farm.core.agent.core import compute_effective_reproduction_cost

        agent = SimpleNamespace(
            environment=SimpleNamespace(intrinsic_evolution_policy=None)
        )
        self.assertEqual(compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_disabled_policy_returns_base_cost(self):
        from farm.core.agent.core import compute_effective_reproduction_cost

        policy = IntrinsicEvolutionPolicy(enabled=False)
        agent = SimpleNamespace(
            environment=SimpleNamespace(intrinsic_evolution_policy=policy)
        )
        self.assertEqual(compute_effective_reproduction_cost(agent, 5.0), 5.0)

    def test_carrying_capacity_term_alone(self):
        from farm.core.agent.config.component_configs import ReproductionPressureConfig
        from farm.core.agent.core import compute_effective_reproduction_cost

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
        cost = compute_effective_reproduction_cost(agent, base_cost=10.0)
        # 10.0 + 1.0 * 10.0 * (25/50) = 10.0 + 5.0 = 15.0
        self.assertAlmostEqual(cost, 15.0, places=6)


class TestTrajectoryTelemetryFields(unittest.TestCase):
    """Tests for the new selection-pressure telemetry in trajectory records."""

    def _stub_run_simulation(self, num_agents: int = 4, num_steps: int = 3):
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]

        # Add get_component stub so telemetry computation works.
        def _make_get_component(rc):
            def _get_component(name):
                return rc if name == "reproduction" else None
            return _get_component

        for agent in agents:
            repro_cfg = SimpleNamespace(offspring_cost=5.0)
            repro_comp = SimpleNamespace(config=repro_cfg)
            agent.get_component = _make_get_component(repro_comp)
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


# ──────────────────────────────────────────────────────────────────────────────
# InitialConditionsConfig tests
# ──────────────────────────────────────────────────────────────────────────────


class TestInitialConditionsConfig(unittest.TestCase):
    """Unit tests for InitialConditionsConfig validation and resolve()."""

    def test_default_profile_is_stable(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig()
        self.assertEqual(cfg.profile, "stable")

    def test_valid_profiles_accepted(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        for profile in ("stable", "stress", "exploratory", "legacy"):
            with self.subTest(profile=profile):
                cfg = InitialConditionsConfig(profile=profile)
                self.assertEqual(cfg.profile, profile)

    def test_none_profile_accepted(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig(profile=None)
        self.assertIsNone(cfg.profile)

    def test_unknown_profile_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(profile="nonexistent")

    def test_negative_warmup_steps_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(warmup_steps=-1)

    def test_zero_transient_window_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(transient_window=0)

    def test_negative_initial_agent_resource_level_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(initial_agent_resource_level=-1.0)

    def test_negative_initial_resource_count_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(initial_resource_count=-1)

    def test_out_of_range_resource_regen_rate_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(resource_regen_rate=1.1)

    def test_negative_resource_regen_amount_raises(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        with self.assertRaises(ValueError):
            InitialConditionsConfig(resource_regen_amount=-1)

    def test_stable_profile_resolve_returns_higher_resources(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        stable = InitialConditionsConfig(profile="stable").resolve()
        legacy = InitialConditionsConfig(profile="legacy").resolve()

        # stable must give agents more resources than legacy (which has None → no override)
        self.assertIsNotNone(stable["initial_agent_resource_level"])
        self.assertIsNone(legacy["initial_agent_resource_level"])

    def test_stable_resource_level_greater_than_stress(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        stable = InitialConditionsConfig(profile="stable").resolve()
        stress = InitialConditionsConfig(profile="stress").resolve()
        self.assertGreater(
            stable["initial_agent_resource_level"],
            stress["initial_agent_resource_level"],
        )

    def test_manual_override_wins_over_preset(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig(
            profile="stable",
            initial_agent_resource_level=999,
        )
        resolved = cfg.resolve()
        self.assertEqual(resolved["initial_agent_resource_level"], 999)

    def test_none_profile_with_all_none_overrides_gives_all_none(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig(profile=None)
        resolved = cfg.resolve()
        self.assertIsNone(resolved["initial_agent_resource_level"])
        self.assertIsNone(resolved["initial_resource_count"])
        self.assertIsNone(resolved["resource_regen_rate"])
        self.assertIsNone(resolved["resource_regen_amount"])

    def test_resolve_always_includes_warmup_and_window_keys(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig(warmup_steps=5, transient_window=20)
        resolved = cfg.resolve()
        self.assertEqual(resolved["warmup_steps"], 5)
        self.assertEqual(resolved["transient_window"], 20)

    def test_to_dict_round_trips_profile(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig

        cfg = InitialConditionsConfig(profile="stress", warmup_steps=3)
        d = cfg.to_dict()
        self.assertEqual(d["profile"], "stress")
        self.assertEqual(d["warmup_steps"], 3)
        self.assertIn("resolved", d)


class TestInitialConditionsAppliedToConfig(unittest.TestCase):
    """The runner must apply resolved initial-condition overrides to run_config."""

    def _run_with_profile(self, profile: str):
        """Run with the given profile and return the config passed to run_simulation."""
        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation"
        ) as run_mock:
            from farm.runners.intrinsic_evolution_experiment import (
                InitialConditionsConfig,
                IntrinsicEvolutionExperiment,
                IntrinsicEvolutionExperimentConfig,
            )

            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=1,
                snapshot_interval=1,
                seed=7,
                initial_conditions=InitialConditionsConfig(profile=profile),
            )
            base_config = SimulationConfig()
            IntrinsicEvolutionExperiment(base_config, cfg).run()

        return run_mock.call_args.kwargs["config"]

    def test_stable_profile_overrides_agent_resource_level(self):
        """stable profile must set a non-zero initial_resource_level on run config."""
        run_config = self._run_with_profile("stable")
        # stable preset sets initial_agent_resource_level=20.0
        self.assertEqual(run_config.agent_behavior.initial_resource_level, 20)

    def test_stable_profile_overrides_resource_count(self):
        run_config = self._run_with_profile("stable")
        # stable preset sets initial_resource_count=30
        self.assertEqual(run_config.resources.initial_resources, 30)

    def test_stable_profile_overrides_regen_rate(self):
        run_config = self._run_with_profile("stable")
        self.assertAlmostEqual(run_config.resources.resource_regen_rate, 0.15)

    def test_legacy_profile_does_not_modify_agent_resource_level(self):
        """legacy profile has None overrides → base config values unchanged."""
        base = SimulationConfig()
        original_level = base.agent_behavior.initial_resource_level

        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation"
        ) as run_mock:
            from farm.runners.intrinsic_evolution_experiment import (
                InitialConditionsConfig,
                IntrinsicEvolutionExperiment,
                IntrinsicEvolutionExperimentConfig,
            )

            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=1,
                snapshot_interval=1,
                seed=3,
                initial_conditions=InitialConditionsConfig(profile="legacy"),
            )
            IntrinsicEvolutionExperiment(base, cfg).run()

        passed_config = run_mock.call_args.kwargs["config"]
        self.assertEqual(passed_config.agent_behavior.initial_resource_level, original_level)

    def test_manual_override_applied_regardless_of_profile(self):
        """An explicit override wins over any profile."""
        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation"
        ) as run_mock:
            from farm.runners.intrinsic_evolution_experiment import (
                InitialConditionsConfig,
                IntrinsicEvolutionExperiment,
                IntrinsicEvolutionExperimentConfig,
            )

            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=1,
                snapshot_interval=1,
                seed=2,
                initial_conditions=InitialConditionsConfig(
                    profile="legacy",
                    initial_agent_resource_level=42,
                ),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

        passed_config = run_mock.call_args.kwargs["config"]
        self.assertEqual(passed_config.agent_behavior.initial_resource_level, 42)

    def test_caller_base_config_not_modified(self):
        """The runner must operate on a copy; the caller's config must remain unchanged."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        base = SimulationConfig()
        original_level = base.agent_behavior.initial_resource_level

        with patch("farm.runners.intrinsic_evolution_experiment.run_simulation"):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=1,
                snapshot_interval=1,
                seed=5,
                initial_conditions=InitialConditionsConfig(profile="stable"),
            )
            IntrinsicEvolutionExperiment(base, cfg).run()

        self.assertEqual(base.agent_behavior.initial_resource_level, original_level)


class TestStartupTransientMetrics(unittest.TestCase):
    """Tests for the startup-transient metrics computation and result field."""

    def _make_run_with_population_change(self, initial_agents: int, step_deaths: int):
        """Simulate a run where `step_deaths` agents die on step 1."""
        agents = [_make_fake_agent(0.01) for _ in range(initial_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            if on_environment_ready is not None:
                on_environment_ready(env)
            # Kill some agents on step 0 → appears as deaths on step 1
            for i in range(step_deaths):
                env._agents[i].alive = False
            env.time = 1
            if on_step_end is not None:
                on_step_end(env, 0)
            env.time = 2
            return env

        return _side_effect, env

    def test_startup_transient_metrics_in_result(self):
        from farm.runners.intrinsic_evolution_experiment import (
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect, _ = self._make_run_with_population_change(
            initial_agents=4, step_deaths=2
        )
        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=1, snapshot_interval=1, seed=1)
            result = IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

        self.assertIn("peak_death_rate", result.startup_transient_metrics)
        self.assertIn("peak_birth_rate", result.startup_transient_metrics)
        self.assertIn("oscillation_amplitude", result.startup_transient_metrics)
        self.assertIn("n_steps_observed", result.startup_transient_metrics)
        self.assertIn("transient_window", result.startup_transient_metrics)

    def test_nonzero_death_rate_recorded(self):
        from farm.runners.intrinsic_evolution_experiment import (
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect, _ = self._make_run_with_population_change(
            initial_agents=4, step_deaths=2
        )
        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(num_steps=1, snapshot_interval=1, seed=1)
            result = IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

        # 2 deaths out of 4 initial agents → rate = 0.5
        self.assertAlmostEqual(
            result.startup_transient_metrics["peak_death_rate"], 0.5, places=6
        )

    def test_transient_window_respected(self):
        """Steps beyond transient_window are not counted in the metric."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        agents = [_make_fake_agent(0.01) for _ in range(3)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            if on_environment_ready is not None:
                on_environment_ready(env)
            for step in range(5):
                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)
            env.time = 6
            return env

        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=5,
                snapshot_interval=1,
                seed=42,
                initial_conditions=InitialConditionsConfig(transient_window=2),
            )
            result = IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

        # Only 2 steps recorded in transient window.
        self.assertEqual(result.startup_transient_metrics["n_steps_observed"], 2)
        self.assertEqual(result.startup_transient_metrics["transient_window"], 2)

    def test_empty_input_series_give_zero_transient_metrics(self):
        """Empty inputs to _compute_startup_transient_metrics produce zeroed metrics."""
        from farm.runners.intrinsic_evolution_experiment import (
            _compute_startup_transient_metrics,
        )

        metrics = _compute_startup_transient_metrics([], [], [], transient_window=50)
        self.assertEqual(metrics["peak_birth_rate"], 0.0)
        self.assertEqual(metrics["peak_death_rate"], 0.0)
        self.assertEqual(metrics["oscillation_amplitude"], 0)
        self.assertEqual(metrics["n_steps_observed"], 0)
        self.assertEqual(metrics["transient_window"], 50)

    def _make_seeded_benchmark_side_effect(self):
        """Deterministic startup benchmark model keyed by seed + run config.

        This is a lightweight stand-in for an early-steps ecology where higher
        startup resources reduce initial mortality and therefore the boom-bust
        wave amplitude.
        """

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            run_config = kwargs.get("config")
            seed = int(kwargs.get("seed") or 0)
            rng = random.Random(seed)

            agents = [_make_fake_agent(0.01 + float(i) / 1000) for i in range(12)]
            env = _FakeEnvironment(agents)
            env.initial_diversity_metrics = None
            if on_environment_ready is not None:
                on_environment_ready(env)

            # Resource-rich starts (stable profile) should produce lower early
            # deaths than legacy-like starts under the same seed.
            initial_level = float(getattr(run_config.agent_behavior, "initial_resource_level", 0))
            initial_resources = int(getattr(run_config.resources, "initial_resources", 0))
            scarcity = max(
                0.0,
                1.0
                - min(initial_level / 20.0, 1.0) * 0.6
                - min(initial_resources / 30.0, 1.0) * 0.4,
            )

            # Use only the first few steps as the startup-transient benchmark window.
            for step in range(5):
                alive = [a for a in env._agents if a.alive]
                prev = len(alive)
                if prev <= 1:
                    env.time = step + 1
                    if on_step_end is not None:
                        on_step_end(env, step)
                    continue

                deaths = int(round(scarcity * 4))
                if (step + rng.randint(0, 1)) % 3 == 0:
                    deaths += int(round(scarcity))
                deaths = min(prev - 1, max(0, deaths))
                for victim in alive[:deaths]:
                    victim.alive = False

                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)

            env.time = 6
            return env

        return _side_effect

    def test_stable_profile_reduces_startup_wave_vs_legacy_benchmark(self):
        """Benchmark acceptance check: stable startup wave < legacy startup wave."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect = self._make_seeded_benchmark_side_effect()
        representative_seeds = [7, 19, 41]
        for seed in representative_seeds:
            with self.subTest(seed=seed):
                with patch(
                    "farm.runners.intrinsic_evolution_experiment.run_simulation",
                    side_effect=side_effect,
                ):
                    stable_cfg = IntrinsicEvolutionExperimentConfig(
                        num_steps=5,
                        snapshot_interval=1,
                        seed=seed,
                        initial_conditions=InitialConditionsConfig(profile="stable"),
                    )
                    stable_result = IntrinsicEvolutionExperiment(
                        SimulationConfig(), stable_cfg
                    ).run()

                with patch(
                    "farm.runners.intrinsic_evolution_experiment.run_simulation",
                    side_effect=side_effect,
                ):
                    legacy_cfg = IntrinsicEvolutionExperimentConfig(
                        num_steps=5,
                        snapshot_interval=1,
                        seed=seed,
                        initial_conditions=InitialConditionsConfig(profile="legacy"),
                    )
                    legacy_result = IntrinsicEvolutionExperiment(
                        SimulationConfig(), legacy_cfg
                    ).run()

                self.assertLess(
                    stable_result.startup_transient_metrics["peak_death_rate"],
                    legacy_result.startup_transient_metrics["peak_death_rate"],
                )
                self.assertLess(
                    stable_result.startup_transient_metrics["oscillation_amplitude"],
                    legacy_result.startup_transient_metrics["oscillation_amplitude"],
                )


class TestMetadataPersistsInitialConditions(unittest.TestCase):
    """Metadata JSON must include initial_conditions and startup_transient_metrics."""

    def _stub_run_simulation(self, num_agents: int = 2, num_steps: int = 2):
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            env.initial_diversity_metrics = None
            if on_environment_ready is not None:
                on_environment_ready(env)
            for step in range(num_steps):
                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)
            env.time += 1
            return env

        return _side_effect

    def test_metadata_contains_initial_conditions_section(self):
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect = self._stub_run_simulation()
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=77,
                initial_conditions=InitialConditionsConfig(profile="stable"),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            with open(
                os.path.join(output_dir, "intrinsic_evolution_metadata.json"),
                encoding="utf-8",
            ) as fh:
                meta = json.load(fh)

        self.assertIn("initial_conditions", meta)
        self.assertEqual(meta["initial_conditions"]["profile"], "stable")
        self.assertIn("resolved", meta["initial_conditions"])

    def test_metadata_contains_resolved_initial_conditions(self):
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect = self._stub_run_simulation()
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=88,
                initial_conditions=InitialConditionsConfig(profile="stable"),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            with open(
                os.path.join(output_dir, "intrinsic_evolution_metadata.json"),
                encoding="utf-8",
            ) as fh:
                meta = json.load(fh)

        self.assertIn("resolved_initial_conditions", meta)
        # stable preset sets initial_agent_resource_level to a non-None value
        self.assertIsNotNone(
            meta["resolved_initial_conditions"]["initial_agent_resource_level"]
        )

    def test_metadata_contains_startup_transient_metrics(self):
        from farm.runners.intrinsic_evolution_experiment import (
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect = self._stub_run_simulation()
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=99,
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            with open(
                os.path.join(output_dir, "intrinsic_evolution_metadata.json"),
                encoding="utf-8",
            ) as fh:
                meta = json.load(fh)

        self.assertIn("startup_transient_metrics", meta)
        stm = meta["startup_transient_metrics"]
        self.assertIn("peak_birth_rate", stm)
        self.assertIn("peak_death_rate", stm)
        self.assertIn("oscillation_amplitude", stm)


class TestWarmupSteps(unittest.TestCase):
    """Warmup steps suppress gene-logger snapshots for the first N steps."""

    def _build_side_effect(self, num_agents: int, num_steps: int):
        """Side-effect that honours total_sim_steps from the run_simulation call."""
        agents = [_make_fake_agent(0.01) for _ in range(num_agents)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            actual_steps = kwargs.get("num_steps", num_steps)
            env.initial_diversity_metrics = None
            if on_environment_ready is not None:
                on_environment_ready(env)
            for step in range(actual_steps):
                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)
            env.time += 1
            return env

        return _side_effect, env

    def test_warmup_suppresses_initial_snapshot(self):
        """With warmup_steps=2 and num_steps=3, snapshot at step 0 must be absent."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect, _ = self._build_side_effect(num_agents=2, num_steps=5)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=3,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=11,
                initial_conditions=InitialConditionsConfig(
                    profile="legacy", warmup_steps=2
                ),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            traj_path = os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl")
            with open(traj_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]

        # No step-0 record; post-warmup steps start at 1.
        recorded_steps = [r["step"] for r in lines]
        self.assertNotIn(0, recorded_steps)
        # Post-warmup steps are numbered starting at 1.
        self.assertIn(1, recorded_steps)

    def test_warmup_zero_behaves_like_no_warmup(self):
        """warmup_steps=0 must record step 0 as in the non-warmup path."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        side_effect, _ = self._build_side_effect(num_agents=2, num_steps=2)
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=2,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=13,
                initial_conditions=InitialConditionsConfig(
                    profile="legacy", warmup_steps=0
                ),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

            traj_path = os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl")
            with open(traj_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]

        recorded_steps = [r["step"] for r in lines]
        self.assertIn(0, recorded_steps)

    def test_warmup_extends_total_sim_steps(self):
        """run_simulation must be called with num_steps + warmup_steps."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        with patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation"
        ) as run_mock:
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=10,
                snapshot_interval=1,
                seed=5,
                initial_conditions=InitialConditionsConfig(
                    profile="legacy", warmup_steps=5
                ),
            )
            IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()

        called_num_steps = run_mock.call_args.kwargs["num_steps"]
        self.assertEqual(called_num_steps, 15)  # 10 + 5


class TestDeterministicReproducibility(unittest.TestCase):
    """Fixed seeds must reproduce identical results for repeated runs of the same initial-conditions profile."""

    def _collect_snapshots(self, profile: str, seed: int, num_steps: int = 4):
        """Run with a stub and return the list of trajectory records."""
        from farm.runners.intrinsic_evolution_experiment import (
            InitialConditionsConfig,
            IntrinsicEvolutionExperiment,
            IntrinsicEvolutionExperimentConfig,
        )

        agents = [_make_fake_agent(0.01 + float(i) / 100) for i in range(3)]
        env = _FakeEnvironment(agents)

        def _side_effect(*args, **kwargs):
            on_environment_ready = kwargs.get("on_environment_ready")
            on_step_end = kwargs.get("on_step_end")
            actual_steps = kwargs.get("num_steps", num_steps)
            env.initial_diversity_metrics = None
            if on_environment_ready is not None:
                on_environment_ready(env)
            for step in range(actual_steps):
                env.time = step + 1
                if on_step_end is not None:
                    on_step_end(env, step)
            env.time += 1
            return env

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "farm.runners.intrinsic_evolution_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            cfg = IntrinsicEvolutionExperimentConfig(
                num_steps=num_steps,
                snapshot_interval=1,
                output_dir=output_dir,
                seed=seed,
                initial_conditions=InitialConditionsConfig(profile=profile),
            )
            result = IntrinsicEvolutionExperiment(SimulationConfig(), cfg).run()
        return result

    def test_same_seed_same_profile_gives_same_result(self):
        """Two runs with the same seed and profile must produce identical results."""
        r1 = self._collect_snapshots("stable", seed=42)
        r2 = self._collect_snapshots("stable", seed=42)
        self.assertEqual(r1.final_population, r2.final_population)
        self.assertEqual(r1.num_steps_completed, r2.num_steps_completed)
        self.assertEqual(
            r1.startup_transient_metrics, r2.startup_transient_metrics
        )

    def test_legacy_profile_same_seed_reproducible(self):
        r1 = self._collect_snapshots("legacy", seed=77)
        r2 = self._collect_snapshots("legacy", seed=77)
        self.assertEqual(r1.final_population, r2.final_population)
        self.assertEqual(r1.num_steps_completed, r2.num_steps_completed)


if __name__ == "__main__":
    unittest.main()
