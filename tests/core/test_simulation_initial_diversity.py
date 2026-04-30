"""End-to-end tests that initial-diversity seeding fires from run_simulation.

These exercise the wiring between :class:`SimulationConfig.initial_diversity`
and :func:`farm.core.simulation.run_simulation` via a small in-process run.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from farm.config import SimulationConfig
from farm.core.initial_diversity import (
    InitialDiversityConfig,
    InitialDiversityMetrics,
    SeedingMode,
)
from farm.core.simulation import run_simulation


def _tiny_config(initial_diversity: InitialDiversityConfig) -> SimulationConfig:
    """Build a deterministic, minimal config that produces real agents."""
    cfg = SimulationConfig.from_centralized_config(environment="testing")
    cfg.simulation_steps = 1
    cfg.max_steps = 1
    cfg.population.system_agents = 3
    cfg.population.independent_agents = 2
    cfg.population.control_agents = 0
    cfg.population.order_agents = 0
    cfg.population.chaos_agents = 0
    cfg.environment.width = 20
    cfg.environment.height = 20
    cfg.database.use_in_memory_db = True
    cfg.database.persist_db_on_completion = False
    cfg.initial_diversity = initial_diversity
    return cfg


class TestRunSimulationInitialDiversity(unittest.TestCase):
    def test_none_mode_attaches_zeroed_metrics_and_no_sidecar(self):
        cfg = _tiny_config(InitialDiversityConfig(mode=SeedingMode.NONE))
        with tempfile.TemporaryDirectory() as out:
            env = run_simulation(num_steps=1, config=cfg, path=out, save_config=False, seed=7)
            metrics = getattr(env, "initial_diversity_metrics", None)
            self.assertIsInstance(metrics, InitialDiversityMetrics)
            assert metrics is not None  # for type narrowing
            self.assertIs(metrics.mode, SeedingMode.NONE)
            self.assertEqual(metrics.agents_processed, 0)
            sidecar = os.path.join(out, "initial_diversity_metadata.json")
            self.assertFalse(os.path.exists(sidecar))

    def test_independent_mutation_writes_sidecar_and_processes_agents(self):
        cfg = _tiny_config(
            InitialDiversityConfig(
                mode=SeedingMode.INDEPENDENT_MUTATION,
                mutation_rate=1.0,
                mutation_scale=0.2,
                seed=11,
            )
        )
        with tempfile.TemporaryDirectory() as out:
            env = run_simulation(num_steps=1, config=cfg, path=out, save_config=False, seed=11)
            metrics = env.initial_diversity_metrics
            self.assertIs(metrics.mode, SeedingMode.INDEPENDENT_MUTATION)
            # System (3) + independent (2) learning agents.
            self.assertEqual(metrics.agents_processed, 5)
            sidecar = os.path.join(out, "initial_diversity_metadata.json")
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["mode"], "independent_mutation")
            self.assertEqual(payload["agents_processed"], 5)

    def test_unique_mode_yields_distinct_chromosomes(self):
        cfg = _tiny_config(
            InitialDiversityConfig(
                mode=SeedingMode.UNIQUE,
                mutation_rate=1.0,
                mutation_scale=0.4,
                max_retries_per_agent=64,
                seed=23,
            )
        )
        with tempfile.TemporaryDirectory() as out:
            env = run_simulation(num_steps=1, config=cfg, path=out, save_config=False, seed=23)
            metrics = env.initial_diversity_metrics
            self.assertIs(metrics.mode, SeedingMode.UNIQUE)
            self.assertEqual(metrics.agents_processed, 5)
            self.assertEqual(metrics.unique_count, 5)
            self.assertEqual(metrics.fallbacks, 0)

    def test_min_distance_mode_records_pairwise_distance(self):
        cfg = _tiny_config(
            InitialDiversityConfig(
                mode=SeedingMode.MIN_DISTANCE,
                mutation_rate=1.0,
                mutation_scale=0.4,
                min_distance=0.01,
                max_retries_per_agent=64,
                seed=37,
            )
        )
        with tempfile.TemporaryDirectory() as out:
            env = run_simulation(num_steps=1, config=cfg, path=out, save_config=False, seed=37)
            metrics = env.initial_diversity_metrics
            self.assertIs(metrics.mode, SeedingMode.MIN_DISTANCE)
            self.assertEqual(metrics.fallbacks, 0)
            self.assertIsNotNone(metrics.min_pairwise_distance)
            assert metrics.min_pairwise_distance is not None  # type narrowing
            self.assertGreaterEqual(metrics.min_pairwise_distance, 0.01)

    def test_seeding_runs_before_on_environment_ready_hook(self):
        """The hook must observe the seeded chromosomes (post-seed state)."""
        seen_metrics = {}

        def _hook(environment):
            seen_metrics["mode"] = environment.initial_diversity_metrics.mode
            seen_metrics["count"] = environment.initial_diversity_metrics.agents_processed

        cfg = _tiny_config(
            InitialDiversityConfig(
                mode=SeedingMode.INDEPENDENT_MUTATION,
                mutation_rate=1.0,
                mutation_scale=0.1,
                seed=51,
            )
        )
        with tempfile.TemporaryDirectory() as out:
            run_simulation(
                num_steps=1,
                config=cfg,
                path=out,
                save_config=False,
                seed=51,
                on_environment_ready=_hook,
            )
        self.assertIs(seen_metrics["mode"], SeedingMode.INDEPENDENT_MUTATION)
        self.assertEqual(seen_metrics["count"], 5)


if __name__ == "__main__":
    unittest.main()
