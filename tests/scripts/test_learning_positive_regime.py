"""Tests for scripts/_learning_positive_regime.py population/ecology helpers."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import STABLE_SUB_PROFILES  # noqa: E402
from scripts._learning_positive_regime import (  # noqa: E402
    DEFAULT_LEARNING_POSITIVE_LOW_CHURN_MAX_POPULATION,
    DEFAULT_LEARNING_POSITIVE_MAX_POPULATION,
    DEFAULT_LEARNING_POSITIVE_POPULATION,
    apply_independent_population,
    build_learning_positive_regime_config,
    maybe_apply_learning_positive_population,
)
from scripts.run_inheritance_mode_ab import _build_parser, _build_runner_args  # noqa: E402
from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    _build_parser as _sweep_parser,
    _prepare_run_dir_for_resume,
    _read_completed_steps,
)


class TestIndependentPopulation(unittest.TestCase):
    def test_apply_independent_population_zeros_non_independent(self):
        config = SimulationConfig.from_centralized_config(environment="testing")
        apply_independent_population(config, population=8, max_population=32)
        self.assertEqual(config.population.independent_agents, 8)
        self.assertEqual(config.population.system_agents, 0)
        self.assertEqual(config.population.control_agents, 0)
        self.assertEqual(config.population.max_population, 32)

    def test_maybe_apply_noop_when_unset(self):
        config = SimulationConfig.from_centralized_config(environment="testing")
        before = config.population.independent_agents
        maybe_apply_learning_positive_population(config, None, None)
        self.assertEqual(config.population.independent_agents, before)

    def test_maybe_apply_defaults_max_population(self):
        config = SimulationConfig.from_centralized_config(environment="testing")
        maybe_apply_learning_positive_population(config, 8, None)
        self.assertEqual(config.population.independent_agents, 8)
        self.assertEqual(config.population.max_population, 32)


class TestLearningPositiveRegimeConfig(unittest.TestCase):
    def test_build_config_applies_ecology_and_disk(self):
        config = build_learning_positive_regime_config(
            "balanced", environment="testing", population=8, max_population=32
        )
        overrides = STABLE_SUB_PROFILES["balanced"]
        self.assertFalse(config.database.use_in_memory_db)
        self.assertEqual(
            config.resources.initial_resources, overrides["initial_resource_count"]
        )


class TestPopulationCliForwarding(unittest.TestCase):
    def test_inheritance_ab_forwards_population(self):
        args = _build_parser().parse_args(
            ["--population", "8", "--max-population", "32"]
        )
        runner_args = _build_runner_args(args, "p2", __import__("pathlib").Path("/tmp/out"))
        self.assertEqual(runner_args.population, 8)
        self.assertEqual(runner_args.max_population, 32)

    def test_seed_sweep_accepts_population_flags(self):
        args = _sweep_parser().parse_args(
            ["--population", "8", "--max-population", "32"]
        )
        self.assertEqual(args.population, 8)
        self.assertEqual(args.max_population, 32)


class TestLowChurnConstants(unittest.TestCase):
    """Validate ecology-variant defaults are self-consistent (#963)."""

    def test_low_churn_max_population_equals_default_population(self):
        self.assertEqual(
            DEFAULT_LEARNING_POSITIVE_LOW_CHURN_MAX_POPULATION,
            DEFAULT_LEARNING_POSITIVE_POPULATION,
        )

    def test_saturated_max_population_exceeds_population(self):
        self.assertGreater(
            DEFAULT_LEARNING_POSITIVE_MAX_POPULATION,
            DEFAULT_LEARNING_POSITIVE_POPULATION,
        )

    def test_low_churn_via_apply_independent_population(self):
        config = SimulationConfig.from_centralized_config(environment="testing")
        apply_independent_population(
            config,
            population=DEFAULT_LEARNING_POSITIVE_POPULATION,
            max_population=DEFAULT_LEARNING_POSITIVE_LOW_CHURN_MAX_POPULATION,
        )
        # cap == start => no growth room
        self.assertEqual(
            config.population.max_population,
            config.population.independent_agents,
        )


class TestResumeHelpers(unittest.TestCase):
    def test_read_completed_steps_missing(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(_read_completed_steps(Path(tmp)))

    def test_prepare_run_dir_removes_incomplete(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_7"
            run_dir.mkdir()
            (run_dir / "partial.db").write_text("x", encoding="utf-8")
            _prepare_run_dir_for_resume(run_dir, num_steps=3000, resume=True)
            self.assertFalse(run_dir.exists())

    def test_prepare_run_dir_keeps_complete(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            run_dir.mkdir()
            meta = {"num_steps_completed": 3000, "final_population": 32}
            (run_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            _prepare_run_dir_for_resume(run_dir, num_steps=3000, resume=True)
            self.assertTrue(run_dir.exists())


if __name__ == "__main__":
    unittest.main()

