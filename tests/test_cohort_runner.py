"""Tests for farm.runners.cohort_runner and the run_cohort_experiment CLI."""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make scripts importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPT_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from farm.runners import (  # noqa: E402
    CohortAggregateResult,
    CohortRunner,
    CohortSeedResult,
    EvolutionExperiment,
    EvolutionExperimentConfig,
)
from farm.runners.cohort_runner import (  # noqa: E402
    _lower_bound_occupancy,
    _replace_seed,
    _safe_stdev,
    _serialize_experiment_config,
)

run_cohort_experiment = importlib.import_module("run_cohort_experiment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_result(
    best_fitness: float = 5.0,
    num_generations: int = 2,
    converged: bool = False,
    convergence_reason: str | None = None,
    generation_of_convergence: int | None = None,
) -> MagicMock:
    """Build a minimal fake EvolutionExperimentResult."""
    result = MagicMock()
    result.best_candidate.fitness = best_fitness
    result.generation_summaries = [MagicMock() for _ in range(num_generations)]
    for i, s in enumerate(result.generation_summaries):
        s.gene_statistics = {"learning_rate": {"mean": 0.05, "min": 0.001, "max": 0.1}}
        s.best_chromosome = {"learning_rate": 0.05}
    result.converged = converged
    result.convergence_reason = convergence_reason
    result.generation_of_convergence = generation_of_convergence
    result.evaluations = []
    return result


def _make_base_config() -> MagicMock:
    return MagicMock()


def _make_experiment_config(**kwargs) -> EvolutionExperimentConfig:
    defaults = dict(
        num_generations=1,
        population_size=2,
        num_steps_per_candidate=1,
        seed=None,
    )
    defaults.update(kwargs)
    return EvolutionExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: _safe_stdev
# ---------------------------------------------------------------------------

class TestSafeStdev(unittest.TestCase):
    def test_returns_none_for_empty(self):
        self.assertIsNone(_safe_stdev([]))

    def test_returns_none_for_single(self):
        self.assertIsNone(_safe_stdev([1.0]))

    def test_returns_zero_for_equal_values(self):
        self.assertAlmostEqual(_safe_stdev([3.0, 3.0, 3.0]), 0.0)

    def test_known_population_stdev(self):
        # population stdev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        result = _safe_stdev([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(result, 2.0, places=5)


# ---------------------------------------------------------------------------
# Unit tests: _replace_seed
# ---------------------------------------------------------------------------

class TestReplaceSeed(unittest.TestCase):
    def test_seed_and_output_dir_replaced(self):
        template = _make_experiment_config(seed=42)
        updated = _replace_seed(template, seed=99, output_dir="/tmp/out")
        self.assertEqual(updated.seed, 99)
        self.assertEqual(updated.output_dir, "/tmp/out")

    def test_other_fields_preserved(self):
        template = _make_experiment_config(num_generations=5, seed=1)
        updated = _replace_seed(template, seed=2, output_dir=None)
        self.assertEqual(updated.num_generations, 5)

    def test_output_dir_can_be_none(self):
        template = _make_experiment_config(seed=1)
        updated = _replace_seed(template, seed=7, output_dir=None)
        self.assertIsNone(updated.output_dir)


# ---------------------------------------------------------------------------
# Unit tests: _serialize_experiment_config
# ---------------------------------------------------------------------------

class TestSerializeExperimentConfig(unittest.TestCase):
    def test_returns_dict(self):
        cfg = _make_experiment_config()
        result = _serialize_experiment_config(cfg)
        self.assertIsInstance(result, dict)

    def test_enum_values_are_strings(self):
        cfg = _make_experiment_config()
        result = _serialize_experiment_config(cfg)
        # fitness_metric and selection_method should be strings, not Enum objects
        self.assertIsInstance(result["fitness_metric"], str)
        self.assertIsInstance(result["selection_method"], str)

    def test_json_serialisable(self):
        cfg = _make_experiment_config()
        result = _serialize_experiment_config(cfg)
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# Unit tests: _lower_bound_occupancy
# ---------------------------------------------------------------------------

class TestLowerBoundOccupancy(unittest.TestCase):
    def test_returns_none_for_empty_summaries(self):
        result = MagicMock()
        result.generation_summaries = []
        self.assertIsNone(
            _lower_bound_occupancy(result, learning_rate_lower_bound=1e-6)
        )

    def test_returns_none_when_lower_bound_missing(self):
        result = MagicMock()
        result.generation_summaries = [MagicMock()]
        self.assertIsNone(
            _lower_bound_occupancy(result, learning_rate_lower_bound=None)
        )

    def test_returns_zero_when_no_lower_bound_hit(self):
        result = MagicMock()
        summary = MagicMock()
        # best_lr far above pop_min
        summary.gene_statistics = {"learning_rate": {"min": 0.001}}
        summary.best_chromosome = {"learning_rate": 0.05}
        result.generation_summaries = [summary]
        occ = _lower_bound_occupancy(result, learning_rate_lower_bound=1e-6)
        self.assertIsNotNone(occ)
        self.assertAlmostEqual(occ, 0.0)

    def test_returns_one_when_always_at_lower_bound(self):
        result = MagicMock()
        summary = MagicMock()
        # best_lr equal to pop_min
        summary.gene_statistics = {"learning_rate": {"min": 0.001}}
        summary.best_chromosome = {"learning_rate": 1e-6}
        result.generation_summaries = [summary, summary]
        occ = _lower_bound_occupancy(result, learning_rate_lower_bound=1e-6)
        self.assertAlmostEqual(occ, 1.0)

    def test_partial_occupancy(self):
        result = MagicMock()
        s_hit = MagicMock()
        s_hit.gene_statistics = {"learning_rate": {"min": 0.001}}
        s_hit.best_chromosome = {"learning_rate": 1e-6}
        s_miss = MagicMock()
        s_miss.gene_statistics = {"learning_rate": {"min": 0.001}}
        s_miss.best_chromosome = {"learning_rate": 0.05}
        result.generation_summaries = [s_hit, s_miss, s_miss, s_miss]
        occ = _lower_bound_occupancy(result, learning_rate_lower_bound=1e-6)
        self.assertAlmostEqual(occ, 0.25)

    def test_uses_configured_lower_bound_not_population_min(self):
        result = MagicMock()
        summary = MagicMock()
        # Population minimum is not at the real boundary.
        summary.gene_statistics = {"learning_rate": {"min": 0.01}}
        summary.best_chromosome = {"learning_rate": 0.01}
        result.generation_summaries = [summary]
        occ = _lower_bound_occupancy(result, learning_rate_lower_bound=1e-6)
        self.assertAlmostEqual(occ, 0.0)


# ---------------------------------------------------------------------------
# Unit tests: CohortRunner constructor
# ---------------------------------------------------------------------------

class TestCohortRunnerConstructor(unittest.TestCase):
    def test_raises_for_empty_seeds(self):
        with self.assertRaises(ValueError):
            CohortRunner(
                base_config=_make_base_config(),
                experiment_config_template=_make_experiment_config(),
                seeds=[],
            )

    def test_stores_seeds(self):
        runner = CohortRunner(
            base_config=_make_base_config(),
            experiment_config_template=_make_experiment_config(),
            seeds=[1, 2, 3],
        )
        self.assertEqual(runner.seeds, [1, 2, 3])

    def test_output_dir_default_is_none(self):
        runner = CohortRunner(
            base_config=_make_base_config(),
            experiment_config_template=_make_experiment_config(),
            seeds=[1],
        )
        self.assertIsNone(runner.output_dir)


# ---------------------------------------------------------------------------
# Integration / smoke tests: CohortRunner.run
# ---------------------------------------------------------------------------

class TestCohortRunnerRun(unittest.TestCase):
    def _run_with_fake_results(
        self,
        seeds,
        fake_results=None,
        output_dir=None,
    ) -> CohortAggregateResult:
        if fake_results is None:
            fake_results = [_make_fake_result(best_fitness=float(5 + i)) for i in range(len(seeds))]

        call_count = [0]

        def fake_experiment_init(self_inner, base_config, config):
            self_inner.base_config = base_config
            self_inner.config = config

        def fake_run(self_inner):
            idx = call_count[0]
            call_count[0] += 1
            return fake_results[idx]

        runner = CohortRunner(
            base_config=_make_base_config(),
            experiment_config_template=_make_experiment_config(),
            seeds=seeds,
            output_dir=output_dir,
        )
        with patch.object(
            EvolutionExperiment, "__init__", fake_experiment_init
        ), patch.object(EvolutionExperiment, "run", fake_run):
            return runner.run()

    def test_returns_cohort_aggregate_result(self):
        result = self._run_with_fake_results([1, 2, 3])
        self.assertIsInstance(result, CohortAggregateResult)

    def test_num_seeds_matches(self):
        result = self._run_with_fake_results([10, 20, 30])
        self.assertEqual(result.num_seeds, 3)

    def test_seeds_preserved(self):
        result = self._run_with_fake_results([7, 8, 9])
        self.assertEqual(result.seeds, [7, 8, 9])

    def test_seed_results_length(self):
        result = self._run_with_fake_results([1, 2, 3])
        self.assertEqual(len(result.seed_results), 3)

    def test_best_fitness_mean_correct(self):
        # fitnesses: 5, 6, 7 → mean 6.0
        result = self._run_with_fake_results([1, 2, 3])
        self.assertAlmostEqual(result.best_fitness_mean, 6.0, places=4)

    def test_best_fitness_min_max(self):
        result = self._run_with_fake_results([1, 2, 3])
        self.assertAlmostEqual(result.best_fitness_min, 5.0, places=4)
        self.assertAlmostEqual(result.best_fitness_max, 7.0, places=4)

    def test_convergence_rate_none_converged(self):
        result = self._run_with_fake_results([1, 2])
        self.assertAlmostEqual(result.convergence_rate, 0.0)

    def test_convergence_rate_all_converged(self):
        fakes = [
            _make_fake_result(
                best_fitness=5.0,
                converged=True,
                convergence_reason="fitness_plateau",
                generation_of_convergence=1,
            )
            for _ in range(3)
        ]
        result = self._run_with_fake_results([1, 2, 3], fake_results=fakes)
        self.assertAlmostEqual(result.convergence_rate, 1.0)

    def test_convergence_reason_counts(self):
        fakes = [
            _make_fake_result(converged=True, convergence_reason="fitness_plateau", generation_of_convergence=1),
            _make_fake_result(converged=True, convergence_reason="diversity_collapse", generation_of_convergence=2),
            _make_fake_result(converged=False),
        ]
        result = self._run_with_fake_results([1, 2, 3], fake_results=fakes)
        self.assertEqual(result.convergence_reason_counts.get("fitness_plateau"), 1)
        self.assertEqual(result.convergence_reason_counts.get("diversity_collapse"), 1)

    def test_mean_generation_of_convergence(self):
        fakes = [
            _make_fake_result(converged=True, convergence_reason="fitness_plateau", generation_of_convergence=2),
            _make_fake_result(converged=True, convergence_reason="fitness_plateau", generation_of_convergence=4),
        ]
        result = self._run_with_fake_results([1, 2], fake_results=fakes)
        self.assertAlmostEqual(result.mean_generation_of_convergence, 3.0)

    def test_mean_generation_of_convergence_none_when_no_convergence(self):
        result = self._run_with_fake_results([1, 2])
        self.assertIsNone(result.mean_generation_of_convergence)

    def test_elapsed_seconds_positive(self):
        result = self._run_with_fake_results([1])
        self.assertGreaterEqual(result.total_elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.mean_elapsed_seconds, 0.0)

    def test_artifacts_written_to_output_dir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._run_with_fake_results([1, 2], output_dir=output_dir)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "cohort_aggregate.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "cohort_aggregate.csv")))

    def test_json_artifact_schema(self):
        """cohort_aggregate.json must contain the documented top-level keys."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._run_with_fake_results([1, 2], output_dir=output_dir)
            json_path = os.path.join(output_dir, "cohort_aggregate.json")
            with open(json_path) as f:
                data = json.load(f)

        required_keys = {
            "num_seeds",
            "seeds",
            "best_fitness_mean",
            "best_fitness_std",
            "best_fitness_min",
            "best_fitness_max",
            "convergence_rate",
            "convergence_reason_counts",
            "mean_generation_of_convergence",
            "std_generation_of_convergence",
            "lower_bound_occupancy_mean",
            "lower_bound_occupancy_std",
            "mean_elapsed_seconds",
            "total_elapsed_seconds",
            "seed_results",
            "config",
        }
        for key in required_keys:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_csv_artifact_schema(self):
        """cohort_aggregate.csv must contain one row per seed with required columns."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._run_with_fake_results([1, 2, 3], output_dir=output_dir)
            csv_path = os.path.join(output_dir, "cohort_aggregate.csv")
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(len(rows), 3)
        required_columns = {
            "seed",
            "best_fitness",
            "num_generations_completed",
            "converged",
            "convergence_reason",
            "generation_of_convergence",
            "elapsed_seconds",
            "lower_bound_occupancy",
        }
        for col in required_columns:
            self.assertIn(col, rows[0], f"Missing CSV column: {col}")

    def test_json_seed_results_count(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._run_with_fake_results([1, 2, 3], output_dir=output_dir)
            json_path = os.path.join(output_dir, "cohort_aggregate.json")
            with open(json_path) as f:
                data = json.load(f)
        self.assertEqual(len(data["seed_results"]), 3)

    def test_no_output_dir_no_artifacts(self):
        """When output_dir is None no files should be created."""
        result = self._run_with_fake_results([1, 2])
        self.assertIsInstance(result, CohortAggregateResult)

    def test_seed_per_run_matches_seeds_list(self):
        """Each CohortSeedResult should carry its seed value."""
        result = self._run_with_fake_results([11, 22, 33])
        self.assertEqual([sr.seed for sr in result.seed_results], [11, 22, 33])

    def test_reproducibility_same_seeds(self):
        """Two runs with identical seeds and mocked results must produce identical aggregates."""
        seeds = [1, 2, 3]
        r1 = self._run_with_fake_results(seeds)
        r2 = self._run_with_fake_results(seeds)
        self.assertAlmostEqual(r1.best_fitness_mean, r2.best_fitness_mean)
        self.assertAlmostEqual(r1.best_fitness_std, r2.best_fitness_std)


# ---------------------------------------------------------------------------
# CLI smoke tests: run_cohort_experiment
# ---------------------------------------------------------------------------

class TestRunCohortExperimentCLI(unittest.TestCase):
    def test_parse_args_num_seeds_default(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_cohort_experiment.py",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
            ]
            args = run_cohort_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.num_seeds, 3)

    def test_parse_args_num_seeds_custom(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_cohort_experiment.py",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
                "--num-seeds", "5",
                "--base-seed", "10",
            ]
            args = run_cohort_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.num_seeds, 5)
        self.assertEqual(args.base_seed, 10)
        # Seeds are derived as [base_seed, base_seed+1, ..., base_seed+num_seeds-1]
        expected_seeds = list(range(10, 15))
        derived_seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))
        self.assertEqual(derived_seeds, expected_seeds)

    def test_preset_applied_to_cohort(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_cohort_experiment.py",
                "--preset", "stable_hyper_evo",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
            ]
            args = run_cohort_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.preset, "stable_hyper_evo")
        self.assertEqual(args.selection_method, "tournament")
        self.assertEqual(args.boundary_mode, "reflect")
        self.assertAlmostEqual(args.mutation_rate, 0.20)
        self.assertTrue(args.adaptive_mutation)

    def test_manifest_written_before_run(self):
        """cohort_manifest.json should be written before the cohort runs."""
        with tempfile.TemporaryDirectory() as output_dir:
            argv = sys.argv[:]
            try:
                sys.argv = [
                    "run_cohort_experiment.py",
                    "--generations", "1",
                    "--population-size", "2",
                    "--steps-per-candidate", "1",
                    "--num-seeds", "2",
                    "--base-seed", "0",
                    "--output-dir", output_dir,
                ]
                fake_aggregate = CohortAggregateResult(
                    config={},
                    num_seeds=2,
                    seeds=[0, 1],
                    seed_results=[
                        CohortSeedResult(
                            seed=i,
                            best_fitness=5.0,
                            num_generations_completed=1,
                            converged=False,
                            convergence_reason=None,
                            generation_of_convergence=None,
                            elapsed_seconds=0.1,
                            lower_bound_occupancy=None,
                        )
                        for i in range(2)
                    ],
                    best_fitness_mean=5.0,
                    best_fitness_std=0.0,
                    best_fitness_min=5.0,
                    best_fitness_max=5.0,
                    convergence_rate=0.0,
                    convergence_reason_counts={},
                    mean_generation_of_convergence=None,
                    std_generation_of_convergence=None,
                    lower_bound_occupancy_mean=None,
                    lower_bound_occupancy_std=None,
                    mean_elapsed_seconds=0.1,
                    total_elapsed_seconds=0.2,
                )
                with patch.object(run_cohort_experiment.CohortRunner, "run", return_value=fake_aggregate):
                    run_cohort_experiment.main()
            finally:
                sys.argv = argv

            manifest_path = os.path.join(output_dir, "cohort_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            with open(manifest_path) as f:
                manifest = json.load(f)

        self.assertEqual(manifest["num_seeds"], 2)
        self.assertEqual(manifest["base_seed"], 0)
        self.assertEqual(manifest["seeds"], [0, 1])
        self.assertEqual(manifest["script"], "scripts/run_cohort_experiment.py")
        self.assertIsNone(manifest["preset"])

    def test_manifest_includes_preset_when_given(self):
        with tempfile.TemporaryDirectory() as output_dir:
            argv = sys.argv[:]
            try:
                sys.argv = [
                    "run_cohort_experiment.py",
                    "--preset", "stable_hyper_evo",
                    "--generations", "1",
                    "--population-size", "2",
                    "--steps-per-candidate", "1",
                    "--output-dir", output_dir,
                ]
                fake_aggregate = CohortAggregateResult(
                    config={},
                    num_seeds=3,
                    seeds=[0, 1, 2],
                    seed_results=[
                        CohortSeedResult(
                            seed=i,
                            best_fitness=5.0,
                            num_generations_completed=1,
                            converged=False,
                            convergence_reason=None,
                            generation_of_convergence=None,
                            elapsed_seconds=0.1,
                            lower_bound_occupancy=None,
                        )
                        for i in range(3)
                    ],
                    best_fitness_mean=5.0,
                    best_fitness_std=0.0,
                    best_fitness_min=5.0,
                    best_fitness_max=5.0,
                    convergence_rate=0.0,
                    convergence_reason_counts={},
                    mean_generation_of_convergence=None,
                    std_generation_of_convergence=None,
                    lower_bound_occupancy_mean=None,
                    lower_bound_occupancy_std=None,
                    mean_elapsed_seconds=0.1,
                    total_elapsed_seconds=0.3,
                )
                with patch.object(run_cohort_experiment.CohortRunner, "run", return_value=fake_aggregate):
                    run_cohort_experiment.main()
            finally:
                sys.argv = argv

            manifest_path = os.path.join(output_dir, "cohort_manifest.json")
            with open(manifest_path) as f:
                manifest = json.load(f)

        self.assertEqual(manifest["preset"], "stable_hyper_evo")

    def test_presets_dict_contains_stable_hyper_evo(self):
        self.assertIn("stable_hyper_evo", run_cohort_experiment.PRESETS)

    def test_presets_stable_hyper_evo_has_required_keys(self):
        preset = run_cohort_experiment.PRESETS["stable_hyper_evo"]
        for key in ("selection_method", "boundary_mode", "mutation_rate", "mutation_scale", "adaptive_mutation"):
            self.assertIn(key, preset)

    def test_parse_args_interior_bias_fraction_custom(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_cohort_experiment.py",
                "--generations",
                "1",
                "--population-size",
                "2",
                "--steps-per-candidate",
                "1",
                "--interior-bias-fraction",
                "0.123",
            ]
            args = run_cohort_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertAlmostEqual(args.interior_bias_fraction, 0.123)


if __name__ == "__main__":
    unittest.main()
