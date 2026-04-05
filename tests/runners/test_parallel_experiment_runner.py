"""Tests for farm/runners/parallel_experiment_runner.py.

Mocks joblib.Parallel and run_simulation to avoid spawning real processes.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from farm.config import SimulationConfig
from farm.runners.parallel_experiment_runner import ParallelExperimentRunner


def _make_config():
    """Return a minimal default SimulationConfig."""
    return SimulationConfig()


def _success_result(i=0):
    return {
        "final_agent_count": 2,
        "output_path": f"/tmp/iter_{i}",
        "success": True,
        "config": {},
    }


class TestParallelExperimentRunnerInit(unittest.TestCase):
    """Tests for ParallelExperimentRunner initialisation."""

    def test_init_stores_config(self):
        config = _make_config()
        runner = ParallelExperimentRunner(
            base_config=config, experiment_name="par_test"
        )
        self.assertIs(runner.base_config, config)
        self.assertEqual(runner.experiment_name, "par_test")

    def test_init_default_n_jobs(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="par_test"
        )
        self.assertEqual(runner.n_jobs, -1)

    def test_init_custom_n_jobs(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="par_test", n_jobs=2
        )
        self.assertEqual(runner.n_jobs, 2)

    def test_init_use_in_memory_db(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(),
            experiment_name="par_test",
            use_in_memory_db=False,
        )
        self.assertFalse(runner.use_in_memory_db)


class TestParallelExperimentRunnerRunSingleSimulation(unittest.TestCase):
    """Tests for run_single_simulation."""

    def setUp(self):
        self.runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="single_sim_test"
        )

    @patch("farm.runners.parallel_experiment_runner.run_simulation")
    def test_run_single_simulation_success(self, mock_run):
        mock_env = Mock()
        mock_env.agents = [Mock(), Mock()]
        mock_env.db = None
        mock_run.return_value = mock_env

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.run_single_simulation(
                config=_make_config(),
                num_steps=5,
                output_path=Path(tmpdir) / "sim",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["final_agent_count"], 2)

    @patch("farm.runners.parallel_experiment_runner.run_simulation")
    def test_run_single_simulation_error_returns_failure(self, mock_run):
        mock_run.side_effect = RuntimeError("crash")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.run_single_simulation(
                config=_make_config(),
                num_steps=5,
                output_path=Path(tmpdir) / "sim",
            )
        self.assertFalse(result["success"])
        self.assertIn("error", result)


class TestParallelExperimentRunnerConfigurePool(unittest.TestCase):
    """Tests for _configure_process_pool."""

    def test_configure_pool_returns_dict_with_n_jobs(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="pool_test", n_jobs=2
        )
        pool_cfg = runner._configure_process_pool()
        self.assertIn("n_jobs", pool_cfg)
        self.assertGreaterEqual(pool_cfg["n_jobs"], 1)

    def test_configure_pool_n_jobs_minus_one_uses_all_cores(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="all_cores_test", n_jobs=-1
        )
        pool_cfg = runner._configure_process_pool()
        self.assertGreaterEqual(pool_cfg["n_jobs"], 1)


class TestParallelExperimentRunnerSaveSummary(unittest.TestCase):
    """Tests for _save_summary."""

    def test_save_summary_creates_file(self):
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="summary_test"
        )
        results = [_success_result(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner._save_summary(results, output_dir)
            summary_file = output_dir / "experiment_summary.json"
            self.assertTrue(summary_file.exists())

    def test_save_summary_content(self):
        import json
        runner = ParallelExperimentRunner(
            base_config=_make_config(), experiment_name="content_test"
        )
        results = [_success_result(0), {"error": "oops", "success": False}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner._save_summary(results, output_dir)
            with open(output_dir / "experiment_summary.json") as f:
                data = json.load(f)
            self.assertEqual(data["successful_runs"], 1)
            self.assertEqual(data["failed_runs"], 1)
            self.assertEqual(data["num_iterations"], 2)


class TestParallelExperimentRunnerRunIterations(unittest.TestCase):
    """Tests for run_iterations using mocked Parallel execution."""

    def setUp(self):
        self.runner = ParallelExperimentRunner(
            base_config=_make_config(),
            experiment_name="par_iter_test",
            n_jobs=1,
        )

    @patch("farm.runners.parallel_experiment_runner.Parallel")
    @patch("farm.runners.parallel_experiment_runner.delayed")
    def test_run_iterations_returns_results(self, mock_delayed, MockParallel):
        results = [_success_result(i) for i in range(2)]
        MockParallel.return_value = Mock(return_value=results)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = self.runner.run_iterations(
                num_iterations=2,
                num_steps=5,
                output_dir=tmpdir,
            )

        self.assertEqual(len(output), 2)
        for r in output:
            self.assertTrue(r["success"])

    @patch("farm.runners.parallel_experiment_runner.Parallel")
    @patch("farm.runners.parallel_experiment_runner.delayed")
    def test_run_iterations_with_variations(self, mock_delayed, MockParallel):
        results = [_success_result(0)]
        MockParallel.return_value = Mock(return_value=results)
        variations = [{"max_steps": 200}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = self.runner.run_iterations(
                num_iterations=1,
                variations=variations,
                num_steps=5,
                output_dir=tmpdir,
            )

        self.assertEqual(len(output), 1)

    @patch("farm.runners.parallel_experiment_runner.Parallel")
    @patch("farm.runners.parallel_experiment_runner.delayed")
    def test_run_iterations_handles_global_error(self, mock_delayed, MockParallel):
        MockParallel.return_value = Mock(side_effect=RuntimeError("parallel crash"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = self.runner.run_iterations(
                num_iterations=2,
                num_steps=5,
                output_dir=tmpdir,
            )

        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].get("global_error", False))


if __name__ == "__main__":
    unittest.main()
