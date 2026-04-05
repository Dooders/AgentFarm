"""Tests for farm/runners/experiment_runner.py.

Mocks the heavy simulation execution (run_simulation) to keep tests fast.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from farm.config import SimulationConfig
from farm.runners.experiment_runner import ExperimentRunner


def _make_config():
    """Return a minimal default SimulationConfig."""
    return SimulationConfig()


def _make_mock_env():
    """Return a minimal mock environment returned by run_simulation."""
    mock_env = Mock()
    mock_env.db = Mock()
    mock_env.db.logger = Mock()
    mock_env.db.logger.flush_all_buffers = Mock()
    mock_env.agents = [Mock(), Mock()]
    return mock_env


class TestExperimentRunnerInit(unittest.TestCase):
    """Tests for ExperimentRunner initialisation."""

    def test_init_creates_directories(self):
        with patch("farm.runners.experiment_runner.os.makedirs") as mock_mkdirs:
            runner = ExperimentRunner(
                base_config=_make_config(),
                experiment_name="test_exp",
            )
            self.assertEqual(runner.experiment_name, "test_exp")
            mock_mkdirs.assert_called()

    def test_init_stores_base_config(self):
        config = _make_config()
        with patch("farm.runners.experiment_runner.os.makedirs"):
            runner = ExperimentRunner(base_config=config, experiment_name="exp_x")
            self.assertIs(runner.base_config, config)

    def test_init_default_results_empty(self):
        with patch("farm.runners.experiment_runner.os.makedirs"):
            runner = ExperimentRunner(base_config=_make_config(), experiment_name="exp_y")
            self.assertEqual(runner.results, [])


class TestExperimentRunnerCreateIterationConfig(unittest.TestCase):
    """Tests for _create_iteration_config."""

    def setUp(self):
        self.config = _make_config()
        with patch("farm.runners.experiment_runner.os.makedirs"):
            self.runner = ExperimentRunner(
                base_config=self.config, experiment_name="iter_cfg_test"
            )

    def test_no_variations_returns_copy(self):
        cfg = self.runner._create_iteration_config(0, variations=None)
        self.assertIsNotNone(cfg)

    def test_variation_applied(self):
        variations = [{"max_steps": 999}]
        cfg = self.runner._create_iteration_config(0, variations=variations)
        self.assertEqual(cfg.max_steps, 999)

    def test_variation_beyond_list_uses_base(self):
        variations = [{"max_steps": 999}]
        cfg = self.runner._create_iteration_config(5, variations=variations)
        # iteration 5 has no variation – falls back to base config value
        self.assertEqual(cfg.max_steps, self.config.max_steps)


class TestExperimentRunnerRunIterations(unittest.TestCase):
    """Tests for run_iterations with mocked run_simulation."""

    def setUp(self):
        self.config = _make_config()
        with patch("farm.runners.experiment_runner.os.makedirs"):
            self.runner = ExperimentRunner(
                base_config=self.config, experiment_name="run_iter_test"
            )

    @patch("farm.runners.experiment_runner.run_simulation")
    def test_run_single_iteration(self, mock_run):
        mock_run.return_value = _make_mock_env()
        with patch("farm.runners.experiment_runner.os.makedirs"):
            self.runner.run_iterations(num_iterations=1, run_analysis=False)
        mock_run.assert_called_once()

    @patch("farm.runners.experiment_runner.run_simulation")
    def test_run_multiple_iterations_calls_run_n_times(self, mock_run):
        mock_run.return_value = _make_mock_env()
        with patch("farm.runners.experiment_runner.os.makedirs"):
            self.runner.run_iterations(num_iterations=3, run_analysis=False)
        self.assertEqual(mock_run.call_count, 3)

    @patch("farm.runners.experiment_runner.run_simulation")
    def test_run_with_path(self, mock_run):
        mock_run.return_value = _make_mock_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "experiment"
            with patch("farm.runners.experiment_runner.os.makedirs"):
                self.runner.run_iterations(
                    num_iterations=1,
                    path=out_path,
                    run_analysis=False,
                )
        mock_run.assert_called_once()

    @patch("farm.runners.experiment_runner.run_simulation")
    def test_failed_iteration_continues(self, mock_run):
        """A failing iteration should be caught and the loop should continue."""
        mock_run.side_effect = [RuntimeError("oops"), _make_mock_env()]
        with patch("farm.runners.experiment_runner.os.makedirs"):
            # Should not raise even though first iteration fails
            self.runner.run_iterations(num_iterations=2, run_analysis=False)

    @patch("farm.runners.experiment_runner.run_simulation")
    def test_config_variations_applied(self, mock_run):
        """Verify config variation is passed to each simulation call."""
        mock_run.return_value = _make_mock_env()
        variations = [{"max_steps": 150}, {"max_steps": 250}]
        with patch("farm.runners.experiment_runner.os.makedirs"):
            self.runner.run_iterations(
                num_iterations=2,
                config_variations=variations,
                run_analysis=False,
            )
        self.assertEqual(mock_run.call_count, 2)
        # First call config should have max_steps=150
        first_call_config = (
            mock_run.call_args_list[0][1].get("config")
            or mock_run.call_args_list[0][0][1]
        )
        self.assertEqual(first_call_config.max_steps, 150)


if __name__ == "__main__":
    unittest.main()
