"""Tests for farm/runners/batch_runner.py.

BatchRunner.run() spawns real processes, so we mock the heavy parts
(Pool.map, run_simulation) to keep these tests fast.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from farm.config import SimulationConfig
from farm.runners.batch_runner import BatchRunner, run_simulation_wrapper


def _make_config():
    """Return a minimal default SimulationConfig."""
    return SimulationConfig()


def _make_mock_env():
    """Return a minimal mock environment."""
    alive_agent = Mock()
    alive_agent.alive = True
    alive_agent.resource_level = 10.0
    dead_agent = Mock()
    dead_agent.alive = False
    dead_agent.resource_level = 0.0
    mock_env = Mock()
    mock_env.agent_objects = [alive_agent, dead_agent]
    mock_env.alive_agent_objects = [alive_agent]
    mock_env.resources = [Mock(amount=50.0)]
    return mock_env


class TestBatchRunnerInit(unittest.TestCase):
    """Tests for BatchRunner initialisation."""

    def test_init_stores_config(self):
        config = _make_config()
        runner = BatchRunner(base_config=config)
        self.assertIs(runner.base_config, config)
        self.assertEqual(runner.results, [])
        self.assertEqual(runner.parameter_variations, {})

    def test_add_parameter_variation(self):
        config = _make_config()
        runner = BatchRunner(base_config=config)
        runner.add_parameter_variation("max_steps", [100, 200])
        runner.add_parameter_variation("seed", [42, 99])
        self.assertIn("max_steps", runner.parameter_variations)
        self.assertIn("seed", runner.parameter_variations)

    def test_create_config_variation_sets_param(self):
        config = _make_config()
        runner = BatchRunner(base_config=config)
        new_cfg = runner._create_config_variation({"max_steps": 999})
        self.assertEqual(new_cfg.max_steps, 999)

    def test_collect_results(self):
        config = _make_config()
        runner = BatchRunner(base_config=config)
        runner._collect_results({"max_steps": 100}, _make_mock_env())
        self.assertEqual(len(runner.results), 1)
        self.assertIn("max_steps", runner.results[0])
        self.assertEqual(runner.results[0]["final_agents"], 1)


class TestBatchRunnerRun(unittest.TestCase):
    """Tests for BatchRunner.run() with mocked subprocess."""

    @patch("farm.runners.batch_runner.Pool")
    @patch("farm.runners.batch_runner.os.makedirs")
    @patch("farm.runners.batch_runner.configure_logging")
    def test_run_creates_result_file(self, mock_log, mock_makedirs, MockPool):
        """run() should call pool.map and save results CSV."""
        config = _make_config()
        runner = BatchRunner(base_config=config)
        runner.add_parameter_variation("max_steps", [10])

        pool_instance = MockPool.return_value.__enter__.return_value
        pool_instance.map.return_value = [_make_mock_env()]

        with patch.object(runner, "_save_results") as mock_save:
            runner.run(experiment_name="test_batch", num_steps=2)
            mock_save.assert_called_once()

    @patch("farm.runners.batch_runner.Pool")
    @patch("farm.runners.batch_runner.os.makedirs")
    @patch("farm.runners.batch_runner.configure_logging")
    def test_run_no_variations_runs_empty(self, mock_log, mock_makedirs, MockPool):
        """run() with no parameter variations should map over empty list."""
        config = _make_config()
        runner = BatchRunner(base_config=config)

        pool_instance = MockPool.return_value.__enter__.return_value
        pool_instance.map.return_value = []

        with patch.object(runner, "_save_results"):
            runner.run(experiment_name="empty_batch", num_steps=2)

        pool_instance.map.assert_called_once()


class TestRunSimulationWrapper(unittest.TestCase):
    """Tests for the module-level run_simulation_wrapper function."""

    @patch("farm.runners.batch_runner.run_simulation")
    def test_success_returns_environment(self, mock_run):
        mock_env = Mock()
        mock_run.return_value = mock_env

        result = run_simulation_wrapper(({}, None, 10, "test.db"))
        self.assertIs(result, mock_env)

    @patch("farm.runners.batch_runner.run_simulation")
    def test_failure_returns_none(self, mock_run):
        mock_run.side_effect = RuntimeError("sim failed")
        result = run_simulation_wrapper(({}, None, 10, "test.db"))
        self.assertIsNone(result)

    @patch("farm.runners.batch_runner.run_simulation")
    def test_config_file_path_loads_yaml(self, mock_run):
        mock_env = Mock()
        mock_run.return_value = mock_env

        with patch("farm.runners.batch_runner.SimulationConfig") as MockCfg:
            instance = Mock()
            MockCfg.from_yaml.return_value = instance
            run_simulation_wrapper(({"max_steps": 10}, "config.yaml", 5, "test.db"))
            MockCfg.from_yaml.assert_called_once_with("config.yaml")


if __name__ == "__main__":
    unittest.main()
