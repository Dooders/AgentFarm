"""
Unit tests for the ParallelExperimentRunner class.

This module tests the functionality of the ParallelExperimentRunner class,
which provides parallel execution of multiple simulations.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from farm.core.config import SimulationConfig
from farm.runners.parallel_experiment_runner import ParallelExperimentRunner


class TestParallelExperimentRunner(unittest.TestCase):
    """Test cases for the ParallelExperimentRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a basic simulation config for testing
        self.config = SimulationConfig()
        self.config.system_agents = 5
        self.config.independent_agents = 5
        self.config.width = 50
        self.config.height = 50
        self.config.use_in_memory_db = True
        self.config.in_memory_db_memory_limit_mb = 100

        # Create a test experiment name
        self.experiment_name = "test_parallel_experiment"

        # Create the runner
        self.runner = ParallelExperimentRunner(
            self.config,
            self.experiment_name,
            n_jobs=2,  # Use 2 jobs for testing
            db_path=Path(self.temp_dir.name) / "test.db",
            use_in_memory_db=True,
            in_memory_db_memory_limit_mb=100,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the runner initializes correctly."""
        # Check that the runner has the correct attributes
        self.assertEqual(self.runner.base_config, self.config)
        self.assertEqual(self.runner.experiment_name, self.experiment_name)
        self.assertEqual(self.runner.n_jobs, 2)
        self.assertEqual(self.runner.db_path, Path(self.temp_dir.name) / "test.db")
        self.assertTrue(self.runner.use_in_memory_db)
        self.assertEqual(self.runner.in_memory_db_memory_limit_mb, 100)

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_configure_process_pool(self, mock_virtual_memory, mock_cpu_count):
        """Test that the process pool is configured correctly."""
        # Mock system resources
        mock_cpu_count.side_effect = lambda logical: 8 if logical else 4
        mock_memory = MagicMock()
        mock_memory.available = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory

        # Test with default n_jobs (-1)
        self.runner.n_jobs = -1
        pool_config = self.runner._configure_process_pool()

        # Should use all available cores
        self.assertEqual(pool_config["n_jobs"], 8)

        # Test with specific n_jobs
        self.runner.n_jobs = 3
        pool_config = self.runner._configure_process_pool()
        self.assertEqual(pool_config["n_jobs"], 3)

        # Test with n_jobs > available cores
        self.runner.n_jobs = 12
        pool_config = self.runner._configure_process_pool()
        self.assertEqual(pool_config["n_jobs"], 8)

        # Test with limited memory
        mock_memory.available = 3 * 1024 * 1024 * 1024  # 3GB
        self.runner.n_jobs = -1
        pool_config = self.runner._configure_process_pool()
        # Should be limited by memory (3GB / 2GB per worker = 1 worker)
        self.assertEqual(pool_config["n_jobs"], 1)

    def test_error_handling(self):
        """Test that errors in simulations are handled correctly."""
        # Create a mock for run_simulation that raises an exception
        with patch("farm.core.parallel_experiment_runner.run_simulation") as mock_run:
            mock_run.side_effect = Exception("Test error")

            # Run a single simulation with error handling
            result = self.runner._run_with_error_handling(
                self.config, 10, Path(self.temp_dir.name), 42
            )

            # Check that the error was captured
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Test error")
            self.assertIn("traceback", result)
            self.assertFalse(result["success"])

            # Check that an error log was created
            error_log_path = Path(self.temp_dir.name) / "error.log"
            self.assertTrue(error_log_path.exists())

            # Check the content of the error log
            with open(error_log_path, "r") as f:
                error_log = f.read()
                self.assertIn("Test error", error_log)

    def test_save_summary(self):
        """Test that the summary is saved correctly."""
        # Create some test results
        results = [
            {"success": True, "final_agent_count": 10},
            {"success": True, "final_agent_count": 15},
            {"success": False, "error": "Test error"},
        ]

        # Save the summary
        output_dir = Path(self.temp_dir.name)
        self.runner._save_summary(results, output_dir)

        # Check that the summary file was created
        summary_path = output_dir / "experiment_summary.json"
        self.assertTrue(summary_path.exists())

        # Check the content of the summary
        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)
            self.assertEqual(summary["experiment_name"], self.experiment_name)
            self.assertEqual(summary["num_iterations"], 3)
            self.assertEqual(summary["successful_runs"], 2)
            self.assertEqual(summary["failed_runs"], 1)
            self.assertEqual(len(summary["results"]), 3)

    @patch("farm.core.parallel_experiment_runner.Parallel")
    def test_run_iterations(self, mock_parallel):
        """Test that iterations are run correctly."""
        # Mock the Parallel execution
        mock_results = [
            {"success": True, "final_agent_count": 10},
            {"success": True, "final_agent_count": 15},
        ]

        # Create a mock function that returns our mock results
        mock_parallel.return_value = lambda *args, **kwargs: mock_results

        # Run iterations
        results = self.runner.run_iterations(
            num_iterations=2, num_steps=10, output_dir=self.temp_dir.name
        )

        # Check that Parallel was called with the correct arguments
        mock_parallel.assert_called_once()
        args, kwargs = mock_parallel.call_args
        self.assertIn("n_jobs", kwargs)

        # Check that the results were returned correctly
        self.assertEqual(results, mock_results)

        # Check that the summary was saved
        summary_path = Path(self.temp_dir.name) / "experiment_summary.json"
        self.assertTrue(summary_path.exists())

    def test_cleanup_resources(self):
        """Test that resources are cleaned up correctly."""
        # This is mostly a coverage test since _cleanup_resources doesn't return anything
        with patch("gc.collect") as mock_gc:
            self.runner._cleanup_resources()
            mock_gc.assert_called_once()

    def test_log_resource_usage(self):
        """Test that resource usage is logged correctly."""
        with patch("psutil.cpu_percent", return_value=50.0), patch(
            "psutil.virtual_memory"
        ) as mock_memory, patch(
            "farm.core.parallel_experiment_runner.logging.Logger.info"
        ) as mock_info, patch(
            "farm.core.parallel_experiment_runner.logging.Logger.warning"
        ) as mock_warning:

            # Test normal resource usage
            mock_memory_obj = MagicMock()
            mock_memory_obj.percent = 70.0
            mock_memory.return_value = mock_memory_obj

            self.runner._log_resource_usage()
            mock_info.assert_called_once()
            mock_warning.assert_not_called()

            # Test high memory usage
            mock_info.reset_mock()
            mock_memory_obj.percent = 95.0

            self.runner._log_resource_usage()
            mock_info.assert_called_once()
            mock_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
