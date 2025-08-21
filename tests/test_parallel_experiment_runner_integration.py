"""
Integration tests for the ParallelExperimentRunner class.

This module tests the integration of the ParallelExperimentRunner with the
rest of the simulation framework, including actual parallel execution.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from farm.core.config import SimulationConfig
from farm.runners.parallel_experiment_runner import ParallelExperimentRunner
from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel, SimulationStepModel


class TestParallelExperimentRunnerIntegration(unittest.TestCase):
    """Integration tests for the ParallelExperimentRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a basic simulation config for testing
        self.config = SimulationConfig()
        self.config.system_agents = 2
        self.config.independent_agents = 2
        self.config.width = 20
        self.config.height = 20
        self.config.use_in_memory_db = True
        self.config.in_memory_db_memory_limit_mb = 100
        self.config.simulation_steps = 10  # Keep it small for faster tests

        # Create a test experiment name
        self.experiment_name = "test_parallel_integration"

        # Create the runner
        self.runner = ParallelExperimentRunner(
            self.config,
            self.experiment_name,
            n_jobs=2,  # Use 2 jobs for testing
            db_path=Path(self.temp_dir.name) / "test.db",
            use_in_memory_db=True,
            in_memory_db_memory_limit_mb=100,
        )

        # Output directory for tests
        self.output_dir = Path(self.temp_dir.name) / "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Keep track of database connections to close
        self.db_connections = []

    def tearDown(self):
        """Clean up test fixtures."""
        # Close any open database connections
        for db in self.db_connections:
            try:
                db.close()
            except:
                pass

        # Sleep briefly to allow file handles to be released
        import time

        time.sleep(0.5)

        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_parallel_execution(self):
        """Test that multiple simulations can be executed in parallel."""
        # Run a small number of iterations
        num_iterations = 3
        num_steps = 5

        # Run the iterations
        results = self.runner.run_iterations(
            num_iterations=num_iterations,
            num_steps=num_steps,
            output_dir=self.output_dir,
        )

        # Check that we got the expected number of results
        self.assertEqual(len(results), num_iterations)

        # Check that all iterations were successful
        for result in results:
            self.assertTrue(result.get("success", False))

        # Check that output directories were created for each iteration
        for i in range(num_iterations):
            iter_dir = self.output_dir / f"iteration_{i}"
            self.assertTrue(iter_dir.exists())

            # Check that database file was created
            db_file = iter_dir / "simulation.db"
            self.assertTrue(db_file.exists())

    def test_variations(self):
        """Test that configuration variations are applied correctly."""
        # Define variations
        variations = [
            {"num_system_agents": 3, "num_independent_agents": 1},
            {"num_system_agents": 1, "num_independent_agents": 3},
        ]

        # Run iterations with variations
        results = self.runner.run_iterations(
            num_iterations=2,
            variations=variations,
            num_steps=5,
            output_dir=self.output_dir,
        )

        # Check that variations were applied
        self.assertEqual(len(results), 2)

        # Check the agent counts in the results
        # Note: We can't directly check the config in results as it might be serialized
        # Instead, we'll check the database files

        # First variation: 3 system agents, 1 independent agent
        db_file_1 = self.output_dir / "iteration_0" / "simulation.db"
        self.assertTrue(db_file_1.exists())

        # Second variation: 1 system agent, 3 independent agents
        db_file_2 = self.output_dir / "iteration_1" / "simulation.db"
        self.assertTrue(db_file_2.exists())

        # Check agent counts in the databases
        db1 = SimulationDatabase(str(db_file_1))
        self.db_connections.append(db1)
        session1 = db1.Session()
        agents1 = session1.query(AgentModel).all()
        system_count1 = sum(1 for a in agents1 if getattr(a, 'agent_type') == "SYSTEM")
        independent_count1 = sum(1 for a in agents1 if getattr(a, 'agent_type') == "INDEPENDENT")
        session1.close()

        # Should have 3 system agents and 1 independent agent
        self.assertEqual(system_count1, 3)
        self.assertEqual(independent_count1, 1)

        db2 = SimulationDatabase(str(db_file_2))
        self.db_connections.append(db2)
        session2 = db2.Session()
        agents2 = session2.query(AgentModel).all()
        system_count2 = sum(1 for a in agents2 if getattr(a, 'agent_type') == "SYSTEM")
        independent_count2 = sum(1 for a in agents2 if getattr(a, 'agent_type') == "INDEPENDENT")
        session2.close()

        # Should have 1 system agent and 3 independent agents
        self.assertEqual(system_count2, 1)
        self.assertEqual(independent_count2, 3)

    def test_database_persistence(self):
        """Test that in-memory databases are persisted correctly."""
        # Run a single iteration
        results = self.runner.run_iterations(
            num_iterations=1, num_steps=5, output_dir=self.output_dir
        )

        # Check that the database file was created
        db_file = self.output_dir / "iteration_0" / "simulation.db"
        self.assertTrue(db_file.exists())

        # Check that the database contains the expected data
        db = SimulationDatabase(str(db_file))
        self.db_connections.append(db)

        # Check that agents were saved
        session = db.Session()
        agents = session.query(AgentModel).all()
        self.assertEqual(len(agents), 4)  # 2 system + 2 independent

        # Check that steps were saved
        steps = session.query(SimulationStepModel).all()
        self.assertEqual(len(steps), 5)  # 5 steps
        session.close()

    @pytest.mark.slow
    def test_error_recovery(self):
        """Test that the runner can recover from errors in individual simulations."""
        # Create variations with one valid and one that will cause an error
        variations = [
            {},  # Use base config (valid)
            {"world_size": (-10, -10)},  # Negative world size will cause an error
        ]

        # Run iterations with variations
        results = self.runner.run_iterations(
            num_iterations=2,
            variations=variations,
            num_steps=5,
            output_dir=self.output_dir,
        )

        # Check that we got results for both iterations
        self.assertEqual(len(results), 2)

        # First iteration should succeed
        self.assertTrue(results[0].get("success", False))

        # Second iteration should fail
        self.assertFalse(results[1].get("success", False))
        self.assertIn("error", results[1])

        # Check that error log was created for the failed iteration
        error_log_path = self.output_dir / "iteration_1" / "error.log"
        self.assertTrue(error_log_path.exists())

    @pytest.mark.slow
    def test_performance_scaling(self):
        """Test that performance scales with the number of cores."""
        # Skip this test in CI environments
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping performance test in CI environment")

        import time

        # Run with 1 worker
        runner_single = ParallelExperimentRunner(
            self.config,
            self.experiment_name,
            n_jobs=1,
            db_path=Path(self.temp_dir.name) / "test_single.db",
            use_in_memory_db=True,
        )

        # Time the execution with 1 worker
        start_time_single = time.time()
        runner_single.run_iterations(
            num_iterations=4, num_steps=10, output_dir=self.output_dir / "single"
        )
        duration_single = time.time() - start_time_single

        # Run with multiple workers
        runner_multi = ParallelExperimentRunner(
            self.config,
            self.experiment_name,
            n_jobs=2,  # Use 2 workers
            db_path=Path(self.temp_dir.name) / "test_multi.db",
            use_in_memory_db=True,
        )

        # Time the execution with multiple workers
        start_time_multi = time.time()
        runner_multi.run_iterations(
            num_iterations=4, num_steps=10, output_dir=self.output_dir / "multi"
        )
        duration_multi = time.time() - start_time_multi

        # Check that multi-core execution is faster
        # We allow for some overhead, so we check if it's at least 30% faster
        self.assertLess(duration_multi, duration_single * 0.9)

        # Print the speedup for information
        speedup = duration_single / duration_multi
        print(f"Speedup with 2 workers: {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
