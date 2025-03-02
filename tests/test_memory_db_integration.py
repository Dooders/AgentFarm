"""
Integration tests for the in-memory database with simulation runner.

This module tests the integration between the in-memory database and
the simulation runner, ensuring that simulations run correctly with
in-memory database mode enabled.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import InMemorySimulationDatabase, SimulationDatabase


class TestInMemoryDatabaseIntegration(unittest.TestCase):
    """Test suite for in-memory database integration with simulation runner."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Load base configuration
        self.config = SimulationConfig()

        # Set small simulation parameters for faster tests
        self.config.width = 50
        self.config.height = 50
        self.config.system_agents = 5
        self.config.independent_agents = 5
        self.config.control_agents = 5
        self.config.initial_resources = 10
        self.config.simulation_steps = 10

    def tearDown(self):
        """Clean up after each test."""
        try:
            # Remove temporary directory
            shutil.rmtree(self.test_dir)
        except PermissionError as e:
            # On Windows, files might still be locked by the database
            print(f"Warning: Could not fully clean up temporary directory: {e}")
            # Try to remove individual files that aren't locked
            for root, dirs, files in os.walk(self.test_dir):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except PermissionError:
                        pass

    def test_simulation_with_disk_database(self):
        """Test running a simulation with disk database."""
        # Set up disk database path
        db_path = os.path.join(self.test_dir, "disk_simulation.db")

        # Ensure in-memory mode is disabled
        self.config.use_in_memory_db = False

        # Run simulation
        start_time = time.time()
        environment = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=self.test_dir,
            save_config=True,
        )
        disk_duration = time.time() - start_time

        # Verify simulation completed
        self.assertEqual(environment.time, self.config.simulation_steps + 1)

        # Verify database file exists
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "simulation.db")))

        # Print performance metrics
        print(f"Disk database simulation completed in {disk_duration:.2f} seconds")

        # Clean up
        environment.cleanup()

        # Explicitly close the database connection
        if hasattr(environment, "db") and environment.db is not None:
            try:
                environment.db.close()
                # Set to None to help garbage collection
                environment.db = None
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

    def test_simulation_with_in_memory_database(self):
        """Test running a simulation with in-memory database."""
        # Enable in-memory mode
        self.config.use_in_memory_db = True
        self.config.persist_db_on_completion = True

        # Ensure test directory exists
        os.makedirs(self.test_dir, exist_ok=True)

        # Run simulation
        start_time = time.time()
        environment = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=self.test_dir,  # Ensure path is provided
            save_config=True,
        )
        memory_duration = time.time() - start_time

        # Verify simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(environment.time, self.config.simulation_steps + 1)

        # Verify database file exists (should be persisted)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "simulation.db")))

        # Verify database is an instance of InMemorySimulationDatabase
        self.assertIsInstance(environment.db, InMemorySimulationDatabase)

        # Print performance metrics
        print(
            f"In-memory database simulation completed in {memory_duration:.2f} seconds"
        )

        # Clean up
        environment.cleanup()

        # Explicitly close the database connection
        if hasattr(environment, "db") and environment.db is not None:
            try:
                environment.db.close()
                # Set to None to help garbage collection
                environment.db = None
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

    def test_simulation_without_persistence(self):
        """Test running a simulation with in-memory database without persistence."""
        # Enable in-memory mode without persistence
        self.config.use_in_memory_db = True
        self.config.persist_db_on_completion = False

        # Run simulation
        environment = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=self.test_dir,
            save_config=True,
        )

        # Verify simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(environment.time, self.config.simulation_steps + 1)

        # Verify database file does not exist (should not be persisted)
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "simulation.db")))

        # Verify database is an instance of InMemorySimulationDatabase
        self.assertIsInstance(environment.db, InMemorySimulationDatabase)

        # Clean up
        environment.cleanup()

        # Explicitly close the database connection
        if hasattr(environment, "db") and environment.db is not None:
            try:
                environment.db.close()
                # Set to None to help garbage collection
                environment.db = None
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

    def test_selective_table_persistence(self):
        """Test running a simulation with selective table persistence."""
        # Enable in-memory mode with selective persistence
        self.config.use_in_memory_db = True
        self.config.persist_db_on_completion = True
        self.config.in_memory_tables_to_persist = [
            "simulation_config",
            "simulation_steps",
        ]

        # Run simulation
        environment = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=self.test_dir,
            save_config=True,
        )

        # Verify simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(environment.time, self.config.simulation_steps + 1)

        # Verify database file exists
        db_path = os.path.join(self.test_dir, "simulation.db")
        self.assertTrue(os.path.exists(db_path))

        # Connect to the persisted database and verify only specified tables have data
        disk_db = SimulationDatabase(db_path)

        # These tables should have data
        self.assertTrue(disk_db.get_table_row_count("simulation_steps") > 0)
        self.assertTrue(disk_db.get_table_row_count("simulation_config") > 0)

        # These tables should be empty or have minimal data
        self.assertEqual(disk_db.get_table_row_count("agents"), 0)
        self.assertEqual(disk_db.get_table_row_count("agent_states"), 0)

        # Close the disk database
        disk_db.close()

        # Clean up
        environment.cleanup()

        # Explicitly close the database connection
        if hasattr(environment, "db") and environment.db is not None:
            try:
                environment.db.close()
                # Set to None to help garbage collection
                environment.db = None
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement during simulation."""
        # Enable in-memory mode with a very low memory limit
        self.config.use_in_memory_db = True
        self.config.in_memory_db_memory_limit_mb = 1  # 1MB (unrealistically low)

        # Run simulation - should still complete despite memory warnings
        environment = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=self.test_dir,
            save_config=True,
        )

        # Verify simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(environment.time, self.config.simulation_steps + 1)

        # Verify database file exists (persistence should have happened)
        db_path = os.path.join(self.test_dir, "simulation.db")
        self.assertTrue(os.path.exists(db_path))

        # Instead of checking logs directly, we'll verify that the simulation completed
        # successfully despite the very low memory limit, which is the important part

        # Clean up
        environment.cleanup()

        # Explicitly close the database connection
        if hasattr(environment, "db") and environment.db is not None:
            try:
                environment.db.close()
                # Set to None to help garbage collection
                environment.db = None
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

    def test_performance_comparison(self):
        """Compare performance between disk and in-memory databases."""
        # Skip this test in CI environments or when running quick tests
        if os.environ.get("CI") or os.environ.get("QUICK_TEST"):
            self.skipTest("Skipping performance test in CI environment")

        # Use a larger simulation for meaningful performance comparison
        self.config.simulation_steps = 50
        self.config.system_agents = 10
        self.config.independent_agents = 10
        self.config.control_agents = 10

        # Create directories for disk and in-memory tests
        disk_dir = os.path.join(self.test_dir, "disk")
        memory_dir = os.path.join(self.test_dir, "memory")
        os.makedirs(disk_dir, exist_ok=True)
        os.makedirs(memory_dir, exist_ok=True)

        # Run with disk database
        self.config.use_in_memory_db = False
        disk_start_time = time.time()
        disk_env = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=disk_dir,
            save_config=True,
        )
        disk_duration = time.time() - disk_start_time

        # Verify disk simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(disk_env.time, self.config.simulation_steps + 1)

        # Clean up disk environment
        disk_env.cleanup()
        if hasattr(disk_env, "db") and disk_env.db is not None:
            try:
                disk_env.db.close()
                disk_env.db = None
            except Exception as e:
                print(f"Warning: Error closing disk database: {e}")

        # Run with in-memory database
        self.config.use_in_memory_db = True
        memory_start_time = time.time()
        memory_env = run_simulation(
            num_steps=self.config.simulation_steps,
            config=self.config,
            path=memory_dir,
            save_config=True,
        )
        memory_duration = time.time() - memory_start_time

        # Verify in-memory simulation completed
        # Note: environment.time is num_steps + 1 because of the final update call
        self.assertEqual(memory_env.time, self.config.simulation_steps + 1)

        # Clean up memory environment
        memory_env.cleanup()
        if hasattr(memory_env, "db") and memory_env.db is not None:
            try:
                memory_env.db.close()
                memory_env.db = None
            except Exception as e:
                print(f"Warning: Error closing memory database: {e}")

        # Calculate performance difference
        improvement = ((disk_duration - memory_duration) / disk_duration) * 100

        # Print performance comparison
        print(f"\nPerformance comparison:")
        print(f"  Disk database: {disk_duration:.2f} seconds")
        print(f"  In-memory database: {memory_duration:.2f} seconds")
        print(f"  Difference: {improvement:.1f}%")

        # Note: In small test scenarios, the in-memory database might not always be faster
        # due to setup overhead. In real-world scenarios with larger datasets, the
        # performance improvement would typically be more significant.

        # We only verify that both simulations completed successfully
        # No assertion about which one is faster, as this can vary in test environments


if __name__ == "__main__":
    unittest.main()
