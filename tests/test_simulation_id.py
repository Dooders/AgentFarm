"""Test simulation_id propagation through the simulation system.

This test validates that simulation_id is properly generated, assigned, and propagated
through all database records when running a simulation.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from typing import Dict, List, Optional

from farm.config import SimulationConfig
from farm.config.config import (
    EnvironmentConfig,
    PopulationConfig,
    ResourceConfig,
    DatabaseConfig,
)
from farm.core.simulation import run_simulation
from farm.utils.identity import Identity

# Shared Identity instance for efficiency in tests
_shared_identity = Identity()


class TestSimulationID(unittest.TestCase):
    """Test case for verifying simulation_id propagation."""

    def setUp(self):
        """Set up the test case with a temporary directory for the database."""
        self.test_dir = tempfile.mkdtemp()

        # Create a minimalist configuration
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=100, height=100),
            resources=ResourceConfig(initial_resources=100),
            population=PopulationConfig(
                system_agents=2,
                independent_agents=2,
                control_agents=0,
            ),
            database=DatabaseConfig(use_in_memory_db=False),
        )

        # Path for simulation database
        self.db_path = os.path.join(self.test_dir, "simulation")

    def tearDown(self):
        """Clean up temporary files after the test."""
        # Make sure database connections are closed properly
        try:
            import sqlite3
            import time

            # Close any open connections to the test database
            if hasattr(self, "db_conn") and self.db_conn:
                self.db_conn.close()

            if hasattr(self, "environment") and self.environment:
                self.environment.cleanup()

            # Force garbage collection to release file handles
            import gc

            gc.collect()

            # Give the OS a moment to release the file handles
            time.sleep(1)

            # Try a few times to remove the directory
            for attempt in range(3):
                try:
                    shutil.rmtree(self.test_dir)
                    break
                except (PermissionError, OSError) as e:
                    if attempt == 2:  # Last attempt
                        print(f"Warning: Could not delete test directory: {e}")
                    else:
                        time.sleep(1)  # Wait a bit and try again
        except Exception as e:
            print(f"Error during teardown: {e}")

    def test_simulation_id_generation(self):
        """Test that simulation_id is generated correctly."""
        sim_id = str(_shared_identity.simulation_id())
        self.assertTrue(sim_id.startswith("sim_"))
        self.assertTrue(len(sim_id) > 5)  # Should have some length beyond the prefix

        # Test custom prefix
        custom_id = str(_shared_identity.simulation_id(prefix="test"))
        self.assertTrue(custom_id.startswith("test_"))

    def test_simulation_id_propagation(self):
        """Test that simulation_id is propagated to all database records."""
        # Generate a unique simulation_id for this test
        simulation_id = str(_shared_identity.simulation_id(prefix="test"))

        # Run a minimal simulation with our simulation_id
        self.environment = run_simulation(
            num_steps=3,  # Just a few steps to test the concept
            config=self.config,
            path=self.db_path,
            simulation_id=simulation_id,
        )

        # Verify the environment has the correct simulation_id
        self.assertEqual(self.environment.simulation_id, simulation_id)

        # Check the database for the simulation_id in various tables
        db_file = os.path.join(self.db_path, f"simulation_{simulation_id}.db")
        self.assertTrue(
            os.path.exists(db_file), f"Database file {db_file} does not exist"
        )

        self.db_conn = sqlite3.connect(db_file)
        cursor = self.db_conn.cursor()

        # List of tables to check for simulation_id
        tables_to_check = [
            "agents",
            "agent_states",
            "resource_states",
            "simulation_steps",
            "agent_actions",
            "simulation_config",
        ]

        for table in tables_to_check:
            # Check if the table exists
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            if cursor.fetchone() is None:
                self.fail(f"Table {table} does not exist in the database")

            # Check if the table has a simulation_id column
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            self.assertIn(
                "simulation_id",
                column_names,
                f"Table {table} does not have simulation_id column",
            )

            # Count records with our simulation_id
            cursor.execute(
                f"SELECT COUNT(*) FROM {table} WHERE simulation_id = ?",
                (simulation_id,),
            )
            count = cursor.fetchone()[0]

            # The table should have at least one record with our simulation_id
            # (unless it's a table that might not have entries in this test)
            self.assertTrue(
                count > 0, f"Table {table} has no records with our simulation_id"
            )

            # All records in the table should have our simulation_id
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total = cursor.fetchone()[0]
            self.assertEqual(
                count,
                total,
                f"Not all records in {table} have our simulation_id (found {count} of {total})",
            )

        # Clean up
        cursor.close()
        self.db_conn.close()
        self.db_conn = None

    def test_simulation_id_auto_generation(self):
        """Test that simulation_id is automatically generated when not provided."""
        # Run a minimal simulation without specifying simulation_id
        environment = run_simulation(num_steps=2, config=self.config, path=self.db_path)

        # Verify the environment has a simulation_id
        self.assertIsNotNone(environment.simulation_id)
        self.assertTrue(environment.simulation_id.startswith("sim_"))

        # The simulation_id should be propagated to the database
        db_file = os.path.join(self.db_path, f"simulation_{environment.simulation_id}.db")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check just one table as an example
        cursor.execute("SELECT DISTINCT simulation_id FROM agents")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], environment.simulation_id)

        cursor.close()
        conn.close()


if __name__ == "__main__":
    unittest.main()
