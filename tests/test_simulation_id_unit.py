"""Unit tests for simulation_id propagation through key components.

These tests validate that simulation_id is properly passed between key components
without running the entire simulation infrastructure.
"""

import os
import sqlite3
import tempfile
import unittest

from farm.database.data_logging import DataLogger
from farm.database.database import SimulationDatabase
from farm.database.models import Simulation
from farm.utils.identity import Identity

# Shared Identity instance for efficiency in tests
_shared_identity = Identity()


class TestSimulationIDPropagation(unittest.TestCase):
    """Test case for verifying simulation_id propagation between components."""

    def setUp(self):
        """Set up temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.simulation_id = str(_shared_identity.simulation_id(prefix="unit_test"))
        self.db = None

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, "db") and self.db is not None:
            # Make sure to flush all buffers and close all connections
            if hasattr(self.db, "logger") and self.db.logger is not None:
                self.db.logger.flush_all_buffers()
            self.db.close()
            self.db = None

        # Force garbage collection to release file handles
        import gc

        gc.collect()

        # Wait a moment for connections to be fully closed
        import time

        time.sleep(0.1)

        # Now try to remove the database file
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception as e:
            print(f"Warning: Could not remove database file: {e}")

        try:
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory: {e}")

    def test_database_initialization_with_simulation_id(self):
        """Test that SimulationDatabase correctly stores simulation_id."""
        # Initialize database with simulation_id
        db = SimulationDatabase(self.db_path, simulation_id=self.simulation_id)

        # Verify the simulation_id is stored correctly
        self.assertEqual(db.simulation_id, self.simulation_id)

        # Clean up
        db.close()

    def test_data_logger_initialization_with_simulation_id(self):
        """Test that DataLogger correctly stores simulation_id."""
        # Initialize database and data logger with simulation_id
        db = SimulationDatabase(self.db_path, simulation_id=self.simulation_id)
        logger = DataLogger(db, simulation_id=self.simulation_id)

        # Verify the simulation_id is stored correctly
        self.assertEqual(logger.simulation_id, self.simulation_id)

        # Clean up
        db.close()

    def test_database_passes_simulation_id_to_logger(self):
        """Test that SimulationDatabase passes simulation_id to DataLogger."""
        # Initialize database with simulation_id
        db = SimulationDatabase(self.db_path, simulation_id=self.simulation_id)

        # Verify that the logger has the correct simulation_id
        self.assertEqual(db.logger.simulation_id, self.simulation_id)

        # Clean up
        db.close()

    def test_simulation_id_in_agent_logging(self):
        """Test that simulation_id is included when logging an agent."""
        # Initialize database with simulation_id
        self.db = SimulationDatabase(self.db_path, simulation_id=self.simulation_id)

        # Create the simulation record first to satisfy the foreign key constraint
        def create_simulation_record(session):
            simulation = Simulation(
                simulation_id=self.simulation_id,
                parameters={"test": True},
                simulation_db_path=self.db_path,
            )
            session.add(simulation)

        # Execute in transaction
        self.db._execute_in_transaction(create_simulation_record)

        # Log a simple agent
        agent_id = "test_agent_1"
        self.db.logger.log_agent(
            agent_id=agent_id,
            birth_time=0,
            agent_type="test",
            position=(10.0, 20.0),
            initial_resources=100.0,
            starting_health=100.0,
            starvation_threshold=10,
        )

        # Flush data to database
        self.db.logger.flush_all_buffers()

        # Check if agent was stored with correct simulation_id
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT simulation_id FROM agents WHERE agent_id = ?", (agent_id,)
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Agent was not stored in database")
        self.assertEqual(
            result[0],
            self.simulation_id,
            "Agent record doesn't have the correct simulation_id",
        )

        # Clean up
        cursor.close()
        conn.close()

    def test_simulation_id_in_action_logging(self):
        """Test that simulation_id is included when logging an action."""
        # Initialize database with simulation_id
        self.db = SimulationDatabase(self.db_path, simulation_id=self.simulation_id)

        # Create the simulation record first to satisfy the foreign key constraint
        def create_simulation_record(session):
            simulation = Simulation(
                simulation_id=self.simulation_id,
                parameters={"test": True},
                simulation_db_path=self.db_path,
            )
            session.add(simulation)

        # Execute in transaction
        self.db._execute_in_transaction(create_simulation_record)

        # Log a simple agent first
        agent_id = "test_agent_1"
        self.db.logger.log_agent(
            agent_id=agent_id,
            birth_time=0,
            agent_type="test",
            position=(10.0, 20.0),
            initial_resources=100.0,
            starting_health=100.0,
            starvation_threshold=10,
        )

        # Flush agent data
        self.db.logger.flush_all_buffers()

        # Log a simple action
        step_number = 1

        self.db.logger.log_agent_action(
            step_number=step_number,
            agent_id=agent_id,
            action_type="move",
            resources_before=100.0,
            resources_after=95.0,
            reward=5.0,
        )

        # Flush data to database
        self.db.logger.flush_all_buffers()

        # Check if action was stored with correct simulation_id
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT simulation_id FROM agent_actions 
            WHERE agent_id = ? AND step_number = ?
        """,
            (agent_id, step_number),
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Action was not stored in database")
        self.assertEqual(
            result[0],
            self.simulation_id,
            "Action record doesn't have the correct simulation_id",
        )

        # Clean up
        cursor.close()
        conn.close()


if __name__ == "__main__":
    unittest.main()
