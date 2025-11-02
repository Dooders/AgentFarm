"""
Unit tests for the InMemorySimulationDatabase implementation.

This module contains tests to validate the behavior of the in-memory database,
including creation, operations, persistence, and memory monitoring.
"""

import os
import shutil
import tempfile
import time
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import psutil
from sqlalchemy import text

from farm.config import SimulationConfig
from farm.database.database import InMemorySimulationDatabase, SimulationDatabase


class TestInMemorySimulationDatabase(unittest.TestCase):
    """Test suite for InMemorySimulationDatabase."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_db.db")

        # Create an in-memory database instance
        self.db = InMemorySimulationDatabase()

        # Sample data for testing
        self.sample_config = {
            "width": 100,
            "height": 100,
            "system_agents": 10,
            "independent_agents": 10,
            "control_agents": 10,
        }

        # Sample agent data
        self.sample_agent_data = {
            "agent_id": "test_agent_1",
            "birth_time": 0,
            "agent_type": "SystemAgent",
            "position": (50, 50),
            "initial_resources": 10,
            "starting_health": 100,
            "genome_id": "genome_1",
            "generation": 1,
        }

        # Sample step data
        self.sample_step_data = {
            "step_number": 1,
            "total_agents": 30,
            "agent_type_counts": {"system": 10, "independent": 10, "control": 10},
            "total_resources": 100,
            "average_agent_resources": 5.0,
        }

    def tearDown(self):
        """Clean up after each test."""
        # Close database connection
        if hasattr(self, "db") and self.db:
            try:
                self.db.close()
            except Exception:
                pass

        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that the database initializes correctly."""
        # Verify database is created
        self.assertIsNotNone(self.db)
        self.assertEqual(self.db.db_path, ":memory:")

        # Verify engine is created
        self.assertIsNotNone(self.db.engine)

        # Verify session factory is created
        self.assertIsNotNone(self.db.Session)

        # Verify logger is created
        self.assertIsNotNone(self.db.logger)

        # Verify query interface is created
        self.assertIsNotNone(self.db.query)

    def test_memory_limit_initialization(self):
        """Test initialization with memory limit."""
        # Create database with memory limit
        memory_limit = 1024  # 1GB
        db = InMemorySimulationDatabase(memory_limit_mb=memory_limit)

        # Verify memory limit is set
        self.assertEqual(db.memory_limit_mb, memory_limit)

        # Clean up
        db.close()

    def test_save_and_retrieve_configuration(self):
        """Test saving and retrieving configuration."""
        # Save configuration
        self.db.save_configuration(self.sample_config)

        # Retrieve configuration
        config = self.db.get_configuration()

        # Verify configuration is retrieved correctly
        for key, value in self.sample_config.items():
            self.assertEqual(config[key], value)

    def test_log_and_retrieve_agent(self):
        """Test logging and retrieving agent data."""
        # Log agent
        self.db.logger.log_agent(
            agent_id=self.sample_agent_data["agent_id"],
            birth_time=self.sample_agent_data["birth_time"],
            agent_type=self.sample_agent_data["agent_type"],
            position=self.sample_agent_data["position"],
            initial_resources=self.sample_agent_data["initial_resources"],
            starting_health=self.sample_agent_data["starting_health"],
            genome_id=self.sample_agent_data["genome_id"],
            generation=self.sample_agent_data["generation"],
        )

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

        # Query agent data
        session = self.db.Session()
        try:
            # Use raw SQL for simplicity
            result = session.execute(
                text("SELECT * FROM agents WHERE agent_id = :agent_id"),
                {"agent_id": self.sample_agent_data["agent_id"]},
            )
            agent_row = result.fetchone()

            # Verify agent data is retrieved correctly
            self.assertIsNotNone(agent_row)
            agent_row = cast(Any, agent_row)
            self.assertEqual(agent_row.agent_id, self.sample_agent_data["agent_id"])
            self.assertEqual(agent_row.agent_type, self.sample_agent_data["agent_type"])
            self.assertEqual(agent_row.birth_time, self.sample_agent_data["birth_time"])
            self.assertEqual(agent_row.generation, self.sample_agent_data["generation"])
        finally:
            session.close()

    def test_log_and_retrieve_step(self):
        """Test logging and retrieving step data."""
        # Log step
        self.db.logger.log_step(
            step_number=self.sample_step_data["step_number"],
            agent_states=[],  # Empty for simplicity
            resource_states=[],  # Empty for simplicity
            metrics=self.sample_step_data,
        )

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

        # Query step data
        session = self.db.Session()
        try:
            # Use raw SQL for simplicity
            result = session.execute(
                text("SELECT * FROM simulation_steps WHERE step_number = :step_number"),
                {"step_number": self.sample_step_data["step_number"]},
            )
            step_row = result.fetchone()

            # Verify step data is retrieved correctly
            self.assertIsNotNone(step_row)
            step_row = cast(Any, step_row)
            self.assertEqual(step_row.step_number, self.sample_step_data["step_number"])
            self.assertEqual(
                step_row.total_agents, self.sample_step_data["total_agents"]
            )
            # Check agent_type_counts JSON column
            import json
            agent_counts = json.loads(step_row.agent_type_counts) if isinstance(step_row.agent_type_counts, str) else step_row.agent_type_counts
            self.assertEqual(agent_counts.get("system"), self.sample_step_data["agent_type_counts"]["system"])
            self.assertEqual(agent_counts.get("independent"), self.sample_step_data["agent_type_counts"]["independent"])
            self.assertEqual(agent_counts.get("control"), self.sample_step_data["agent_type_counts"]["control"])
        finally:
            session.close()

    def test_persist_to_disk(self):
        """Test persisting in-memory database to disk."""
        # Add some data
        self.db.save_configuration(self.sample_config)

        # Log agent
        self.db.logger.log_agent(
            agent_id=self.sample_agent_data["agent_id"],
            birth_time=self.sample_agent_data["birth_time"],
            agent_type=self.sample_agent_data["agent_type"],
            position=self.sample_agent_data["position"],
            initial_resources=self.sample_agent_data["initial_resources"],
            starting_health=self.sample_agent_data["starting_health"],
            genome_id=self.sample_agent_data["genome_id"],
            generation=self.sample_agent_data["generation"],
        )

        # Log step
        self.db.logger.log_step(
            step_number=self.sample_step_data["step_number"],
            agent_states=[],  # Empty for simplicity
            resource_states=[],  # Empty for simplicity
            metrics=self.sample_step_data,
        )

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

        # Persist to disk
        stats = self.db.persist_to_disk(self.db_path, show_progress=False)

        # Verify persistence statistics
        self.assertIsNotNone(stats)
        self.assertIn("tables_copied", stats)
        self.assertIn("rows_copied", stats)
        self.assertIn("duration", stats)

        # Verify database file exists
        self.assertTrue(os.path.exists(self.db_path))

        # Open the persisted database and verify data
        disk_db = SimulationDatabase(self.db_path)
        try:
            # Verify configuration
            config = disk_db.get_configuration()
            for key, value in self.sample_config.items():
                self.assertEqual(config[key], value)

            # Verify agent data
            session = disk_db.Session()
            try:
                result = session.execute(
                    text("SELECT * FROM agents WHERE agent_id = :agent_id"),
                    {"agent_id": self.sample_agent_data["agent_id"]},
                )
                agent_row = result.fetchone()
                self.assertIsNotNone(agent_row)
                agent_row = cast(Any, agent_row)
                self.assertEqual(agent_row.agent_id, self.sample_agent_data["agent_id"])
            finally:
                session.close()

            # Verify step data
            session = disk_db.Session()
            try:
                result = session.execute(
                    text(
                        "SELECT * FROM simulation_steps WHERE step_number = :step_number"
                    ),
                    {"step_number": self.sample_step_data["step_number"]},
                )
                step_row = result.fetchone()
                self.assertIsNotNone(step_row)
                step_row = cast(Any, step_row)
                self.assertEqual(
                    step_row.step_number, self.sample_step_data["step_number"]
                )
            finally:
                session.close()
        finally:
            disk_db.close()

    def test_selective_persistence(self):
        """Test persisting only selected tables."""
        # Add some data
        self.db.save_configuration(self.sample_config)

        # Log agent
        self.db.logger.log_agent(
            agent_id=self.sample_agent_data["agent_id"],
            birth_time=self.sample_agent_data["birth_time"],
            agent_type=self.sample_agent_data["agent_type"],
            position=self.sample_agent_data["position"],
            initial_resources=self.sample_agent_data["initial_resources"],
            starting_health=self.sample_agent_data["starting_health"],
            genome_id=self.sample_agent_data["genome_id"],
            generation=self.sample_agent_data["generation"],
        )

        # Log step
        self.db.logger.log_step(
            step_number=self.sample_step_data["step_number"],
            agent_states=[],  # Empty for simplicity
            resource_states=[],  # Empty for simplicity
            metrics=self.sample_step_data,
        )

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

        # Persist only simulation_config table
        tables_to_persist = ["simulation_config"]
        stats = self.db.persist_to_disk(
            self.db_path, tables=tables_to_persist, show_progress=False
        )

        # Verify persistence statistics
        self.assertEqual(stats["tables_copied"], 1)

        # Open the persisted database and verify data
        disk_db = SimulationDatabase(self.db_path)
        try:
            # Verify configuration exists
            config = disk_db.get_configuration()
            self.assertIsNotNone(config)

            # Verify agent data does not exist
            session = disk_db.Session()
            try:
                result = session.execute(text("SELECT COUNT(*) FROM agents"))
                count = result.scalar()
                self.assertEqual(count, 0)
            finally:
                session.close()
        finally:
            disk_db.close()

    @patch("psutil.Process")
    def test_memory_monitoring(self, mock_process):
        """Test memory usage monitoring."""
        # Mock psutil.Process to return predictable memory values
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        # Mock memory_info to return a namedtuple with rss attribute
        memory_info = MagicMock()
        memory_info.rss = 500 * 1024 * 1024  # 500MB
        mock_process_instance.memory_info.return_value = memory_info

        # Create database with memory monitoring
        with patch("threading.Thread") as mock_thread:
            db = InMemorySimulationDatabase(memory_limit_mb=1000)

            # Verify thread was started for memory monitoring
            self.assertTrue(mock_thread.called)

            # Manually call the monitor_memory function to simulate thread execution
            # Extract the target function from the Thread constructor call
            monitor_func = mock_thread.call_args[1]["target"]

            # Call it once to populate memory_usage_samples
            monitor_func()

            # Verify memory usage was recorded
            self.assertEqual(len(db.memory_usage_samples), 1)
            self.assertEqual(db.memory_usage_samples[0], 500)

            # Get memory usage stats
            stats = db.get_memory_usage()
            self.assertEqual(stats["current_mb"], 500)
            self.assertEqual(stats["limit_mb"], 1000)
            self.assertEqual(stats["usage_percent"], 50)

            # Clean up
            db.close()

    @patch("psutil.Process")
    def test_memory_warning_threshold(self, mock_process):
        """Test memory warning threshold."""
        # Mock psutil.Process to return memory values near warning threshold
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        # Mock memory_info to return a value above warning threshold
        memory_info = MagicMock()
        memory_info.rss = 850 * 1024 * 1024  # 850MB (85% of 1000MB limit)
        mock_process_instance.memory_info.return_value = memory_info

        # Create database with memory monitoring
        with patch("threading.Thread") as mock_thread, patch(
            "logging.warning"
        ) as mock_warning:

            db = InMemorySimulationDatabase(memory_limit_mb=1000)

            # Extract the target function from the Thread constructor call
            monitor_func = mock_thread.call_args[1]["target"]

            # Call it to trigger warning
            monitor_func()

            # Verify warning was logged
            self.assertTrue(mock_warning.called)

            # Clean up
            db.close()

    @patch("psutil.Process")
    def test_memory_critical_threshold(self, mock_process):
        """Test memory critical threshold."""
        # Mock psutil.Process to return memory values near critical threshold
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        # Mock memory_info to return a value above critical threshold
        memory_info = MagicMock()
        memory_info.rss = 960 * 1024 * 1024  # 960MB (96% of 1000MB limit)
        mock_process_instance.memory_info.return_value = memory_info

        # Create database with memory monitoring
        with patch("threading.Thread") as mock_thread, patch(
            "logging.critical"
        ) as mock_critical:

            db = InMemorySimulationDatabase(memory_limit_mb=1000)

            # Extract the target function from the Thread constructor call
            monitor_func = mock_thread.call_args[1]["target"]

            # Call it to trigger critical warning
            monitor_func()

            # Verify critical warning was logged
            self.assertTrue(mock_critical.called)

            # Clean up
            db.close()

    def test_memory_trend_detection(self):
        """Test memory usage trend detection."""
        # Create database
        db = InMemorySimulationDatabase(memory_limit_mb=1000)

        # Manually set memory samples to simulate increasing trend
        db.memory_usage_samples = [100, 120, 150]

        # Get memory usage stats
        stats = db.get_memory_usage()
        self.assertEqual(stats["trend"], "increasing")

        # Manually set memory samples to simulate decreasing trend
        db.memory_usage_samples = [150, 120, 100]

        # Get memory usage stats
        stats = db.get_memory_usage()
        self.assertEqual(stats["trend"], "decreasing")

        # Manually set memory samples to simulate stable trend
        db.memory_usage_samples = [100, 102, 101]

        # Get memory usage stats
        stats = db.get_memory_usage()
        self.assertEqual(stats["trend"], "stable")

        # Clean up
        db.close()

    def test_large_batch_persistence(self):
        """Test persisting large batches of data."""
        # Create a large number of step records
        num_steps = 2000

        # Log steps
        for step in range(num_steps):
            self.db.logger.log_step(
                step_number=step,
                agent_states=[],  # Empty for simplicity
                resource_states=[],  # Empty for simplicity
                metrics={"total_agents": 30, "step_number": step},
            )

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

        # Persist to disk
        start_time = time.time()
        stats = self.db.persist_to_disk(self.db_path, show_progress=False)
        end_time = time.time()

        # Verify persistence statistics
        self.assertGreaterEqual(stats["rows_copied"], num_steps)

        # Verify database file exists
        self.assertTrue(os.path.exists(self.db_path))

        # Open the persisted database and verify data
        disk_db = SimulationDatabase(self.db_path)
        try:
            # Verify step count
            session = disk_db.Session()
            try:
                result = session.execute(text("SELECT COUNT(*) FROM simulation_steps"))
                count = result.scalar()
                self.assertEqual(count, num_steps)
            finally:
                session.close()
        finally:
            disk_db.close()

        # Print performance metrics
        print(f"Persisted {num_steps} steps in {end_time - start_time:.2f} seconds")
        print(
            f"Persistence rate: {num_steps / (end_time - start_time):.2f} rows/second"
        )

    def test_error_handling_during_persistence(self):
        """Test error handling during persistence."""
        # Add some data
        self.db.save_configuration(self.sample_config)

        # Mock os.makedirs to raise an exception
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = PermissionError("Permission denied")

            # Attempt to persist to disk
            with self.assertRaises(Exception):
                self.db.persist_to_disk("/invalid/path/db.db", show_progress=False)

    def test_integration_with_simulation_config(self):
        """Test integration with SimulationConfig."""
        # Create a simulation config with in-memory database settings
        config = SimulationConfig()
        config.use_in_memory_db = True
        config.in_memory_db_memory_limit_mb = 2048
        config.persist_db_on_completion = True

        # Verify settings
        self.assertTrue(config.use_in_memory_db)
        self.assertEqual(config.in_memory_db_memory_limit_mb, 2048)
        self.assertTrue(config.persist_db_on_completion)

        # Convert to dict and back to ensure serialization works
        config_dict = config.to_dict()
        new_config = SimulationConfig.from_dict(config_dict)

        # Verify settings are preserved
        self.assertTrue(new_config.use_in_memory_db)
        self.assertEqual(new_config.in_memory_db_memory_limit_mb, 2048)
        self.assertTrue(new_config.persist_db_on_completion)


if __name__ == "__main__":
    unittest.main()
