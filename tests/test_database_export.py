"""
Tests for database export functionality.

Tests the export_data method of DatabaseProtocol implementations,
verifying that data can be exported in various formats with proper filtering.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd
import pytest
from farm.core.interfaces import DatabaseProtocol
from farm.database.database import SimulationDatabase


class TestDatabaseExport(unittest.TestCase):
    """Test database export functionality across different formats and filters."""

    def setUp(self):
        """Set up test database with sample data."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name

        # Create database and add simulation record
        self.db: DatabaseProtocol = SimulationDatabase(self.db_path, simulation_id="test_export_sim")
        self.db.add_simulation_record(
            simulation_id="test_export_sim", start_time=datetime.now(), status="running", parameters={"test": True}
        )

        # Add sample agent data
        self._add_sample_agent_data()

        # Add sample action data
        self._add_sample_action_data()

        # Log some steps
        self._add_sample_step_data()

    def tearDown(self):
        """Clean up test database and exported files."""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

        # Clean up any exported files
        test_files = [
            # Base export files
            "test_export.csv",
            "test_export.xlsx",
            "test_export.json",
            "test_export.parquet",
            # CSV format creates separate files per data type + metadata
            "test_export_agents.csv",
            "test_export_actions.csv",
            "test_export_metrics.csv",
            "test_export_resources.csv",
            "test_export_metadata.json",
            # Parquet format creates separate files per data type
            "test_export_agents.parquet",
            "test_export_actions.parquet",
            "test_export_metrics.parquet",
            "test_export_resources.parquet",
            # Filtered export (only agents + metadata)
            "test_export_filtered_agents.csv",
            "test_export_filtered_metadata.json",
            # Range export creates separate files per data type + metadata
            "test_export_range_agents.csv",
            "test_export_range_actions.csv",
            "test_export_range_metrics.csv",
            "test_export_range_resources.csv",
            "test_export_range_metadata.json",
            # No metadata export
            "test_export_no_metadata.json",
        ]

        for filename in test_files:
            if os.path.exists(filename):
                os.unlink(filename)

    def _add_sample_agent_data(self):
        """Add sample agent data for testing."""
        agent_data = [
            {
                "simulation_id": "test_export_sim",
                "agent_id": "agent_1",
                "birth_time": 0,
                "agent_type": "BaseAgent",
                "position": (10.0, 20.0),
                "initial_resources": 50.0,
                "starting_health": 100.0,
                "genome_id": "genome_1",
                "generation": 1,
            },
            {
                "simulation_id": "test_export_sim",
                "agent_id": "agent_2",
                "birth_time": 5,
                "agent_type": "BaseAgent",
                "position": (30.0, 40.0),
                "initial_resources": 75.0,
                "starting_health": 100.0,
                "starvation_counter": 0,
                "genome_id": "genome_2",
                "generation": 1,
            },
        ]
        self.db.logger.log_agents_batch(agent_data)

    def _add_sample_action_data(self):
        """Add sample action data for testing."""
        self.db.logger.log_agent_action(
            step_number=1,
            agent_id="agent_1",
            action_type="move",
            action_target_id=None,
            reward=1.0,
            details={
                "position": (15.0, 25.0),
                "agent_resources_before": 50.0,
                "agent_resources_after": 49.0,
            },
        )

        self.db.logger.log_agent_action(
            step_number=2,
            agent_id="agent_2",
            action_type="move",
            action_target_id=None,
            reward=1.0,
            details={
                "position": (30.0, 40.0),
                "agent_resources_before": 75.0,
                "agent_resources_after": 74.0,
            },
        )

    def _add_sample_step_data(self):
        """Add sample step data for testing."""
        # Log step data - agent states as tuples: (agent_id, position_x, position_y, resource_level, current_health, starting_health, starvation_counter, is_defending, total_reward, age)
        agent_states = [
            ("agent_1", 15.0, 25.0, 49.0, 100.0, 100.0, 0, False, 1.0, 1),
            ("agent_2", 30.0, 40.0, 85.0, 100.0, 100.0, 0, False, 2.0, 2),
        ]
        resource_states = [("resource_1", 35.0, 45.0, 50.0)]
        metrics = {"total_agents": 2, "total_resources": 100.0}

        self.db.logger.log_step(1, agent_states, resource_states, metrics)
        self.db.logger.log_step(2, agent_states, resource_states, metrics)

        # Flush buffers to ensure data is written
        self.db.logger.flush_all_buffers()

    def test_export_csv_format(self):
        """Test exporting data in CSV format."""
        filepath = "test_export.csv"

        # Export all data types
        self.db.export_data(filepath, format="csv")

        # Check that files were created
        self.assertTrue(os.path.exists("test_export_agents.csv"))
        self.assertTrue(os.path.exists("test_export_actions.csv"))
        self.assertTrue(os.path.exists("test_export_metrics.csv"))

        # Verify CSV content
        agents_df = pd.read_csv("test_export_agents.csv")
        self.assertGreater(len(agents_df), 0)
        self.assertIn("agent_id", agents_df.columns)
        # Should contain data for both agents across steps
        agent_ids = agents_df["agent_id"].tolist()
        self.assertIn("agent_1", agent_ids)
        self.assertIn("agent_2", agent_ids)

    def test_export_excel_format(self):
        """Test exporting data in Excel format."""
        pytest.importorskip("openpyxl", reason="openpyxl required for Excel export")

        filepath = "test_export.xlsx"

        # Export data
        self.db.export_data(filepath, format="excel")

        # Check that file was created
        self.assertTrue(os.path.exists(filepath))

        # Verify Excel content
        agents_df = pd.read_excel(filepath, sheet_name="agents")
        self.assertGreater(len(agents_df), 0)
        self.assertIn("agent_id", agents_df.columns)

    def test_export_json_format(self):
        """Test exporting data in JSON format."""
        filepath = "test_export.json"

        # Export data
        self.db.export_data(filepath, format="json")

        # Check that file was created
        self.assertTrue(os.path.exists(filepath))

        # Verify JSON content
        with open(filepath, "r") as f:
            data = json.load(f)

        self.assertIn("agents", data)
        self.assertIn("actions", data)
        self.assertIn("metrics", data)
        self.assertGreater(len(data["agents"]), 0)
        # Should contain agent data (agent states joined with agent metadata)
        agent_ids = [agent["agent_id"] for agent in data["agents"]]
        self.assertIn("agent_1", agent_ids)
        self.assertIn("agent_2", agent_ids)

    def test_export_parquet_format(self):
        """Test exporting data in Parquet format."""
        filepath = "test_export.parquet"

        # Export data
        self.db.export_data(filepath, format="parquet")

        # Check that files were created
        self.assertTrue(os.path.exists("test_export_agents.parquet"))
        self.assertTrue(os.path.exists("test_export_actions.parquet"))
        self.assertTrue(os.path.exists("test_export_metrics.parquet"))

        # Verify Parquet content
        agents_df = pd.read_parquet("test_export_agents.parquet")
        self.assertGreater(len(agents_df), 0)
        self.assertIn("agent_id", agents_df.columns)
        agent_ids = agents_df["agent_id"].tolist()
        self.assertIn("agent_1", agent_ids)
        self.assertIn("agent_2", agent_ids)

    def test_export_with_data_type_filter(self):
        """Test exporting with specific data type filters."""
        filepath = "test_export_filtered.csv"

        # Export only agents data
        self.db.export_data(filepath, format="csv", data_types=["agents"])

        # Check that only agents file was created
        self.assertTrue(os.path.exists("test_export_filtered_agents.csv"))
        self.assertFalse(os.path.exists("test_export_filtered_actions.csv"))
        self.assertFalse(os.path.exists("test_export_filtered_metrics.csv"))

    def test_export_with_step_range_filter(self):
        """Test exporting with step range filters."""
        filepath = "test_export_range.csv"

        # Add more step data
        self.db.logger.log_step(3, [], [], {"total_agents": 0})
        self.db.logger.flush_all_buffers()

        # Export only steps 1-2
        self.db.export_data(filepath, format="csv", start_step=1, end_step=2)

        # Verify content
        metrics_df = pd.read_csv("test_export_range_metrics.csv")
        # Should only have steps 1 and 2
        self.assertTrue(len(metrics_df) <= 2)

    def test_export_unsupported_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.db.export_data("test.invalid", format="invalid")

        self.assertIn("Unsupported export format", str(cm.exception))

    def test_export_without_metadata(self):
        """Test exporting without metadata inclusion."""
        filepath = "test_export_no_metadata.json"

        # Export without metadata
        self.db.export_data(filepath, format="json", include_metadata=False)

        # Check that file was created
        self.assertTrue(os.path.exists(filepath))

        # Verify no metadata in JSON
        with open(filepath, "r") as f:
            data = json.load(f)

        # Metadata should not be present or should be minimal
        if "metadata" in data:
            # If present, it should be empty or minimal
            self.assertTrue(len(data["metadata"]) == 0 or data["metadata"] is None)


if __name__ == "__main__":
    unittest.main()
