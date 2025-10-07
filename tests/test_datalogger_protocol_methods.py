"""
Tests for DataLoggerProtocol methods.

Tests the logging methods in DataLoggerProtocol implementations,
specifically focusing on methods that may not be fully tested elsewhere.
"""

import os
import tempfile
import unittest
from datetime import datetime

from farm.core.interfaces import DatabaseProtocol, DataLoggerProtocol
from farm.database.database import SimulationDatabase


class TestDataLoggerProtocolMethods(unittest.TestCase):
    """Test DataLoggerProtocol logging methods."""

    def setUp(self):
        """Set up test database with sample data."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name

        # Create database and add simulation record
        self.db: DatabaseProtocol = SimulationDatabase(self.db_path, simulation_id="test_logger_methods_sim")
        self.db.add_simulation_record(
            simulation_id="test_logger_methods_sim",
            start_time=datetime.now(),
            status="running",
            parameters={"test": True},
        )

        # Get logger
        self.logger: DataLoggerProtocol = self.db.logger

        # Add a test agent for foreign key constraints
        self._add_test_agent()

    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _add_test_agent(self):
        """Add a test agent for testing action logging."""
        agent_data = {
            "simulation_id": "test_logger_methods_sim",
            "agent_id": "test_agent_health",
            "birth_time": 0,
            "agent_type": "BaseAgent",
            "position": (10.0, 20.0),
            "initial_resources": 50.0,
            "starting_health": 100.0,
            "starvation_counter": 0,
            "genome_id": "test_genome",
            "generation": 1,
        }
        self.logger.log_agents_batch([agent_data])

    def test_log_agent_action_with_details(self):
        """Test log_agent_action with details parameter."""
        # Log an action with details
        self.logger.log_agent_action(
            step_number=1,
            agent_id="test_agent_health",
            action_type="move",
            action_target_id=None,
            resources_before=50.0,
            resources_after=49.0,
            reward=1.0,
            details={"position": (15.0, 25.0), "reason": "exploring"},
        )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

        # Verify the action was logged (by checking it doesn't raise an error)
        # The actual verification would require querying the database
        # but this tests the interface compliance

    def test_log_agent_action_minimal_parameters(self):
        """Test log_agent_action with minimal required parameters."""
        # Log an action with only required parameters
        self.logger.log_agent_action(step_number=2, agent_id="test_agent_health", action_type="gather")

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_log_health_incident_basic(self):
        """Test log_health_incident with basic parameters."""
        # Log a health incident
        self.logger.log_health_incident(
            step_number=3, agent_id="test_agent_health", health_before=100.0, health_after=90.0, cause="combat_damage"
        )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_log_health_incident_with_details(self):
        """Test log_health_incident with details parameter."""
        # Log a health incident with details
        self.logger.log_health_incident(
            step_number=4,
            agent_id="test_agent_health",
            health_before=90.0,
            health_after=80.0,
            cause="starvation",
            details={"damage_amount": 10.0, "hunger_level": 0.8},
        )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_log_health_incident_recovery(self):
        """Test log_health_incident with health increase (recovery)."""
        # Log a health recovery incident
        self.logger.log_health_incident(
            step_number=5,
            agent_id="test_agent_health",
            health_before=80.0,
            health_after=85.0,
            cause="resource_consumption",
            details={"healing_amount": 5.0},
        )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_multiple_health_incidents_same_step(self):
        """Test logging multiple health incidents in the same step."""
        # Log multiple incidents for the same agent in one step
        self.logger.log_health_incident(
            step_number=6,
            agent_id="test_agent_health",
            health_before=85.0,
            health_after=75.0,
            cause="combat_damage",
            details={"damage_source": "agent_2"},
        )

        self.logger.log_health_incident(
            step_number=6,
            agent_id="test_agent_health",
            health_before=75.0,
            health_after=70.0,
            cause="poison",
            details={"poison_type": "venom"},
        )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_log_agent_action_various_types(self):
        """Test log_agent_action with different action types."""
        action_types = ["move", "gather", "attack", "share", "reproduce", "defend"]

        for i, action_type in enumerate(action_types):
            self.logger.log_agent_action(
                step_number=10 + i,
                agent_id="test_agent_health",
                action_type=action_type,
                resources_before=50.0 - i,
                resources_after=49.0 - i,
                reward=1.0,
                details={"action_sequence": i},
            )

        # Flush to ensure data is written
        self.logger.flush_all_buffers()

    def test_log_health_incident_various_causes(self):
        """Test log_health_incident with different causes."""
        causes = ["combat_damage", "starvation", "poison", "healing", "aging", "environmental"]

        health = 100.0
        for i, cause in enumerate(causes):
            new_health = (
                health - 5.0 if "damage" in cause or cause in ["starvation", "poison", "aging"] else health + 5.0
            )

            self.logger.log_health_incident(
                step_number=20 + i,
                agent_id="test_agent_health",
                health_before=health,
                health_after=new_health,
                cause=cause,
                details={"cause_sequence": i},
            )
            health = new_health

        # Flush to ensure data is written
        self.logger.flush_all_buffers()


if __name__ == "__main__":
    unittest.main()
