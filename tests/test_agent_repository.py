import unittest
from unittest.mock import Mock, patch

from sqlalchemy.orm import Session

from farm.database.data_types import AgentInfo, HealthIncidentData
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
)
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.session_manager import SessionManager


class TestAgentRepository(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for AgentRepository tests."""
        self.session_manager = Mock(spec=SessionManager)
        self.repository = AgentRepository(self.session_manager)
        self.mock_session = Mock(spec=Session)

    def test_get_agent_by_id(self):
        """Test retrieving an agent by ID."""
        # Arrange
        mock_agent = AgentModel(
            agent_id="test_agent_1",
            birth_time=100,
            agent_type="hunter",
            position_x=1.0,
            position_y=2.0,
            initial_resources=100.0,
            starting_health=100.0,
        )

        self.mock_session.query.return_value.options.return_value.get.return_value = (
            mock_agent
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_agent_by_id("test_agent_1")

        # Assert
        self.assertIsNotNone(result)
        assert result is not None  # Type assertion for linter
        self.assertEqual(result.agent_id, "test_agent_1")
        self.assertEqual(result.agent_type, "hunter")
        self.assertEqual(result.birth_time, 100)

    def test_get_actions_by_agent_id(self):
        """Test retrieving actions for a specific agent."""
        # Arrange
        mock_actions = [
            ActionModel(
                action_id=1,
                step_number=1,
                agent_id="test_agent_1",
                action_type="move",
                resources_before=100,
                resources_after=90,
                reward=0.5,
            ),
            ActionModel(
                action_id=2,
                step_number=2,
                agent_id="test_agent_1",
                action_type="gather",
                resources_before=90,
                resources_after=100,
                reward=1.0,
            ),
        ]

        self.mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_actions
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_agent_id("test_agent_1")

        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].action_type, "move")
        self.assertEqual(result[1].action_type, "gather")

    def test_get_states_by_agent_id(self):
        """Test retrieving states for a specific agent."""
        # Arrange
        mock_states = [
            AgentStateModel(
                id="test_agent_1-1",
                step_number=1,
                agent_id="test_agent_1",
                position_x=1.0,
                position_y=1.0,
                resource_level=100,
                current_health=100,
            ),
            AgentStateModel(
                id="test_agent_1-2",
                step_number=2,
                agent_id="test_agent_1",
                position_x=2.0,
                position_y=2.0,
                resource_level=90,
                current_health=95,
            ),
        ]

        self.mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_states
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_states_by_agent_id("test_agent_1")

        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[1].step_number, 2)

    def test_get_health_incidents_by_agent_id(self):
        """Test retrieving health incidents for a specific agent."""
        # Arrange
        mock_incidents = [
            HealthIncident(
                step_number=1,
                agent_id="test_agent_1",
                health_before=100,
                health_after=90,
                cause="damage",
                details="combat injury",
            )
        ]

        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            mock_incidents
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_health_incidents_by_agent_id("test_agent_1")

        # Assert
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HealthIncidentData)
        self.assertEqual(result[0].health_before, 100)
        self.assertEqual(result[0].health_after, 90)
        self.assertEqual(result[0].cause, "damage")

    def test_get_agent_info(self):
        """Test retrieving comprehensive agent information."""
        # Arrange
        mock_agent = AgentModel(
            agent_id="test_agent_1",
            agent_type="hunter",
            birth_time=100,
            death_time=None,
            generation=1,
            genome_id="genome_1",
        )

        mock_latest_state = AgentStateModel(
            agent_id="test_agent_1",
            step_number=10,
            current_health=95,
            resource_level=80,
            position_x=1.0,
            position_y=2.0,
        )

        mock_action_stats = [
            ("move", 5, 0.5),
            ("gather", 3, 1.0),
        ]

        self.mock_session.query.return_value.get.return_value = mock_agent
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            mock_latest_state
        )
        self.mock_session.query.return_value.filter.return_value.group_by.return_value.all.return_value = (
            mock_action_stats
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_agent_info("test_agent_1")

        # Assert
        self.assertIsInstance(result, AgentInfo)
        assert result is not None  # Type assertion for linter
        self.assertEqual(result.agent_id, "test_agent_1")
        self.assertEqual(result.agent_type, "hunter")
        self.assertEqual(result.current_health, 95)
        self.assertEqual(result.current_resources, 80)
        self.assertEqual(result.position, (1.0, 2.0))
        self.assertEqual(len(result.action_stats), 2)

    def test_get_agent_current_stats(self):
        """Test retrieving current statistics for an agent."""
        # Arrange
        mock_latest_state = AgentStateModel(
            agent_id="test_agent_1",
            step_number=10,
            current_health=95,
            resource_level=80,
            total_reward=100,
            age=10,
            is_defending=False,
            position_x=1.0,
            position_y=2.0,
        )

        self.mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            mock_latest_state
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_agent_current_stats("test_agent_1")

        # Assert
        self.assertEqual(result["health"], 95)
        self.assertEqual(result["resources"], 80)
        self.assertEqual(result["total_reward"], 100)
        self.assertEqual(result["age"], 10)
        self.assertEqual(result["is_defending"], False)
        self.assertEqual(result["current_position"], (1.0, 2.0))

    def test_get_agent_children(self):
        """Test retrieving children of an agent."""
        # Arrange
        mock_parent = AgentModel(agent_id="parent_1", genome_id="genome_parent_1")
        mock_children = [
            AgentModel(
                agent_id="child_1", genome_id="genome_child_1_parent_1", birth_time=200
            ),
            AgentModel(
                agent_id="child_2", genome_id="genome_child_2_parent_1", birth_time=300
            ),
        ]

        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_parent
        )
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            mock_children
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_agent_children("parent_1")

        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].agent_id, "child_1")
        self.assertEqual(result[1].agent_id, "child_2")


if __name__ == "__main__":
    unittest.main()
