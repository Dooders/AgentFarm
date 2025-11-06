"""Comprehensive tests for AgentRepository with mocked sessions.

Tests all query methods, relationships, edge cases, and error handling.
"""

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


class TestAgentRepositoryComprehensive(unittest.TestCase):
    """Comprehensive tests for AgentRepository."""

    def setUp(self):
        """Set up test fixtures."""
        self.session_manager = Mock(spec=SessionManager)
        self.repository = AgentRepository(self.session_manager)
        self.mock_session = Mock(spec=Session)

        # Setup default mock behavior
        def execute_with_retry_side_effect(func):
            return func(self.mock_session)

        self.session_manager.execute_with_retry.side_effect = (
            execute_with_retry_side_effect
        )

    def test_get_agent_by_id_with_relationships(self):
        """Test get_agent_by_id loads relationships correctly."""
        mock_agent = Mock(spec=AgentModel)
        mock_agent.agent_id = "test_agent_1"
        mock_agent.states = []
        mock_agent.actions = []
        mock_agent.health_incidents = []

        mock_query = Mock()
        mock_query.options.return_value.get.return_value = mock_agent
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_by_id("test_agent_1")

        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "test_agent_1")
        # Verify joinedload was called
        mock_query.options.assert_called_once()

    def test_get_agent_by_id_not_found(self):
        """Test get_agent_by_id returns None when agent not found."""
        mock_query = Mock()
        mock_query.options.return_value.get.return_value = None
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_by_id("nonexistent")

        self.assertIsNone(result)

    def test_get_actions_by_agent_id_empty(self):
        """Test get_actions_by_agent_id with no actions."""
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = []
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_actions_by_agent_id("test_agent_1")

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_get_states_by_agent_id_ordered(self):
        """Test get_states_by_agent_id returns ordered states."""
        mock_states = [
            Mock(spec=AgentStateModel, step_number=1),
            Mock(spec=AgentStateModel, step_number=2),
            Mock(spec=AgentStateModel, step_number=3),
        ]

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_states
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_states_by_agent_id("test_agent_1")

        self.assertEqual(len(result), 3)

    def test_get_health_incidents_by_agent_id(self):
        """Test get_health_incidents_by_agent_id with JSON details."""
        mock_incident = Mock(spec=HealthIncident)
        mock_incident.step_number = 10
        mock_incident.health_before = 100.0
        mock_incident.health_after = 80.0
        mock_incident.cause = "combat"
        mock_incident.details = '{"damage": 20, "attacker": "agent_2"}'

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_incident
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_health_incidents_by_agent_id("test_agent_1")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HealthIncidentData)
        self.assertEqual(result[0].step, 10)
        self.assertEqual(result[0].health_before, 100.0)
        self.assertEqual(result[0].health_after, 80.0)
        self.assertEqual(result[0].cause, "combat")

    def test_get_health_incidents_with_invalid_json(self):
        """Test get_health_incidents handles invalid JSON gracefully."""
        mock_incident = Mock(spec=HealthIncident)
        mock_incident.step_number = 10
        mock_incident.health_before = 100.0
        mock_incident.health_after = 80.0
        mock_incident.cause = "combat"
        mock_incident.details = "invalid json {"

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_incident
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_health_incidents_by_agent_id("test_agent_1")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].details, {})

    def test_get_ordered_actions_with_agent_filter(self):
        """Test get_ordered_actions with agent_id filter."""
        mock_actions = [
            Mock(spec=ActionModel, step_number=1),
            Mock(spec=ActionModel, step_number=2),
        ]

        mock_query = Mock()
        mock_query.order_by.return_value.filter.return_value.all.return_value = (
            mock_actions
        )
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_ordered_actions(agent_id="test_agent_1")

        self.assertEqual(len(result), 2)

    def test_get_ordered_actions_without_filter(self):
        """Test get_ordered_actions without agent_id filter."""
        mock_actions = [Mock(spec=ActionModel) for _ in range(5)]

        mock_query = Mock()
        mock_query.order_by.return_value.all.return_value = mock_actions
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_ordered_actions()

        self.assertEqual(len(result), 5)

    def test_get_action_statistics(self):
        """Test get_action_statistics with grouped results."""
        mock_result = Mock()
        mock_result.action_type = "move"
        mock_result.reward_avg = 0.5
        mock_result.count = 10
        mock_result.rewards = "0.5,0.6,0.4,0.5,0.5,0.6,0.4,0.5,0.5,0.6"

        mock_query = Mock()
        mock_query.group_by.return_value.all.return_value = [mock_result]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_action_statistics()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].action_type, "move")
        self.assertEqual(result[0].count, 10)
        self.assertGreater(result[0].reward_std, 0)

    def test_get_action_statistics_with_agent_filter(self):
        """Test get_action_statistics with agent_id filter."""
        mock_result = Mock()
        mock_result.action_type = "gather"
        mock_result.reward_avg = 1.0
        mock_result.count = 5
        mock_result.rewards = "1.0,1.0,1.0,1.0,1.0"

        mock_query = Mock()
        mock_query.group_by.return_value.filter.return_value.all.return_value = [
            mock_result
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_action_statistics(agent_id="test_agent_1")

        self.assertEqual(len(result), 1)

    def test_get_action_statistics_empty_rewards(self):
        """Test get_action_statistics handles empty rewards."""
        mock_result = Mock()
        mock_result.action_type = "idle"
        mock_result.reward_avg = 0.0
        mock_result.count = 0
        mock_result.rewards = None

        mock_query = Mock()
        mock_query.group_by.return_value.all.return_value = [mock_result]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_action_statistics()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].reward_std, 0.0)

    def test_get_actions_with_states(self):
        """Test get_actions_with_states joins correctly."""
        mock_action = Mock(spec=ActionModel)
        mock_state = Mock(spec=AgentStateModel)
        mock_row = (mock_action, mock_state)

        mock_query = Mock()
        mock_query.join.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_actions_with_states()

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_get_actions_with_states_with_filter(self):
        """Test get_actions_with_states with agent_id filter."""
        mock_action = Mock(spec=ActionModel)
        mock_state = Mock(spec=AgentStateModel)
        mock_row = (mock_action, mock_state)

        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_actions_with_states(agent_id="test_agent_1")

        self.assertEqual(len(result), 1)

    def test_get_agent_info_complete(self):
        """Test get_agent_info returns complete information."""
        mock_agent = Mock(spec=AgentModel)
        mock_agent.agent_id = "test_agent_1"
        mock_agent.agent_type = "hunter"
        mock_agent.birth_time = 100
        mock_agent.death_time = None
        mock_agent.generation = 1
        mock_agent.genome_id = "genome_1"

        mock_latest_state = Mock(spec=AgentStateModel)
        mock_latest_state.current_health = 80.0
        mock_latest_state.resource_level = 50.0
        mock_latest_state.position_x = 10.0
        mock_latest_state.position_y = 20.0

        mock_action_stat = ("move", 10, 0.5)

        # Setup query chain
        self.mock_session.query.return_value.get.return_value = mock_agent

        # Setup latest state query
        state_query = Mock()
        state_query.filter.return_value.order_by.return_value.first.return_value = (
            mock_latest_state
        )
        self.mock_session.query.side_effect = [
            Mock(get=Mock(return_value=mock_agent)),
            state_query,
            Mock(
                filter=Mock(
                    return_value=Mock(
                        group_by=Mock(return_value=Mock(all=Mock(return_value=[mock_action_stat])))
                    )
                )
            ),
        ]

        result = self.repository.get_agent_info("test_agent_1")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, AgentInfo)
        self.assertEqual(result.agent_id, "test_agent_1")

    def test_get_agent_info_not_found(self):
        """Test get_agent_info returns None when agent not found."""
        self.mock_session.query.return_value.get.return_value = None

        result = self.repository.get_agent_info("nonexistent")

        self.assertIsNone(result)

    def test_get_agent_current_stats(self):
        """Test get_agent_current_stats returns current state."""
        mock_state = Mock(spec=AgentStateModel)
        mock_state.current_health = 75.0
        mock_state.resource_level = 60.0
        mock_state.total_reward = 100.0
        mock_state.age = 50
        mock_state.is_defending = False
        mock_state.position_x = 15.0
        mock_state.position_y = 25.0

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.first.return_value = (
            mock_state
        )
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_current_stats("test_agent_1")

        self.assertEqual(result["health"], 75.0)
        self.assertEqual(result["resources"], 60.0)
        self.assertEqual(result["total_reward"], 100.0)
        self.assertEqual(result["age"], 50)
        self.assertEqual(result["is_defending"], False)
        self.assertEqual(result["current_position"], (15.0, 25.0))

    def test_get_agent_current_stats_no_state(self):
        """Test get_agent_current_stats with no state history."""
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.first.return_value = None
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_current_stats("test_agent_1")

        self.assertEqual(result["health"], 0)
        self.assertEqual(result["resources"], 0)
        self.assertEqual(result["total_reward"], 0)
        self.assertEqual(result["age"], 0)
        self.assertEqual(result["is_defending"], False)
        self.assertEqual(result["current_position"], (0, 0))

    def test_get_agent_performance_metrics(self):
        """Test get_agent_performance_metrics calculates correctly."""
        mock_agent = Mock(spec=AgentModel)
        mock_agent.birth_time = 100
        mock_agent.death_time = 200

        mock_metrics = Mock()
        mock_metrics.peak_health = 100.0
        mock_metrics.peak_resources = 150.0

        # Setup query chain
        self.mock_session.query.side_effect = [
            Mock(get=Mock(return_value=mock_agent)),
            Mock(
                filter=Mock(
                    return_value=Mock(first=Mock(return_value=mock_metrics))
                )
            ),
            Mock(
                filter=Mock(
                    return_value=Mock(scalar=Mock(return_value=50))
                )
            ),
        ]

        result = self.repository.get_agent_performance_metrics("test_agent_1")

        self.assertEqual(result["survival_time"], 100)
        self.assertEqual(result["peak_health"], 100.0)
        self.assertEqual(result["peak_resources"], 150.0)
        self.assertEqual(result["total_actions"], 50)

    def test_get_agent_performance_metrics_no_agent(self):
        """Test get_agent_performance_metrics when agent not found."""
        self.mock_session.query.return_value.get.return_value = None

        result = self.repository.get_agent_performance_metrics("nonexistent")

        self.assertEqual(result["survival_time"], 0)
        self.assertEqual(result["peak_health"], 0)
        self.assertEqual(result["peak_resources"], 0)
        self.assertEqual(result["total_actions"], 0)

    def test_get_agent_performance_metrics_alive(self):
        """Test get_agent_performance_metrics for alive agent."""
        mock_agent = Mock(spec=AgentModel)
        mock_agent.birth_time = 100
        mock_agent.death_time = None

        mock_metrics = Mock()
        mock_metrics.peak_health = 100.0
        mock_metrics.peak_resources = 150.0

        # Setup query chain
        self.mock_session.query.side_effect = [
            Mock(get=Mock(return_value=mock_agent)),
            Mock(
                filter=Mock(
                    return_value=Mock(first=Mock(return_value=mock_metrics))
                )
            ),
            Mock(
                filter=Mock(
                    return_value=Mock(scalar=Mock(return_value=30))
                )
            ),
        ]

        result = self.repository.get_agent_performance_metrics("test_agent_1")

        self.assertIsNone(result["survival_time"])

    def test_get_agent_state_history(self):
        """Test get_agent_state_history returns ordered states."""
        mock_states = [
            Mock(spec=AgentStateModel, step_number=i) for i in range(1, 6)
        ]

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = (
            mock_states
        )
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_state_history("test_agent_1")

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[4].step_number, 5)

    def test_get_agent_children(self):
        """Test get_agent_children finds children by genome pattern."""
        mock_parent = Mock(spec=AgentModel)
        mock_parent.genome_id = "parent_genome"

        mock_child1 = Mock(spec=AgentModel)
        mock_child1.genome_id = "parent_genome:other:1"
        mock_child2 = Mock(spec=AgentModel)
        mock_child2.genome_id = "other:parent_genome:2"

        # Setup query chain
        parent_query = Mock()
        parent_query.filter.return_value.first.return_value = mock_parent

        children_query = Mock()
        children_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_child1,
            mock_child2,
        ]

        self.mock_session.query.side_effect = [parent_query, children_query]

        result = self.repository.get_agent_children("parent_genome")

        self.assertEqual(len(result), 2)

    def test_get_agent_children_no_parent(self):
        """Test get_agent_children when parent not found."""
        parent_query = Mock()
        parent_query.filter.return_value.first.return_value = None
        self.mock_session.query.return_value = parent_query

        result = self.repository.get_agent_children("nonexistent")

        self.assertEqual(len(result), 0)

    def test_get_random_agent_id(self):
        """Test get_random_agent_id returns a random agent ID."""
        mock_agents = [("agent_1",), ("agent_2",), ("agent_3",)]

        mock_query = Mock()
        mock_query.all.return_value = mock_agents
        self.mock_session.query.return_value = mock_query

        # Patch random.choice at the module level where it's used
        with patch("random.choice") as mock_choice:
            mock_choice.return_value = ("agent_2",)
            result = self.repository.get_random_agent_id()

        self.assertEqual(result, "agent_2")

    def test_get_random_agent_id_no_agents(self):
        """Test get_random_agent_id returns None when no agents exist."""
        mock_query = Mock()
        mock_query.all.return_value = []
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_random_agent_id()

        self.assertIsNone(result)

    def test_get_agent_positions_over_time(self):
        """Test get_agent_positions_over_time returns position data."""
        mock_row1 = Mock()
        mock_row1.agent_id = "agent_1"
        mock_row1.step_number = 1
        mock_row1.position_x = 10.0
        mock_row1.position_y = 20.0
        mock_row1.position_z = 0.0

        mock_row2 = Mock()
        mock_row2.agent_id = "agent_1"
        mock_row2.step_number = 2
        mock_row2.position_x = 11.0
        mock_row2.position_y = 21.0
        mock_row2.position_z = None

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_row1,
            mock_row2,
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_positions_over_time()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["agent_id"], "agent_1")
        self.assertEqual(result[0]["position_x"], 10.0)
        self.assertEqual(result[1]["position_z"], 0.0)  # None should become 0.0

    def test_get_agent_trajectories(self):
        """Test get_agent_trajectories returns trajectory data."""
        mock_row = Mock()
        mock_row.agent_id = "agent_1"
        mock_row.step_number = 1
        mock_row.position_x = 10.0
        mock_row.position_y = 20.0
        mock_row.position_z = 0.0

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_row
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_trajectories()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["agent_id"], "agent_1")

    def test_get_agent_trajectories_with_filter(self):
        """Test get_agent_trajectories with agent_ids filter."""
        mock_row = Mock()
        mock_row.agent_id = "agent_1"
        mock_row.step_number = 1
        mock_row.position_x = 10.0
        mock_row.position_y = 20.0
        mock_row.position_z = 0.0

        # Create a proper query chain that returns a list
        # The query goes: query -> filter -> filter -> order_by -> all()
        mock_all = Mock(return_value=[mock_row])
        mock_order_by_obj = Mock(all=mock_all)
        mock_order_by = Mock(return_value=mock_order_by_obj)
        mock_filter2_obj = Mock(order_by=mock_order_by)
        mock_filter2 = Mock(return_value=mock_filter2_obj)
        mock_filter1_obj = Mock(filter=mock_filter2)
        mock_filter1 = Mock(return_value=mock_filter1_obj)
        mock_query = Mock(filter=mock_filter1)
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_agent_trajectories(agent_ids=["agent_1"])

        self.assertEqual(len(result), 1)

    def test_get_location_activity_data(self):
        """Test get_location_activity_data returns aggregated location data."""
        mock_row = Mock()
        mock_row.step_number = 1
        mock_row.position_x = 10.0
        mock_row.position_y = 20.0
        mock_row.position_z = 0.0
        mock_row.agent_count = 3

        mock_query = Mock()
        mock_query.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = [
            mock_row
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_location_activity_data()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["step"], 1)
        self.assertEqual(result[0]["agent_count"], 3)


if __name__ == "__main__":
    unittest.main()

