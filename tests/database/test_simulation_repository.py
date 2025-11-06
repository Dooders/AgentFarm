"""Comprehensive tests for SimulationRepository with mocked sessions."""

import unittest
from unittest.mock import Mock

from farm.database.data_types import (
    AgentStates,
    ResourceStates,
    SimulationResults,
    SimulationState,
)
from farm.database.models import (
    AgentModel,
    AgentStateModel,
    ResourceModel,
    SimulationStepModel,
)
from farm.database.repositories.simulation_repository import SimulationRepository


class TestSimulationRepository(unittest.TestCase):
    """Comprehensive tests for SimulationRepository."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = Mock()
        self.repository = SimulationRepository(self.mock_db)
        self.mock_session = Mock()

        def execute_in_transaction_side_effect(func):
            return func(self.mock_session)

        self.repository._execute_in_transaction = execute_in_transaction_side_effect

    def test_agent_states_with_step(self):
        """Test agent_states with specific step number."""
        mock_row = (1, "agent_1", "hunter", 10.0, 20.0, 50.0, 80.0, False)

        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository.agent_states(step_number=1)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AgentStates)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[0].agent_id, "agent_1")

    def test_agent_states_all_steps(self):
        """Test agent_states without step filter."""
        mock_row1 = (1, "agent_1", "hunter", 10.0, 20.0, 50.0, 80.0, False)
        mock_row2 = (2, "agent_1", "hunter", 11.0, 21.0, 55.0, 85.0, False)

        mock_query = Mock()
        mock_query.join.return_value.order_by.return_value.all.return_value = [
            mock_row1,
            mock_row2,
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.agent_states()

        self.assertEqual(len(result), 2)

    def test_resource_states(self):
        """Test resource_states returns resource data."""
        mock_row = ("resource_1", 100.0, 15.0, 25.0)

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository.resource_states(step_number=1)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ResourceStates)
        self.assertEqual(result[0].resource_id, "resource_1")
        self.assertEqual(result[0].amount, 100.0)

    def test_simulation_state(self):
        """Test simulation_state returns state data."""
        mock_step = Mock(spec=SimulationStepModel)
        # as_dict() should return all fields that SimulationState expects
        mock_step.as_dict.return_value = {
            "total_agents": 10,
            "system_agents": 5,
            "independent_agents": 3,
            "control_agents": 2,
            "total_resources": 1000.0,
            "average_agent_resources": 100.0,
            "births": 2,
            "deaths": 1,
            "current_max_generation": 1,
            "resource_efficiency": 0.8,
            "resource_distribution_entropy": 0.5,
            "average_agent_health": 80.0,
            "average_agent_age": 10,
            "average_reward": 0.5,
            "combat_encounters": 5,
            "successful_attacks": 3,
            "resources_shared": 50.0,
            "genetic_diversity": 0.7,
            "dominant_genome_ratio": 0.6,
            "resources_consumed": 200.0,
        }

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_step
        self.mock_session.query.return_value = mock_query

        result = self.repository.simulation_state(step_number=1)

        self.assertIsInstance(result, SimulationState)
        self.assertEqual(result.total_agents, 10)

    def test_agent_states_tuples(self):
        """Test _agent_states_tuples returns tuples."""
        mock_row = (1, "agent_1", "hunter", 10.0, 20.0, 50.0, 80.0, False)

        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository._agent_states_tuples(step_number=1)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(result[0][0], 1)

    def test_resource_states_tuples(self):
        """Test _resource_states_tuples returns tuples."""
        mock_row = ("resource_1", 100.0, 15.0, 25.0)

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_row]
        self.mock_session.query.return_value = mock_query

        result = self.repository._resource_states_tuples(step_number=1)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(result[0][0], "resource_1")

    def test_execute(self):
        """Test execute returns complete simulation results."""
        # Create a proper SimulationState dataclass instance
        from farm.database.data_types import SimulationState
        
        mock_simulation_state = SimulationState(
            total_agents=10,
            system_agents=5,
            independent_agents=3,
            control_agents=2,
            total_resources=1000.0,
            average_agent_resources=100.0,
            births=2,
            deaths=1,
            current_max_generation=1,
            resource_efficiency=0.8,
            resource_distribution_entropy=0.5,
            average_agent_health=80.0,
            average_agent_age=10,
            average_reward=0.5,
            combat_encounters=5,
            successful_attacks=3,
            resources_shared=50.0,
            genetic_diversity=0.7,
            dominant_genome_ratio=0.6,
            resources_consumed=200.0,
        )

        # Mock agent tuples
        mock_agent_tuple = (1, "agent_1", "hunter", 10.0, 20.0, 50.0, 80.0, False)

        # Mock resource tuples
        mock_resource_tuple = ("resource_1", 100.0, 15.0, 25.0)

        # Setup mocks
        self.repository.simulation_state = Mock(
            return_value=mock_simulation_state
        )
        self.repository._agent_states_tuples = Mock(
            return_value=[mock_agent_tuple]
        )
        self.repository._resource_states_tuples = Mock(
            return_value=[mock_resource_tuple]
        )

        result = self.repository.execute(step_number=1)

        self.assertIsInstance(result, SimulationResults)
        self.assertEqual(len(result.agent_states), 1)
        self.assertEqual(len(result.resource_states), 1)
        # execute() returns asdict(simulation_state), so it's a dict, not SimulationState
        self.assertIsInstance(result.simulation_state, dict)
        self.assertEqual(result.simulation_state["total_agents"], 10)


if __name__ == "__main__":
    unittest.main()

