"""Comprehensive tests for PopulationRepository with mocked sessions."""

import unittest
from unittest.mock import Mock, patch

from sqlalchemy.orm import Session

from farm.database.data_types import (
    AgentDistribution,
    AgentEvolutionMetrics,
    AgentStates,
    Population,
)
from farm.database.models import AgentModel
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.session_manager import SessionManager


class TestPopulationRepository(unittest.TestCase):
    """Comprehensive tests for PopulationRepository."""

    def setUp(self):
        """Set up test fixtures."""
        self.session_manager = Mock(spec=SessionManager)
        self.repository = PopulationRepository(self.session_manager)
        self.mock_session = Mock(spec=Session)

        def execute_with_retry_side_effect(func):
            return func(self.mock_session)

        self.session_manager.execute_with_retry.side_effect = (
            execute_with_retry_side_effect
        )

    def test_get_population_data(self):
        """Test get_population_data returns population metrics."""
        mock_row = (1, 10, 1000.0, 500.0)  # step, total_agents, total_resources, resources_consumed

        # Create proper query chain that returns a list
        # The query goes: query -> outerjoin -> filter -> group_by -> all()
        # Then filter_scope is called, which returns the query, then .all() is called
        mock_all = Mock(return_value=[mock_row])
        mock_group_by_obj = Mock(all=mock_all)
        mock_group_by = Mock(return_value=mock_group_by_obj)
        mock_filter_obj = Mock(group_by=mock_group_by)
        mock_filter = Mock(return_value=mock_filter_obj)
        mock_outerjoin_obj = Mock(filter=mock_filter)
        mock_outerjoin = Mock(return_value=mock_outerjoin_obj)
        mock_query = Mock(outerjoin=mock_outerjoin)
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter_scope:
            # filter_scope returns a query object, and then .all() is called on it
            mock_filtered_all = Mock(return_value=[mock_row])
            mock_filtered_query = Mock(all=mock_filtered_all)
            mock_filter_scope.return_value = mock_filtered_query
            result = self.repository.get_population_data(
                self.mock_session, scope="episode"
            )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Population)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[0].total_agents, 10)

    def test_get_agent_type_distribution(self):
        """Test get_agent_type_distribution calculates averages."""
        mock_step1 = Mock()
        mock_step1.agent_type_counts = {"system": 5, "independent": 3, "control": 2}

        mock_step2 = Mock()
        mock_step2.agent_type_counts = {"system": 6, "independent": 4, "control": 2}

        mock_query = Mock()
        mock_query.all.return_value = [mock_step1, mock_step2]
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter:
            mock_filter.return_value = mock_query
            result = self.repository.get_agent_type_distribution(
                self.mock_session, scope="episode"
            )

        self.assertIsInstance(result, AgentDistribution)
        self.assertEqual(result.system_agents, 5.5)
        self.assertEqual(result.independent_agents, 3.5)
        self.assertEqual(result.control_agents, 2.0)

    def test_get_agent_type_distribution_empty(self):
        """Test get_agent_type_distribution with no data."""
        mock_query = Mock()
        mock_query.all.return_value = []
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter:
            mock_filter.return_value = mock_query
            result = self.repository.get_agent_type_distribution(
                self.mock_session, scope="episode"
            )

        self.assertEqual(result.system_agents, 0.0)
        self.assertEqual(result.independent_agents, 0.0)
        self.assertEqual(result.control_agents, 0.0)

    def test_get_states(self):
        """Test get_states returns agent states."""
        mock_row = (1, "agent_1", "hunter", 10.0, 20.0, 50.0, 80.0, False)

        # Create proper query chain that returns a list
        # The query goes: query -> join -> order_by -> (filter_scope) -> all()
        mock_all = Mock(return_value=[mock_row])
        mock_order_by_obj = Mock(all=mock_all)
        mock_order_by = Mock(return_value=mock_order_by_obj)
        mock_join_obj = Mock(order_by=mock_order_by)
        mock_join = Mock(return_value=mock_join_obj)
        mock_query = Mock(join=mock_join)
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter:
            # filter_scope returns a query object, and then .all() is called on it
            mock_filtered_all = Mock(return_value=[mock_row])
            mock_filtered_query = Mock(all=mock_filtered_all)
            mock_filter.return_value = mock_filtered_query
            result = self.repository.get_states(scope="episode")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AgentStates)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[0].agent_id, "agent_1")

    def test_evolution(self):
        """Test evolution returns evolution metrics."""
        mock_agent1 = Mock(spec=AgentModel)
        mock_agent1.genome_id = "genome_1"
        mock_agent1.birth_time = 100
        mock_agent1.death_time = 200

        mock_agent2 = Mock(spec=AgentModel)
        mock_agent2.genome_id = "genome_2"
        mock_agent2.birth_time = 150
        mock_agent2.death_time = None

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_agent1, mock_agent2]
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter:
            mock_filter.return_value = mock_query
            result = self.repository.evolution(
                self.mock_session, scope="episode", generation=1
            )

        self.assertIsInstance(result, AgentEvolutionMetrics)
        self.assertEqual(result.total_agents, 2)
        self.assertEqual(result.unique_genomes, 2)
        self.assertEqual(result.generation, 1)

    def test_evolution_no_agents(self):
        """Test evolution with no agents."""
        # Create proper query chain that returns an empty list
        # The query goes: query -> (filter_scope returns query) -> all()
        # Since generation is None, no additional filter is called
        mock_query = Mock()
        self.mock_session.query.return_value = mock_query

        with patch("farm.database.repositories.population_repository.filter_scope") as mock_filter_scope:
            # filter_scope returns a query object, then .all() is called directly on it
            mock_all = Mock(return_value=[])
            mock_filtered_query = Mock(all=mock_all)
            mock_filter_scope.return_value = mock_filtered_query
            result = self.repository.evolution(
                self.mock_session, scope="episode"
            )

        self.assertEqual(result.total_agents, 0)
        self.assertEqual(result.unique_genomes, 0)
        self.assertEqual(result.average_lifespan, 0)

    def test_get_population_over_time(self):
        """Test get_population_over_time returns time series."""
        mock_row1 = (1, 10, {"system": 5, "independent": 3, "control": 2}, 1000.0)
        mock_row2 = (2, 12, {"system": 6, "independent": 4, "control": 2}, 1200.0)

        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [
            mock_row1,
            mock_row2,
        ]
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_population_over_time()

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Population)
        self.assertEqual(result[0].step_number, 1)
        self.assertEqual(result[0].system_agents, 5)
        self.assertEqual(result[1].avg_resources, 100.0)  # 1200 / 12

    def test_get_all_agents(self):
        """Test get_all_agents returns all agents ordered by birth_time."""
        mock_agents = [
            Mock(spec=AgentModel, birth_time=200),
            Mock(spec=AgentModel, birth_time=100),
        ]

        mock_query = Mock()
        mock_query.order_by.return_value.all.return_value = mock_agents
        self.mock_session.query.return_value = mock_query

        result = self.repository.get_all_agents()

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

