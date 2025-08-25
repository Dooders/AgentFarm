import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from farm.database.models import AgentModel
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.repositories.base_repository import BaseRepository
from farm.database.repositories.learning_repository import LearningRepository
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.repositories.simulation_repository import SimulationRepository
from farm.database.session_manager import SessionManager


class TestBaseRepository(unittest.TestCase):
    def setUp(self):
        self.session_manager = MagicMock(spec=SessionManager)
        self.session = MagicMock(spec=Session)
        self.session_manager.execute_with_retry.side_effect = lambda x: x(self.session)
        self.model = AgentModel
        self.repository = BaseRepository(self.session_manager, self.model)

    def test_add(self):
        entity = self.model()
        self.repository.add(entity)
        self.session.add.assert_called_once_with(entity)

    def test_get_by_id(self):
        entity_id = 1
        mock_entity = MagicMock()
        self.session.query.return_value.get.return_value = mock_entity
        result = self.repository.get_by_id(entity_id)
        self.session.query.assert_called_once_with(self.model)
        self.session.query.return_value.get.assert_called_once_with(entity_id)
        self.assertEqual(result, mock_entity)

    def test_update(self):
        entity = self.model()
        self.repository.update(entity)
        self.session.merge.assert_called_once_with(entity)

    def test_delete(self):
        entity = self.model()
        self.repository.delete(entity)
        self.session.delete.assert_called_once_with(entity)

    def test_execute_in_transaction(self):
        def func(session):
            return "result"

        result = self.repository._execute_in_transaction(func)
        self.assertEqual(result, "result")

    def test_execute_in_transaction_rollback_on_error(self):
        def func(session):
            raise SQLAlchemyError("error")

        with self.assertRaises(SQLAlchemyError):
            self.repository._execute_in_transaction(func)


class TestAgentRepository(unittest.TestCase):
    def setUp(self):
        self.session_manager = MagicMock(spec=SessionManager)
        self.repository = AgentRepository(self.session_manager)

    def test_get_agent_by_id(self):
        mock_agent = MagicMock()
        self.session_manager.execute_with_retry.return_value = mock_agent
        result = self.repository.get_agent_by_id("test_agent")
        self.assertIsNotNone(result)

    def test_get_actions_by_agent_id(self):
        mock_actions = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_actions
        result = self.repository.get_actions_by_agent_id("test_agent")
        self.assertIsNotNone(result)

    def test_get_states_by_agent_id(self):
        mock_states = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_states
        result = self.repository.get_states_by_agent_id("test_agent")
        self.assertIsNotNone(result)


class TestLearningRepository(unittest.TestCase):
    def setUp(self):
        self.session_manager = MagicMock(spec=SessionManager)
        self.repository = LearningRepository(self.session_manager)
        self.mock_session = MagicMock(spec=Session)

    def test_get_learning_progress(self):
        mock_progress = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_progress
        result = self.repository.get_learning_progress(self.mock_session, "simulation")
        self.assertIsNotNone(result)

    def test_get_module_performance(self):
        mock_performance = {"module1": MagicMock()}
        self.session_manager.execute_with_retry.return_value = mock_performance
        result = self.repository.get_module_performance(self.mock_session, "simulation")
        self.assertIsNotNone(result)

    def test_get_agent_learning_stats(self):
        mock_stats = {"module1": MagicMock()}
        self.session_manager.execute_with_retry.return_value = mock_stats
        result = self.repository.get_agent_learning_stats(self.mock_session)
        self.assertIsNotNone(result)

    def test_get_learning_experiences(self):
        mock_experiences = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_experiences
        result = self.repository.get_learning_experiences(
            self.mock_session, "simulation"
        )
        self.assertIsNotNone(result)


class TestPopulationRepository(unittest.TestCase):
    def setUp(self):
        self.session_manager = MagicMock(spec=SessionManager)
        self.repository = PopulationRepository(self.session_manager)
        self.mock_session = MagicMock(spec=Session)

    def test_get_population_data(self):
        mock_data = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_data
        result = self.repository.get_population_data(self.mock_session, "simulation")
        self.assertIsNotNone(result)

    def test_get_agent_type_distribution(self):
        mock_distribution = MagicMock()
        self.session_manager.execute_with_retry.return_value = mock_distribution
        result = self.repository.get_agent_type_distribution(
            self.mock_session, "simulation"
        )
        self.assertIsNotNone(result)

    def test_get_states(self):
        mock_states = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_states
        result = self.repository.get_states("simulation")
        self.assertIsNotNone(result)

    def test_get_all_agents(self):
        mock_agents = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_agents
        result = self.repository.get_all_agents()
        self.assertIsNotNone(result)


class TestSimulationRepository(unittest.TestCase):
    def setUp(self):
        self.session_manager = MagicMock(spec=SessionManager)
        self.repository = SimulationRepository(self.session_manager)

    def test_agent_states(self):
        mock_states = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_states
        result = self.repository.agent_states()
        self.assertIsNotNone(result)

    def test_resource_states(self):
        mock_states = [MagicMock(), MagicMock()]
        self.session_manager.execute_with_retry.return_value = mock_states
        result = self.repository.resource_states(step_number=1)
        self.assertIsNotNone(result)

    def test_simulation_state(self):
        mock_state = MagicMock()
        self.session_manager.execute_with_retry.return_value = mock_state
        result = self.repository.simulation_state(step_number=1)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
