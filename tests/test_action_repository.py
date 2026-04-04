import unittest
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from farm.database.data_types import AgentActionData
from farm.database.models import ActionModel, AgentModel, Base
from farm.database.repositories.action_repository import ActionRepository
from farm.database.session_manager import SessionManager


class TestActionRepository(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for ActionRepository tests."""
        self.session_manager = Mock(spec=SessionManager)
        self.repository = ActionRepository(self.session_manager)
        self.mock_session = Mock(spec=Session)

    def test_get_actions_by_scope_basic(self):
        """Test basic action retrieval by scope."""
        # Arrange
        mock_actions = [
            ActionModel(
                agent_id=1,
                action_type="move",
                step_number=1,
                action_target_id=None,
                reward=0.5,
                details={"direction": "north"},
            )
        ]

        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = (
            mock_actions
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_scope("episode", agent_id="1")

        # Assert
        self.assertEqual(len(result), 1)
        action = result[0]
        self.assertEqual(action.agent_id, 1)
        self.assertEqual(action.action_type, "move")
        self.assertEqual(action.step_number, 1)
        self.assertEqual(action.details, {"direction": "north"})

    def test_get_actions_by_scope_with_step_range(self):
        """Test action retrieval with step range filtering."""
        # Arrange
        step_range = (1, 3)
        mock_actions = [
            ActionModel(agent_id=1, step_number=2, action_type="move"),
            ActionModel(agent_id=1, step_number=3, action_type="gather"),
        ]

        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = (
            mock_actions
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_scope(
            "episode", agent_id="1", step_range=step_range
        )

        # Assert
        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(
                step_range[0] <= action.step_number <= step_range[1]
                for action in result
            )
        )

    def test_get_actions_by_scope_empty_result(self):
        """Test handling of empty result sets."""
        # Arrange
        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = (
            []
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_scope("episode")

        # Assert
        self.assertEqual(len(result), 0)

    def test_get_actions_by_scope_with_specific_step(self):
        """Test action retrieval for a specific step."""
        # Arrange
        specific_step = 5
        mock_action = ActionModel(
            agent_id="1", step_number=specific_step, action_type="gather"
        )

        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = [
            mock_action
        ]
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_scope("episode", step=specific_step)

        # Assert
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].step_number, specific_step)

    def test_get_actions_by_scope_query_construction(self):
        """Test proper construction of the database query."""
        # Arrange
        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = (
            []
        )
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        self.repository.get_actions_by_scope("episode")

        # Assert
        self.mock_session.query.assert_called_once_with(ActionModel)
        # Verify join was called once
        self.assertEqual(
            self.mock_session.query.return_value.join.call_count,
            1,
            "join() should be called exactly once",
        )

        # Verify join arguments
        join_args = self.mock_session.query.return_value.join.call_args[0]
        self.assertEqual(
            join_args[0], AgentModel, "First join argument should be AgentModel"
        )
        # For the second argument (the ON clause), we can only verify it's a BinaryExpression
        join_condition = str(join_args[1]).lower()
        self.assertTrue(
            "agent_id" in join_condition and "=" in join_condition,
            f"Join condition '{join_condition}' should compare agent_ids",
        )

        self.mock_session.query.return_value.join.return_value.order_by.assert_called_once_with(
            ActionModel.step_number, ActionModel.agent_id
        )

    def test_get_actions_by_scope_data_conversion(self):
        """Test conversion of database models to data types."""
        # Arrange
        mock_action = ActionModel(
            agent_id="1",
            action_type="move",
            step_number=1,
            action_target_id=2,
            reward=5.0,
            details={"distance": 10},
        )

        self.mock_session.query.return_value.join.return_value.order_by.return_value.all.return_value = [
            mock_action
        ]
        self.session_manager.execute_with_retry.side_effect = lambda x: x(
            self.mock_session
        )

        # Act
        result = self.repository.get_actions_by_scope("episode")

        # Assert
        self.assertEqual(len(result), 1)
        action_data = result[0]
        self.assertIsInstance(action_data, AgentActionData)
        self.assertEqual(action_data.agent_id, mock_action.agent_id)
        self.assertEqual(action_data.action_type, mock_action.action_type)
        self.assertEqual(action_data.details, mock_action.details)


class _SessionManagerForTest:
    """Minimal stand-in: run repository callbacks on a shared in-memory engine."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def execute_with_retry(self, operation, max_retries=3):
        sess = self._session_factory()
        try:
            result = operation(sess)
            sess.commit()
            return result
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()


class TestActionRepositoryFilterScopeIntegration(unittest.TestCase):
    """Exercise filter_scope against a real query (entity step_number resolution)."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.repository = ActionRepository(_SessionManagerForTest(self.SessionFactory))

        sess = self.SessionFactory()
        sess.add(
            AgentModel(
                agent_id="a1",
                birth_time=0,
                agent_type="t",
                position_x=0.0,
                position_y=0.0,
                initial_resources=1.0,
                starting_health=1.0,
            )
        )
        for step, atype in [(1, "move"), (2, "gather"), (3, "move")]:
            sess.add(
                ActionModel(
                    step_number=step,
                    agent_id="a1",
                    action_type=atype,
                )
            )
        sess.commit()
        sess.close()

    def tearDown(self):
        self.engine.dispose()

    def test_get_actions_by_scope_step_filters_sqlalchemy(self):
        result = self.repository.get_actions_by_scope("step", step=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].step_number, 2)
        self.assertEqual(result[0].action_type, "gather")

    def test_get_actions_by_scope_step_range_filters(self):
        result = self.repository.get_actions_by_scope(
            "step_range", step_range=(1, 2)
        )
        self.assertEqual(len(result), 2)
        steps = sorted(a.step_number for a in result)
        self.assertEqual(steps, [1, 2])


if __name__ == "__main__":
    unittest.main()
