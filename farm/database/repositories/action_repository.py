from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from database.data_types import AgentActionData
from database.models import ActionModel
from database.repositories.base_repository import BaseRepository
from database.scope_utils import filter_scope
from database.session_manager import SessionManager


class ActionRepository(BaseRepository[ActionModel]):
    """Repository class for managing agent action records in the database.

    This class provides methods to query and retrieve agent actions based on various
    filtering criteria such as scope, agent ID, and step numbers.

    Args:
        session_manager (SessionManager): Session manager for database operations.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize repository with session manager.

        Parameters
        ----------
        session_manager : SessionManager
            Session manager instance for database operations
        """
        self.session_manager = session_manager

    def get_actions_by_scope(
        self,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        """Retrieve agent actions filtered by scope and other optional parameters.

        Parameters
        ----------
        scope : str
            The scope to filter actions by (e.g., 'episode', 'experiment')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        step : Optional[int], optional
            Specific step number to filter by. Defaults to None.
        step_range : Optional[Tuple[int, int]], optional
            Range of step numbers to filter by. Defaults to None.

        Returns
        -------
        List[AgentActionData]
            List of agent actions matching the specified criteria,
            ordered by step number and agent ID.
        """

        def query_actions(session: Session) -> List[AgentActionData]:
            query = session.query(ActionModel).order_by(
                ActionModel.step_number, ActionModel.agent_id
            )

            query = filter_scope(query, scope, agent_id, step, step_range)

            actions = query.all()
            return [
                AgentActionData(
                    agent_id=action.agent_id,
                    action_type=action.action_type,
                    step_number=action.step_number,
                    action_target_id=action.action_target_id,
                    resources_before=action.resources_before,
                    resources_after=action.resources_after,
                    state_before_id=action.state_before_id,
                    state_after_id=action.state_after_id,
                    reward=action.reward,
                    details=action.details if action.details else None,
                )
                for action in actions
            ]

        return self.session_manager.execute_with_retry(query_actions)
