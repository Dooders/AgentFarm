from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from farm.database.data_types import AgentInfo, HealthIncidentData
from farm.database.models import ActionModel, AgentModel, AgentStateModel, HealthIncident
from farm.database.repositories.base_repository import BaseRepository
from farm.database.session_manager import SessionManager


class AgentRepository(BaseRepository[AgentModel]):
    """Repository for handling agent-related data operations.

    This class provides methods to query and retrieve agents and their related data
    such as actions and states.

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

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentModel]:
        """Retrieve an agent by their unique identifier.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        Optional[AgentModel]
            The agent if found, None otherwise
            Fields:
            - agent_id: str (primary key)
            - birth_time: int
            - death_time: Optional[int]
            - agent_type: str
            - position_x: float
            - position_y: float
            - initial_resources: float
            - starting_health: float
            - starvation_threshold: int
            - genome_id: str
            - generation: int

            Relationships:
            - states: List[AgentStateModel]
            - actions: List[ActionModel]
            - health_incidents: List[HealthIncident]
            - learning_experiences: List[LearningExperience]
            - targeted_actions: List[ActionModel]
        """
        def query_agent(session: Session) -> Optional[AgentModel]:
            # Get agent with all relationships loaded
            agent = (
                session.query(AgentModel)
                .options(
                    joinedload(AgentModel.states),
                    joinedload(AgentModel.actions),
                    joinedload(AgentModel.health_incidents)
                )
                .get(agent_id)
            )
            return agent

        return self.session_manager.execute_with_retry(query_agent)

    def get_actions_by_agent_id(self, agent_id: str) -> List[ActionModel]:
        """Retrieve actions by agent ID.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[ActionModel]
            List of actions performed by the agent
            Fields:
            - action_id: int (primary key)
            - step_number: int
            - agent_id: str
            - action_type: str
            - action_target_id: Optional[str]
            - state_before_id: Optional[str]
            - state_after_id: Optional[str]
            - resources_before: float
            - resources_after: float
            - reward: float
            - details: Optional[str]

            Relationships:
            - agent: AgentModel
            - state_before: Optional[AgentStateModel]
            - state_after: Optional[AgentStateModel]
        """

        def query_actions(session: Session) -> List[ActionModel]:
            return (
                session.query(ActionModel)
                .filter(ActionModel.agent_id == agent_id)
                .all()
            )

        return self.session_manager.execute_with_retry(query_actions)

    def get_states_by_agent_id(self, agent_id: str) -> List[AgentStateModel]:
        """Retrieve states by agent ID.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[AgentStateModel]
            List of states associated with the agent
            Fields:
            - id: str (primary key, format: "agent_id-step_number")
            - step_number: int
            - agent_id: str
            - position_x: float
            - position_y: float
            - position_z: float
            - resource_level: float
            - current_health: float
            - is_defending: bool
            - total_reward: float
            - age: int

            Relationships:
            - agent: AgentModel
        """

        def query_states(session: Session) -> List[AgentStateModel]:
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .all()
            )

        return self.session_manager.execute_with_retry(query_states)

    def get_health_incidents_by_agent_id(
        self, agent_id: str
    ) -> List[HealthIncidentData]:
        """Retrieve health incident history for a specific agent.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[HealthIncidentData]
            List of health incidents, each containing:
            - step: Simulation step when incident occurred
            - health_before: Health value before incident
            - health_after: Health value after incident
            - cause: Reason for health change
            - details: Additional incident-specific information
        """

        def query_incidents(session: Session) -> List[HealthIncidentData]:
            incidents = (
                session.query(HealthIncident)
                .filter(HealthIncident.agent_id == agent_id)
                .order_by(HealthIncident.step_number)
                .all()
            )

            return [
                HealthIncidentData(
                    step=incident.step_number,
                    health_before=incident.health_before,
                    health_after=incident.health_after,
                    cause=incident.cause,
                    details=incident.details,
                )
                for incident in incidents
            ]

        return self.session_manager.execute_with_retry(query_incidents)

    def get_ordered_actions(self, agent_id: Optional[int] = None) -> List[ActionModel]:
        """Get actions ordered by agent_id and step_number.

        Retrieves a chronologically ordered list of actions, optionally filtered by agent.

        Args:
            agent_id (Optional[int]): If provided, only returns actions for this agent.

        Returns:
            List[ActionModel]: Ordered list of actions containing:
                - action_id: Unique identifier
                - step_number: Simulation step when action occurred
                - agent_id: ID of agent performing action
                - action_type: Type of action performed
                - action_target_id: ID of target agent (if any)
                - reward: Reward received for action
                - details: Additional action-specific information

        Examples:
            >>> actions = repository.get_ordered_actions(agent_id=1)
            >>> print(f"First action: {actions[0].action_type}")
            >>> print(f"Total actions: {len(actions)}")
        """
        query = self.session.query(ActionModel).order_by(
            ActionModel.agent_id, ActionModel.step_number
        )
        if agent_id:
            query = query.filter(ActionModel.agent_id == agent_id)
        return query.all()

    def get_action_statistics(self, agent_id: Optional[int] = None) -> List[Any]:
        """Get statistical aggregates for actions.

        Calculates summary statistics for each action type, including reward variance
        and averages. Results are grouped by action type.

        Args:
            agent_id (Optional[int]): If provided, only analyzes actions for this agent.

        Returns:
            List[Any]: List of statistics per action type, each containing:
                - action_type: Type of action
                - reward_std: Standard deviation of rewards
                - reward_avg: Average reward
                - count: Number of times action was performed

        Examples:
            >>> stats = repository.get_action_statistics(agent_id=1)
            >>> for stat in stats:
            ...     print(f"{stat.action_type}: avg={stat.reward_avg:.2f}, "
            ...           f"std={stat.reward_std:.2f}, n={stat.count}")
        """
        query = self.session.query(
            ActionModel.action_type,
            func.stddev(ActionModel.reward).label("reward_std"),
            func.avg(ActionModel.reward).label("reward_avg"),
            func.count().label("count"),
        ).group_by(ActionModel.action_type)

        if agent_id:
            query = query.filter(ActionModel.agent_id == agent_id)

        return query.all()

    def get_actions_with_states(
        self, agent_id: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """Get actions with their corresponding states.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of tuples containing (action, state) pairs
        """
        query = self.session.query(ActionModel, AgentStateModel).join(
            AgentStateModel, ActionModel.state_before_id == AgentStateModel.id
        )

        if agent_id:
            query = query.filter(ActionModel.agent_id == agent_id)

        return query.all()

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get comprehensive information about an agent.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            AgentInfo object containing agent details, or None if agent not found
        """
        def query_agent(session: Session) -> Optional[AgentInfo]:
            agent = session.query(AgentModel).get(agent_id)
            if not agent:
                return None

            # Get latest state
            latest_state = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number.desc())
                .first()
            )

            # Get action statistics
            action_stats = (
                session.query(
                    ActionModel.action_type,
                    func.count().label('count'),
                    func.avg(ActionModel.reward).label('avg_reward')
                )
                .filter(ActionModel.agent_id == agent_id)
                .group_by(ActionModel.action_type)
                .all()
            )

            return AgentInfo(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                birth_time=agent.birth_time,
                death_time=agent.death_time,
                generation=agent.generation,
                genome_id=agent.genome_id,
                current_health=latest_state.current_health if latest_state else None,
                current_resources=latest_state.resource_level if latest_state else None,
                position=(
                    latest_state.position_x,
                    latest_state.position_y
                ) if latest_state else None,
                action_stats={
                    stat.action_type: {
                        'count': stat.count,
                        'avg_reward': stat.avg_reward
                    } for stat in action_stats
                }
            )

        return self.session_manager.execute_with_retry(query_agent)
