import json
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func
from sqlalchemy.sql.functions import Function
from sqlalchemy.orm import Session, joinedload


from farm.database.data_types import AgentInfo, HealthIncidentData
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
)
from farm.database.repositories.base_repository import BaseRepository
from farm.database.session_manager import SessionManager
from farm.database.utilities import safe_json_loads


class AgentRepository(BaseRepository[AgentModel]):
    # Type hint to help linter understand func object
    _func = func
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
        super().__init__(session_manager, AgentModel)
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
            - starvation_counter: int
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
                    joinedload(AgentModel.health_incidents),
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

            result = []
            for incident in incidents:
                # Extract values to satisfy type checker
                step = getattr(incident, "step_number")
                health_before = getattr(incident, "health_before")
                health_after = getattr(incident, "health_after")
                cause = getattr(incident, "cause")
                details_str = getattr(incident, "details", None)

                details_dict = safe_json_loads(details_str) if details_str else {}
                if details_dict is None:
                    details_dict = {}

                result.append(
                    HealthIncidentData(
                        step=step,
                        health_before=health_before,
                        health_after=health_after,
                        cause=cause,
                        details=details_dict,
                    )
                )
            return result

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

        def query_ordered_actions(session: Session) -> List[ActionModel]:
            query = session.query(ActionModel).order_by(
                ActionModel.agent_id, ActionModel.step_number
            )
            if agent_id:
                query = query.filter(ActionModel.agent_id == agent_id)
            return query.all()

        return self.session_manager.execute_with_retry(query_ordered_actions)

    def get_action_statistics(self, agent_id: Optional[int] = None) -> List[Any]:
        """Get statistical aggregates for actions.

        Args:
            agent_id (Optional[int]): If provided, only analyzes actions for this agent.

        Returns:
            List[Any]: List of statistics per action type containing:
                - action_type: Type of action
                - reward_std: Standard deviation of rewards
                - reward_avg: Average reward
                - count: Number of times action was performed
        """

        def query_action_stats(session: Session) -> List[Any]:
            # First get all actions with rewards
            query = session.query(
                ActionModel.action_type,
                func.avg(ActionModel.reward).label("reward_avg"),
                func.count(ActionModel.action_id).label("count"),
                func.group_concat(ActionModel.reward).label("rewards"),
            ).group_by(ActionModel.action_type)

            if agent_id:
                query = query.filter(ActionModel.agent_id == agent_id)

            results = query.all()

            # Calculate standard deviation manually
            final_stats = []
            for result in results:
                if result.rewards:
                    # Convert string of rewards back to list of floats
                    rewards = [float(r) for r in result.rewards.split(",") if r]
                    # Calculate standard deviation
                    if len(rewards) > 1:
                        mean = sum(rewards) / len(rewards)
                        variance = sum((x - mean) ** 2 for x in rewards) / (
                            len(rewards) - 1
                        )
                        std_dev = variance**0.5
                    else:
                        std_dev = 0.0
                else:
                    std_dev = 0.0

                # Create a new result object with std dev
                final_stats.append(
                    type(
                        "ActionStat",
                        (),
                        {
                            "action_type": result.action_type,
                            "reward_std": std_dev,
                            "reward_avg": result.reward_avg,
                            "count": result.count,
                        },
                    )
                )

            return final_stats

        return self.session_manager.execute_with_retry(query_action_stats)

    def get_actions_with_states(
        self, agent_id: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """Get actions with their corresponding states.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of tuples containing (action, state) pairs
        """

        def query_actions_states(session: Session) -> List[Tuple[Any, Any]]:
            # Since state_before_id was removed, we need to join on agent_id and step_number
            # to find the corresponding state for each action
            q = session.query(ActionModel, AgentStateModel).join(
                AgentStateModel, 
                and_(
                    ActionModel.agent_id == AgentStateModel.agent_id,
                    ActionModel.step_number == AgentStateModel.step_number
                )
            )
            if agent_id:
                q = q.filter(ActionModel.agent_id == agent_id)
            return [tuple(row) for row in q.all()]

        return self.session_manager.execute_with_retry(query_actions_states)

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
                    func.count(ActionModel.action_id).label("count"),
                    func.avg(ActionModel.reward).label("avg_reward"),
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
                current_health=(
                    getattr(latest_state, "current_health", None)
                    if latest_state
                    else None
                ),
                current_resources=(
                    getattr(latest_state, "resource_level", None)
                    if latest_state
                    else None
                ),
                position=(
                    (
                        getattr(latest_state, "position_x", 0),
                        getattr(latest_state, "position_y", 0),
                    )
                    if latest_state
                    else None
                ),
                action_stats={
                    str(stat[0]): {
                        "count": float(stat[1]),
                        "avg_reward": float(stat[2]) if stat[2] else 0.0,
                    }
                    for stat in action_stats
                },
            )

        return self.session_manager.execute_with_retry(query_agent)

    def get_agent_current_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get current statistics for an agent.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current agent statistics:
            - health: Current health value
            - resources: Current resource level
            - total_reward: Total accumulated reward
            - age: Current age
            - is_defending: Current defense status
            - current_position: Tuple of (x, y) coordinates
        """

        def query_stats(session: Session) -> Dict[str, Any]:
            # Get the latest state for the agent
            latest_state = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number.desc())
                .first()
            )

            if not latest_state:
                return {
                    "health": 0,
                    "resources": 0,
                    "total_reward": 0,
                    "age": 0,
                    "is_defending": False,
                    "current_position": (0, 0),
                }

            return {
                "health": latest_state.current_health,
                "resources": latest_state.resource_level,
                "total_reward": latest_state.total_reward,
                "age": latest_state.age,
                "is_defending": latest_state.is_defending,
                "current_position": (latest_state.position_x, latest_state.position_y),
            }

        return self.session_manager.execute_with_retry(query_stats)

    def get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance metrics:
            - survival_time: Total time agent has survived
            - peak_health: Highest health value achieved
            - peak_resources: Maximum resources accumulated
            - total_actions: Number of actions taken
        """

        def query_metrics(session: Session) -> Dict[str, Any]:
            # Get agent birth and death time
            agent = session.query(AgentModel).get(agent_id)
            if not agent:
                return {
                    "survival_time": 0,
                    "peak_health": 0,
                    "peak_resources": 0,
                    "total_actions": 0,
                }

            # Calculate survival time
            survival_time = (agent.death_time or float("inf")) - agent.birth_time

            # Get peak health and resources from states
            states_metrics = (
                session.query(
                    func.max(AgentStateModel.current_health).label("peak_health"),
                    func.max(AgentStateModel.resource_level).label("peak_resources"),
                )
                .filter(AgentStateModel.agent_id == agent_id)
                .first()
            )

            # Get total number of actions
            total_actions = (
                session.query(func.count(ActionModel.action_id))
                .filter(ActionModel.agent_id == agent_id)
                .scalar()
            )

            return {
                "survival_time": (
                    survival_time if survival_time != float("inf") else None
                ),
                "peak_health": states_metrics.peak_health if states_metrics else 0,
                "peak_resources": (
                    states_metrics.peak_resources if states_metrics else 0
                ),
                "total_actions": total_actions or 0,
            }

        return self.session_manager.execute_with_retry(query_metrics)

    def get_agent_state_history(self, agent_id: str) -> List[AgentStateModel]:
        """Get the complete state history for an agent.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[AgentStateModel]
            List of agent states ordered by step number, containing:
            - step_number: Simulation step
            - current_health: Health value at that step
            - resource_level: Resource amount at that step
            - total_reward: Accumulated reward at that step
            - age: Agent age at that step
            - position_x, position_y: Position coordinates
            - is_defending: Defense status
        """

        def query_history(session: Session) -> List[AgentStateModel]:
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number)
                .all()
            )

        return self.session_manager.execute_with_retry(query_history)

    def get_agent_children(self, agent_id: str) -> List[AgentModel]:
        """Get all children of an agent using genome information.

        Args:
            agent_id: ID of the parent agent

        Returns:
            List of child agents
        """

        def query_children(session: Session) -> List[AgentModel]:
            # Get the parent agent's genome ID
            parent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )
            if not parent:
                return []

            # Find children by checking genome parent IDs
            children = (
                session.query(AgentModel)
                .filter(
                    AgentModel.genome_id.like(f"%{agent_id}%")
                )  # Check if parent ID is in genome
                .order_by(AgentModel.birth_time)
                .all()
            )

            return children

        return self.session_manager.execute_with_retry(query_children)

    def get_random_agent_id(self) -> Optional[str]:
        """Get a random agent ID from the database.

        Returns
        -------
        Optional[str]
            A random agent ID if agents exist, None otherwise
        """

        def query_random_agent(session: Session) -> Optional[str]:
            # Get all agent IDs and select one randomly
            import random
            all_agents = session.query(AgentModel.agent_id).all()
            if not all_agents:
                return None
            random_agent = random.choice(all_agents)
            return random_agent[0]

        return self.session_manager.execute_with_retry(query_random_agent)

    def get_agent_positions_over_time(self) -> List[Dict[str, Any]]:
        """Get agent positions over time for spatial analysis.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing agent position data with keys:
            agent_id, step, position_x, position_y, position_z
        """

        def query_positions(session: Session) -> List[Dict[str, Any]]:
            # Query agent states with position information
            result = session.query(
                AgentStateModel.agent_id,
                AgentStateModel.step_number,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentStateModel.position_z
            ).filter(
                AgentStateModel.position_x.isnot(None),
                AgentStateModel.position_y.isnot(None)
            ).order_by(
                AgentStateModel.agent_id,
                AgentStateModel.step_number
            ).all()

            return [
                {
                    'agent_id': row.agent_id,
                    'step': row.step_number,
                    'position_x': row.position_x,
                    'position_y': row.position_y,
                    'position_z': row.position_z or 0.0
                }
                for row in result
            ]

        return self.session_manager.execute_with_retry(query_positions)

    def get_agent_trajectories(self, agent_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get agent movement trajectories for spatial analysis.

        Parameters
        ----------
        agent_ids : Optional[List[int]]
            Specific agent IDs to analyze. If None, analyze all agents.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing trajectory data
        """

        def query_trajectories(session: Session) -> List[Dict[str, Any]]:
            query = session.query(
                AgentStateModel.agent_id,
                AgentStateModel.step_number,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentStateModel.position_z
            ).filter(
                AgentStateModel.position_x.isnot(None),
                AgentStateModel.position_y.isnot(None)
            )

            if agent_ids:
                query = query.filter(AgentStateModel.agent_id.in_(agent_ids))

            result = query.order_by(
                AgentStateModel.agent_id,
                AgentStateModel.step_number
            ).all()

            return [
                {
                    'agent_id': row.agent_id,
                    'step': row.step_number,
                    'position_x': row.position_x,
                    'position_y': row.position_y,
                    'position_z': row.position_z or 0.0
                }
                for row in result
            ]

        return self.session_manager.execute_with_retry(query_trajectories)

    def get_location_activity_data(self) -> List[Dict[str, Any]]:
        """Get location-based activity data for spatial analysis.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing location activity data
        """

        def query_location_activity(session: Session) -> List[Dict[str, Any]]:
            # Get agent positions and activity data
            result = session.query(
                AgentStateModel.step_number,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentStateModel.position_z,
                func.count(AgentStateModel.agent_id).label('agent_count')
            ).filter(
                AgentStateModel.position_x.isnot(None),
                AgentStateModel.position_y.isnot(None)
            ).group_by(
                AgentStateModel.step_number,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentStateModel.position_z
            ).order_by(
                AgentStateModel.step_number
            ).all()

            return [
                {
                    'step': row.step_number,
                    'position_x': row.position_x,
                    'position_y': row.position_y,
                    'position_z': row.position_z or 0.0,
                    'agent_count': row.agent_count
                }
                for row in result
            ]

        return self.session_manager.execute_with_retry(query_location_activity)
