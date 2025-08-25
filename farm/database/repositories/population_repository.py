from typing import List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from farm.database.data_types import (
    AgentDistribution,
    AgentEvolutionMetrics,
    AgentStates,
    Population,
)
from farm.database.models import AgentModel, AgentStateModel, SimulationStepModel
from farm.database.repositories.base_repository import BaseRepository
from farm.database.scope_utils import filter_scope
from farm.database.session_manager import SessionManager


class PopulationRepository(BaseRepository[SimulationStepModel]):
    """Handles retrieval and analysis of population statistics.

    This class encapsulates methods for analyzing population dynamics, resource utilization,
    and agent distributions across the simulation steps. It provides comprehensive statistics
    about agent populations, resource consumption, and survival metrics.

    Methods
    -------
    get_population_data(session, scope, agent_id=None, step=None, step_range=None)
        Retrieves base population and resource data for each simulation step.

    get_agent_type_distribution(session, scope, agent_id=None, step=None, step_range=None)
        Analyzes the distribution of different agent types (system, independent, control)
        across the simulation.

    get_states(scope, agent_id=None, step=None, step_range=None)
        Retrieves detailed agent states including position, resources, and health,
        filtered by various criteria.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        super().__init__(session_manager, SimulationStepModel)

    def get_population_data(
        self,
        session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[Population]:
        """Retrieve base population and resource data for each simulation step.

        Parameters
        ----------
        scope : str
            The scope to filter data by (e.g., 'episode', 'experiment')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        step : Optional[int], optional
            Specific step number to filter by. Defaults to None.
        step_range : Optional[Tuple[int, int]], optional
            Range of step numbers to filter by. Defaults to None.

        Returns
        -------
        List[Population]
            List of Population objects containing step-wise metrics
        """
        query = (
            session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.total_agents,
                SimulationStepModel.total_resources,
                func.sum(AgentStateModel.resource_level).label("resources_consumed"),
            )
            .outerjoin(
                AgentStateModel,
                AgentStateModel.step_number == SimulationStepModel.step_number,
            )
            .filter(SimulationStepModel.total_agents > 0)
            .group_by(SimulationStepModel.step_number)
        )

        # Apply scope filtering
        query = filter_scope(query, scope, str(agent_id), step, step_range)

        pop_data = query.all()

        return [
            Population(
                step_number=row[0],
                total_agents=row[1],
                total_resources=row[2],
                resources_consumed=row[3],
            )
            for row in pop_data
        ]

    def get_agent_type_distribution(
        self,
        session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> AgentDistribution:
        """Analyze the distribution of different agent types across the simulation.

        Parameters
        ----------
        scope : str
            The scope to filter data by (e.g., 'episode', 'experiment')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        step : Optional[int], optional
            Specific step number to filter by. Defaults to None.
        step_range : Optional[Tuple[int, int]], optional
            Range of step numbers to filter by. Defaults to None.

        Returns
        -------
        AgentDistribution
            Distribution metrics containing:
            - system_agents : float
                Average number of system-controlled agents
            - independent_agents : float
                Average number of independently operating agents
            - control_agents : float
                Average number of control group agents
        """
        query = session.query(
            func.avg(SimulationStepModel.system_agents).label("avg_system"),
            func.avg(SimulationStepModel.independent_agents).label("avg_independent"),
            func.avg(SimulationStepModel.control_agents).label("avg_control"),
        )

        # Apply scope filtering
        query = filter_scope(query, scope, str(agent_id), step, step_range)
        type_stats = query.first()

        return AgentDistribution(
            system_agents=float(
                type_stats[0] if type_stats and type_stats[0] is not None else 0
            ),
            independent_agents=float(
                type_stats[1] if type_stats and type_stats[1] is not None else 0
            ),
            control_agents=float(
                type_stats[2] if type_stats and type_stats[2] is not None else 0
            ),
        )

    def get_states(
        self,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentStates]:
        """Retrieve agent states filtered by scope and other parameters.

        Parameters
        ----------
        scope : str
            The scope to filter states by (e.g., 'episode', 'experiment')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        step : Optional[int], optional
            Specific step number to filter by. Defaults to None.
        step_range : Optional[Tuple[int, int]], optional
            Range of step numbers to filter by. Defaults to None.

        Returns
        -------
        List[AgentStates]
            List of agent states matching the specified criteria,
            ordered by step number and agent ID.
        """

        def query_states(session: Session) -> List[AgentStates]:
            query = (
                session.query(
                    AgentStateModel.step_number,
                    AgentStateModel.agent_id,
                    AgentModel.agent_type,
                    AgentStateModel.position_x,
                    AgentStateModel.position_y,
                    AgentStateModel.resource_level,
                    AgentStateModel.current_health,
                    AgentStateModel.is_defending,
                )
                .join(AgentModel)
                .order_by(AgentStateModel.step_number, AgentStateModel.agent_id)
            )

            query = filter_scope(query, scope, str(agent_id), step, step_range)

            results = query.all()
            return [
                AgentStates(
                    step_number=row[0],
                    agent_id=row[1],
                    agent_type=row[2],
                    position_x=row[3],
                    position_y=row[4],
                    resource_level=row[5],
                    current_health=row[6],
                    is_defending=row[7],
                )
                for row in results
            ]

        return self.session_manager.execute_with_retry(query_states)

    def evolution(
        self,
        session,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        generation: Optional[int] = None,
    ) -> AgentEvolutionMetrics:
        """Get evolution metrics for agents.

        Parameters
        ----------
        scope : str
            The scope to filter data by (e.g., 'episode', 'experiment')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        step : Optional[int], optional
            Specific step number to filter by. Defaults to None.
        step_range : Optional[Tuple[int, int]], optional
            Range of step numbers to filter by. Defaults to None.
        generation : Optional[int], optional
            Specific generation to analyze. If None, analyzes all generations.

        Returns
        -------
        AgentEvolutionMetrics
            Evolution metrics including:
            - total_agents: int
                Number of agents in the generation
            - unique_genomes: int
                Number of distinct genetic configurations
            - average_lifespan: timedelta
                Mean survival duration
            - generation: Optional[int]
                Generation number (None if analyzing all generations)
        """
        query = session.query(AgentModel)

        # Apply scope filtering
        query = filter_scope(query, scope, str(agent_id), step, step_range)

        if generation is not None:
            query = query.filter(AgentModel.generation == generation)

        results = query.all()

        # Calculate metrics
        total_agents = len(results)
        unique_genomes = len(set(a.genome_id for a in results if a.genome_id))
        avg_lifespan = (
            sum((a.death_time - a.birth_time) if a.death_time else 0 for a in results)
            / total_agents
            if total_agents > 0
            else 0
        )

        return AgentEvolutionMetrics(
            total_agents=total_agents,
            unique_genomes=unique_genomes,
            average_lifespan=avg_lifespan,
            generation=generation,
        )

    def get_all_agents(self) -> List[AgentModel]:
        """Retrieve all agents from the database.

        Returns
        -------
        List[AgentModel]
            List of all agents with their basic information.
        """

        def query_agents(session: Session) -> List[AgentModel]:
            return (
                session.query(AgentModel).order_by(AgentModel.birth_time.desc()).all()
            )

        return self.session_manager.execute_with_retry(query_agents)
