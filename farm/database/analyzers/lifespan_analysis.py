import pandas as pd
from sqlalchemy import case, func

from farm.database.data_types import (
    AgentLifespanResults,
    LifespanStatistics,
    SurvivalRatesByGeneration,
)
from farm.database.models import AgentModel
from farm.database.utilities import execute_query


class AgentLifespanAnalysis:
    """Handles retrieval and analysis of agent lifespan statistics.

    This class provides methods to analyze and compute various statistics related to agent
    lifespans, including survival rates, generational trends, and type-specific metrics
    within the simulation database.

    Parameters
    ----------
    database : SimulationDatabase
        The database instance to use for executing queries

    Methods
    -------
    lifespan_statistics() -> LifespanStatistics
        Calculate comprehensive lifespan statistics across agent types and generations
    survival_rates() -> SurvivalRatesByGeneration
        Calculate the survival rates for each generation of agents
    execute() -> AgentLifespanResults
        Calculate and combine all agent lifespan statistics
    """

    def __init__(self, database):
        """Initialize the retriever with a database connection.

        Parameters
        ----------
        database : SimulationDatabase
            The database instance to use for executing queries
        """
        self.db = database

    @execute_query
    def lifespan_statistics(self, session) -> LifespanStatistics:
        """Calculate comprehensive lifespan statistics across agent types and generations.

        Queries the database to compute statistical measures of agent lifespans,
        including averages, extremes, and breakdowns by type and generation.

        Parameters
        ----------
        session : Session
            SQLAlchemy session provided by the execute_query decorator

        Returns
        -------
        LifespanStatistics
            A data class containing:
            - average_lifespan : float
                Mean lifespan across all agents
            - maximum_lifespan : float
                Longest recorded lifespan
            - minimum_lifespan : float
                Shortest recorded lifespan
            - lifespan_by_type : Dict[str, float]
                Mean lifespan for each agent type
            - lifespan_by_generation : Dict[int, float]
                Mean lifespan for each generation
        """
        lifespans = (
            session.query(
                AgentModel.agent_type,
                AgentModel.generation,
                (AgentModel.death_time - AgentModel.birth_time).label("lifespan"),
            )
            .filter(AgentModel.death_time.isnot(None))
            .all()
        )

        lifespan_data = pd.DataFrame(
            lifespans, columns=["agent_type", "generation", "lifespan"]
        )

        return LifespanStatistics(
            average_lifespan=lifespan_data["lifespan"].mean(),
            maximum_lifespan=lifespan_data["lifespan"].max(),
            minimum_lifespan=lifespan_data["lifespan"].min(),
            lifespan_by_type=lifespan_data.groupby("agent_type")["lifespan"]
            .mean()
            .to_dict(),
            lifespan_by_generation=lifespan_data.groupby("generation")["lifespan"]
            .mean()
            .to_dict(),
        )

    @execute_query
    def survival_rates(self, session) -> SurvivalRatesByGeneration:
        """Calculate the survival rates for each generation of agents.

        Queries the database to compute the percentage of agents still alive
        (not marked with a death_time) within each generation.

        Parameters
        ----------
        session : Session
            SQLAlchemy session provided by the execute_query decorator

        Returns
        -------
        SurvivalRatesByGeneration
            A data class containing:
            - rates : Dict[int, float]
                Dictionary mapping generation numbers to their survival rates (0-100%)
        """
        survival_rates = (
            session.query(
                AgentModel.generation,
                func.count(case((AgentModel.death_time.is_(None), 1)))
                * 100.0
                / func.count(),
            )
            .group_by(AgentModel.generation)
            .all()
        )

        survival_data = pd.DataFrame(
            survival_rates, columns=["generation", "survival_rate"]
        )

        return SurvivalRatesByGeneration(
            rates=survival_data.set_index("generation")["survival_rate"].to_dict()
        )

    @execute_query
    def execute(self, session) -> AgentLifespanResults:
        """Calculate and combine all agent lifespan statistics.

        Aggregates results from lifespan_statistics() and survival_rates()
        to provide a comprehensive view of agent performance and longevity across
        the simulation.

        Parameters
        ----------
        session : Session
            SQLAlchemy session provided by the execute_query decorator

        Returns
        -------
        AgentLifespanResults
            A data class containing:
            - lifespan_statistics : LifespanStatistics
                Comprehensive lifespan statistics
            - survival_rates : SurvivalRatesByGeneration
                Survival rates by generation
        """
        # Get lifespan statistics
        lifespan_stats = self.lifespan_statistics(session)

        # Get survival rates
        survival_rates = self.survival_rates(session)

        return AgentLifespanResults(
            lifespan_statistics=lifespan_stats,
            survival_rates=survival_rates,
        )
