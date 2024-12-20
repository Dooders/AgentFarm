"""Database interface for retrieving simulation state data.

This module provides classes and utilities for accessing and retrieving simulation
state data from the database, including agent positions, resource distributions,
and overall simulation metrics.

The module is designed to work with SQLAlchemy and provides a clean interface
for querying complex simulation state data while maintaining separation of concerns
between database access and simulation logic.

Classes
-------
SimulationStateRetriever
    Main class for retrieving simulation state data from the database
"""

from typing import List, Optional

from database.data_types import (
    AgentStates,
    ResourceStates,
    SimulationResults,
    SimulationState,
)
from database.models import AgentModel, AgentState, ResourceState, SimulationStep
from database.utilities import execute_query


class SimulationStateRetriever:
    """Handles retrieval of simulation state data from the database.

    This class encapsulates methods for retrieving agent states, resource states,
    and overall simulation state data for specific simulation steps. It provides
    a clean interface for accessing simulation data while handling all database
    query complexity internally.

    The class uses SQLAlchemy for database operations and returns strongly-typed
    dataclass objects containing the requested simulation state data.

    Methods
    -------
    agent_states(step_number: Optional[int] = None) -> List[AgentStates]
        Retrieve agent states for a specific step or all steps
    resource_states(step_number: int) -> List[ResourceStates]
        Retrieve resource states for a specific step
    simulation_state(step_number: int) -> SimulationState
        Retrieve overall simulation metrics for a specific step
    execute(step_number: int) -> SimulationResults
        Retrieve complete simulation state for a specific step

    Attributes
    ----------
    db : SimulationDatabase
        Database connection instance used for queries
    """

    def __init__(self, database):
        """Initialize with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    @execute_query
    def agent_states(
        self, session, step_number: Optional[int] = None
    ) -> List[AgentStates]:
        """Retrieve agent states for a specific step or all steps.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            SQLAlchemy database session for executing queries
        step_number : Optional[int], default=None
            The simulation step number to retrieve data for.
            If None, returns data for all steps.

        Returns
        -------
        List[AgentStates]
            List of agent states, where each AgentStates object contains:
            - step_number: int
                Simulation step number
            - agent_id: int
                Agent's unique identifier
            - agent_type: str
                Agent's category
            - position_x: float
                Agent's x coordinate
            - position_y: float
                Agent's y coordinate
            - resource_level: float
                Agent's current resources
            - current_health: float
                Agent's health level
            - is_defending: bool
                Agent's defensive status

        Notes
        -----
        Results are ordered by:
        - step_number, agent_id when retrieving all steps
        - agent_id only when retrieving a single step
        """
        query = session.query(
            AgentState.step_number,
            AgentState.agent_id,
            AgentModel.agent_type,
            AgentState.position_x,
            AgentState.position_y,
            AgentState.resource_level,
            AgentState.current_health,
            AgentState.is_defending,
        ).join(AgentModel)

        if step_number is not None:
            query = query.filter(AgentState.step_number == step_number)
        else:
            query = query.order_by(AgentState.step_number, AgentState.agent_id)

        results = query.all()

        # Map results to AgentStates dataclass instances
        agent_states_list = [
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

        return agent_states_list

    @execute_query
    def resource_states(self, session, step_number: int) -> List[ResourceStates]:
        """Retrieve resource states for a specific step.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            SQLAlchemy database session for executing queries
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        List[ResourceStates]
            List of resource states, where each ResourceStates object contains:
            - resource_id: int
                Unique identifier for the resource
            - amount: float
                Current amount of the resource
            - position_x: float
                X coordinate of resource position
            - position_y: float
                Y coordinate of resource position

        Notes
        -----
        Resources that were depleted or removed will not be included in the results.
        Positions are in simulation grid coordinates.
        """
        query = session.query(
            ResourceState.resource_id,
            ResourceState.amount,
            ResourceState.position_x,
            ResourceState.position_y,
        ).filter(ResourceState.step_number == step_number)

        results = query.all()

        return [
            ResourceStates(
                resource_id=row[0],
                amount=row[1],
                position_x=row[2],
                position_y=row[3],
            )
            for row in results
        ]

    @execute_query
    def simulation_state(self, session, step_number: int) -> SimulationState:
        """Retrieve simulation state for a specific step.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            SQLAlchemy database session for executing queries
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        SimulationState
            Simulation state object containing:
            - step_number: int
                Current step number
            - total_agents: int
                Total number of agents alive
            - total_resources: float
                Total resources available
            - average_agent_health: float
                Mean health across all agents
            - average_agent_resources: float
                Mean resources per agent
            - births: int
                Number of births this step
            - deaths: int
                Number of deaths this step
            - system_metrics: Dict[str, float]
                Additional system performance metrics

        Returns None if the step number is not found in the database.
        """
        query = session.query(SimulationStep).filter(
            SimulationStep.step_number == step_number
        )

        result = query.first()

        return SimulationState(**result.as_dict())

    def execute(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        SimulationResults
            Data containing agent states, resource states, and simulation metrics
        """
        return SimulationResults(
            agent_states=self.agent_states(step_number),
            resource_states=self.resource_states(step_number),
            simulation_state=self.simulation_state(step_number),
        )
