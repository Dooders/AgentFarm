from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from farm.database.models import (
    AgentModel,
    AgentStateModel,
    ResourceModel,
    SimulationStepModel,
)
from farm.database.repositories.base_repository import BaseRepository
from farm.database.session_manager import SessionManager


class GUIRepository(BaseRepository[SimulationStepModel]):
    """Repository for handling GUI-related data queries.

    This repository provides methods to retrieve historical simulation data
    for visualization and analysis in the GUI components.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize repository with session manager.

        Parameters
        ----------
        session_manager : SessionManager
            Session manager instance for database operations
        """
        super().__init__(session_manager, SimulationStepModel)

    def get_historical_data(
        self,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Retrieve historical metrics for the entire simulation.

        Parameters
        ----------
        agent_id : Optional[int]
            Filter results for specific agent
        step_range : Optional[Tuple[int, int]]
            Filter results for step range (start, end)

        Returns
        -------
        Dict
            Dictionary containing:
            - steps: List of step numbers
            - metrics: Dict of metric lists including:
                - total_agents
                - system_agents
                - independent_agents
                - control_agents
                - total_resources
                - average_agent_resources

        Notes
        -----
        This method is useful for generating time series plots of simulation metrics.
        The returned lists are ordered by step number.
        """

        def _query(session: Session) -> Dict:
            # Build base query
            query = session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.total_agents,
                SimulationStepModel.agent_type_counts,
                SimulationStepModel.total_resources,
                SimulationStepModel.average_agent_resources,
            ).order_by(SimulationStepModel.step_number)

            # Apply filters if provided
            if agent_id is not None:
                query = query.filter(SimulationStepModel.agent_id == agent_id)

            if step_range is not None:
                start, end = step_range
                query = query.filter(
                    SimulationStepModel.step_number >= start,
                    SimulationStepModel.step_number <= end,
                )

            # Execute query and process results
            rows = query.all()

            return {
                "steps": [row[0] for row in rows],
                "metrics": {
                    "total_agents": [row[1] for row in rows],
                    "system_agents": [(row[2] or {}).get("system", 0) if row[2] else 0 for row in rows],
                    "independent_agents": [(row[2] or {}).get("independent", 0) if row[2] else 0 for row in rows],
                    "control_agents": [(row[2] or {}).get("control", 0) if row[2] else 0 for row in rows],
                    "total_resources": [row[3] for row in rows],
                    "average_agent_resources": [row[4] for row in rows],
                },
            }

        return self.session_manager.execute_with_retry(_query)

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics for simulation metrics.

        Returns
        -------
        Dict
            Dictionary containing summary statistics for each metric:
            - min, max, avg values
            - standard deviation
            - total count
        """

        def _query(session: Session) -> Dict:
            summary = session.query(
                func.min(SimulationStepModel.total_agents).label("min_agents"),
                func.max(SimulationStepModel.total_agents).label("max_agents"),
                func.avg(SimulationStepModel.total_agents).label("avg_agents"),
                func.stddev(SimulationStepModel.total_agents).label("std_agents"),
                func.count(SimulationStepModel.step_number).label("total_steps"),
                # Add similar aggregates for other metrics...
            ).first()

            if not summary:
                return {
                    "agents": {
                        "min": 0,
                        "max": 0,
                        "avg": 0,
                        "std": 0,
                    },
                    "total_steps": 0,
                }

            return {
                "agents": {
                    "min": summary.min_agents,
                    "max": summary.max_agents,
                    "avg": summary.avg_agents,
                    "std": summary.std_agents,
                },
                "total_steps": summary.total_steps,
            }

        return self.session_manager.execute_with_retry(_query)

    def get_step_data(self, step_number: int) -> Dict:
        """Get all metrics for a specific simulation step.

        Parameters
        ----------
        step_number : int
            The simulation step to query

        Returns
        -------
        Dict
            All metrics for the specified step
        """

        def _query(session: Session) -> Dict:
            step = (
                session.query(SimulationStepModel)
                .filter(SimulationStepModel.step_number == step_number)
                .first()
            )

            if not step:
                return {}

            agent_counts = step.agent_type_counts or {}
            return {
                "total_agents": step.total_agents,
                "system_agents": agent_counts.get("system", 0),
                "independent_agents": agent_counts.get("independent", 0),
                "control_agents": agent_counts.get("control", 0),
                "total_resources": step.total_resources,
                "average_agent_resources": step.average_agent_resources,
            }

        return self.session_manager.execute_with_retry(_query)

    def get_simulation_data(self, step_number: int) -> Dict:
        """Retrieve all simulation data for a specific time step.

        Parameters
        ----------
        step_number : int
            The simulation step to retrieve data for

        Returns
        -------
        Dict
            Dictionary containing:
            - agent_states: List of tuples (agent_id, type, x, y, resources, health, defending)
            - resource_states: List of tuples (resource_id, amount, x, y)
            - metrics: Dict of aggregate metrics for the step

        Example
        -------
        >>> data = repo.get_simulation_data(100)
        >>> print(f"Number of agents: {len(data['agent_states'])}")
        >>> print(f"Total resources: {data['metrics']['total_resources']}")
        """

        def _query(session: Session) -> Dict:
            # Get agent states with agent type
            agent_states = (
                session.query(
                    AgentStateModel.agent_id,
                    AgentModel.agent_type,
                    AgentStateModel.position_x,
                    AgentStateModel.position_y,
                    AgentStateModel.resource_level,
                    AgentStateModel.current_health,
                    AgentStateModel.is_defending,
                )
                .join(AgentModel)
                .filter(AgentStateModel.step_number == step_number)
                .all()
            )

            # Get resource states
            resource_states = (
                session.query(
                    ResourceModel.resource_id,
                    ResourceModel.amount,
                    ResourceModel.position_x,
                    ResourceModel.position_y,
                )
                .filter(ResourceModel.step_number == step_number)
                .all()
            )

            # Get metrics
            metrics = (
                session.query(SimulationStepModel)
                .filter(SimulationStepModel.step_number == step_number)
                .first()
            )

            if metrics:
                agent_counts = metrics.agent_type_counts or {}
                metrics_dict = {
                    "total_agents": metrics.total_agents,
                    "system_agents": agent_counts.get("system", 0),
                    "independent_agents": agent_counts.get("independent", 0),
                    "control_agents": agent_counts.get("control", 0),
                    "total_resources": metrics.total_resources,
                    "average_agent_resources": metrics.average_agent_resources,
                    "births": metrics.births,
                    "deaths": metrics.deaths,
                    "average_agent_health": metrics.average_agent_health,
                    "combat_encounters": metrics.combat_encounters,
                    "resources_shared": metrics.resources_shared,
                }
            else:
                metrics_dict = {
                    "total_agents": 0,
                    "system_agents": 0,
                    "independent_agents": 0,
                    "control_agents": 0,
                    "total_resources": 0,
                    "average_agent_resources": 0,
                    "births": 0,
                    "deaths": 0,
                    "average_agent_health": 0,
                    "combat_encounters": 0,
                    "resources_shared": 0,
                }

            return {
                "agent_states": agent_states,
                "resource_states": resource_states,
                "metrics": metrics_dict,
            }

        return self.session_manager.execute_with_retry(_query)
