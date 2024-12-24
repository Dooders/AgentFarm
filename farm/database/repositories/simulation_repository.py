from typing import List, Optional

from farm.database.data_types import (
    AgentStates,
    ResourceStates,
    SimulationResults,
    SimulationState,
)
from farm.database.models import (
    AgentModel,
    AgentStateModel,
    ResourceModel,
    SimulationStepModel,
)
from farm.database.repositories.base_repository import BaseRepository


class SimulationRepository(BaseRepository[SimulationStepModel, AgentStateModel]):
    """Handles retrieval of simulation state data from the database."""

    def __init__(self, db):
        """Initialize with database connection and model type."""
        super().__init__(db, SimulationStepModel)

    def agent_states(self, step_number: Optional[int] = None) -> List[AgentStates]:
        """Retrieve agent states for a specific step or all steps."""

        def _query(session):
            query = session.query(
                AgentStateModel.step_number,
                AgentStateModel.agent_id,
                AgentModel.agent_type,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentStateModel.resource_level,
                AgentStateModel.current_health,
                AgentStateModel.is_defending,
            ).join(AgentModel)

            if step_number is not None:
                query = query.filter(AgentStateModel.step_number == step_number)
            else:
                query = query.order_by(
                    AgentStateModel.step_number, AgentStateModel.agent_id
                )

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

        return self._execute_in_transaction(_query)

    def resource_states(self, step_number: int) -> List[ResourceStates]:
        """Retrieve resource states for a specific step."""

        def _query(session):
            query = session.query(
                ResourceModel.resource_id,
                ResourceModel.amount,
                ResourceModel.position_x,
                ResourceModel.position_y,
            ).filter(ResourceModel.step_number == step_number)

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

        return self._execute_in_transaction(_query)

    def simulation_state(self, step_number: int) -> SimulationState:
        """Retrieve simulation state for a specific step."""

        def _query(session):
            query = session.query(SimulationStepModel).filter(
                SimulationStepModel.step_number == step_number
            )

            result = query.first()

            return SimulationState(**result.as_dict())

        return self._execute_in_transaction(_query)

    def execute(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step."""
        return SimulationResults(
            agent_states=self.agent_states(step_number),
            resource_states=self.resource_states(step_number),
            simulation_state=self.simulation_state(step_number),
        )
