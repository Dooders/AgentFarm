"""Data logging module for simulation database.

This module handles all data logging operations including buffered writes
and transaction management for simulation data. It provides methods for
logging various types of data like agent actions, learning experiences,
and health incidents.

Features:
- Buffered batch operations for performance
- Transaction safety
- Automatic buffer flushing
- Comprehensive error handling
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError

from farm.core.interfaces import DatabaseProtocol, DataLoggerProtocol
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
    InteractionModel,
    LearningExperienceModel,
    ReproductionEventModel,
    ResourceModel,
    SimulationStepModel,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataLoggingConfig:
    """Configuration for data logging."""

    buffer_size: int = 1000  # Maximum size of buffers before auto-flush
    commit_interval: int = 30  # Maximum time (in seconds) between commits


class DataLogger(DataLoggerProtocol):
    """Handles data logging operations for the simulation database."""

    def __init__(
        self,
        database: DatabaseProtocol,
        simulation_id: Optional[str] = None,
        config: Optional[DataLoggingConfig] = None,
    ):
        """Initialize the data logger.

        Parameters
        ----------
        database : DatabaseProtocol
            Database instance to use for logging
        simulation_id : Optional[str], optional
            Unique identifier for this simulation, by default None
        config : Optional[DataLoggingConfig], optional
            Configuration for data logging, by default None
        """
        if not simulation_id:
            raise ValueError("simulation_id is required for DataLogger")

        self.simulation_id = simulation_id
        self.db = database

        # Use provided config or create default
        config = config or DataLoggingConfig()
        self._buffer_size = config.buffer_size
        self._commit_interval = config.commit_interval

        self._last_commit_time = time.time()
        #! Is there a better way to manage all these buffers?
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []
        self._resource_buffer = []
        self._interaction_buffer = []
        self.reproduction_buffer = []
        self._step_buffer = []

    def _check_time_based_flush(self):
        """Check if we should flush based on time interval."""
        current_time = time.time()
        if current_time - self._last_commit_time >= self._commit_interval:
            self.flush_all_buffers()
            self._last_commit_time = current_time

    def log_agent_action(
        self,
        step_number: int,
        agent_id: str,
        action_type: str,
        action_target_id: Optional[str] = None,
        reward: Optional[float] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Buffer an agent action with enhanced validation and error handling."""
        try:
            # Input validation
            if step_number < 0:
                raise ValueError("step_number must be non-negative")
            if not isinstance(action_type, str):
                action_type = str(action_type)

            #! Shouldnt this be a dataclass?
            action_data = {
                "simulation_id": self.simulation_id,
                "step_number": step_number,
                "agent_id": agent_id,
                "action_type": action_type,
                "action_target_id": action_target_id,
                "reward": reward,
                "details": json.dumps(details) if details else None,
            }

            self._action_buffer.append(action_data)

            if len(self._action_buffer) >= self._buffer_size:
                self.flush_action_buffer()

        except ValueError as e:
            logger.error(
                "invalid_agent_action_input",
                error_type="ValueError",
                error_message=str(e),
            )
            raise
        except TypeError as e:
            logger.error(
                "agent_action_type_error",
                error_type="TypeError",
                error_message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "agent_action_logging_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def log_interaction_edge(
        self,
        step_number: int,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        interaction_type: str,
        action_type: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Buffer an interaction as an edge between nodes.

        Ensures consistent structure for downstream analytics and visualization.
        """
        try:
            if step_number < 0:
                raise ValueError("step_number must be non-negative")

            edge = {
                "simulation_id": self.simulation_id,
                "step_number": step_number,
                "source_type": str(source_type),
                "source_id": str(source_id),
                "target_type": str(target_type),
                "target_id": str(target_id),
                "interaction_type": str(interaction_type),
                "action_type": str(action_type) if action_type else None,
                "details": details if details is not None else None,
            }

            self._interaction_buffer.append(edge)

            if len(self._interaction_buffer) >= self._buffer_size:
                self.flush_interaction_buffer()

        except Exception as e:
            logger.error(
                "interaction_edge_logging_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def log_learning_experience(
        self,
        step_number: int,
        agent_id: str,
        module_type: str,
        module_id: int,
        action_taken: int,
        action_taken_mapped: str,
        reward: float,
    ) -> None:
        """Buffer a learning experience."""
        try:
            exp_data = {
                "simulation_id": self.simulation_id,
                "step_number": step_number,
                "agent_id": agent_id,
                "module_type": module_type,
                "module_id": module_id,
                "action_taken": action_taken,
                "action_taken_mapped": action_taken_mapped,
                "reward": reward,
            }

            self._learning_exp_buffer.append(exp_data)

            if len(self._learning_exp_buffer) >= self._buffer_size:
                self.flush_learning_buffer()

        except Exception as e:
            logger.error(
                "learning_experience_logging_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def log_health_incident(
        self,
        step_number: int,
        agent_id: str,
        health_before: float,
        health_after: float,
        cause: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Buffer a health incident."""
        try:
            incident_data = {
                "simulation_id": self.simulation_id,
                "step_number": step_number,
                "agent_id": agent_id,
                "health_before": health_before,
                "health_after": health_after,
                "cause": cause,
                "details": json.dumps(details) if details else None,
            }

            self._health_incident_buffer.append(incident_data)

            if len(self._health_incident_buffer) >= self._buffer_size:
                self.flush_health_buffer()

        except Exception as e:
            logger.error(
                "health_incident_logging_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        offspring_id: str | None,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_initial_resources: float | None,
        failure_reason: str | None,
        parent_generation: int,
        offspring_generation: int | None,
        parent_position: tuple[float, float],
    ) -> None:
        """Buffer a reproduction event for batch processing."""
        event = {
            "simulation_id": self.simulation_id,
            "step_number": step_number,
            "parent_id": parent_id,
            "offspring_id": offspring_id,
            "success": success,
            "parent_resources_before": parent_resources_before,
            "parent_resources_after": parent_resources_after,
            "offspring_initial_resources": offspring_initial_resources,
            "failure_reason": failure_reason,
            "parent_generation": parent_generation,
            "offspring_generation": offspring_generation,
            "parent_position_x": parent_position[0],
            "parent_position_y": parent_position[1],
        }
        self.reproduction_buffer.append(event)

        if len(self.reproduction_buffer) >= self._buffer_size:
            self.flush_reproduction_buffer()

    def flush_action_buffer(self) -> None:
        """Flush the action buffer by batch inserting all buffered actions."""
        if not self._action_buffer:
            return

        buffer_copy = list(self._action_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(ActionModel, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._action_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(
                "action_buffer_flush_failed",
                buffer_size=len(self._action_buffer),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def flush_interaction_buffer(self) -> None:
        """Flush buffered interaction edges to the database."""
        if not self._interaction_buffer:
            return

        buffer_copy = list(self._interaction_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(InteractionModel, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._interaction_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(
                "interaction_buffer_flush_failed",
                buffer_size=len(self._interaction_buffer),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def flush_learning_buffer(self) -> None:
        """Flush the learning experience buffer with transaction safety."""
        if not self._learning_exp_buffer:
            return

        buffer_copy = list(self._learning_exp_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(LearningExperienceModel, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._learning_exp_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(
                "learning_buffer_flush_failed",
                buffer_size=len(self._learning_exp_buffer),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def flush_health_buffer(self) -> None:
        """Flush the health incident buffer with transaction safety."""
        if not self._health_incident_buffer:
            return

        buffer_copy = list(self._health_incident_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(HealthIncident, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._health_incident_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush health buffer: {e}")
            raise

    def flush_reproduction_buffer(self):
        """Flush buffered reproduction events to database."""
        if not self.reproduction_buffer:
            return

        def _flush(session):
            for event_data in self.reproduction_buffer:
                event = ReproductionEventModel(**event_data)
                session.add(event)

        self.db._execute_in_transaction(_flush)
        self.reproduction_buffer.clear()

    def flush_all_buffers(self) -> None:
        """Flush all data buffers to the database in a single transaction."""

        def _flush(session):
            # Disable autoflush during bulk operations
            original_autoflush = session.autoflush
            session.autoflush = False

            try:
                # Use bulk_insert_mappings instead of bulk_save_objects for dictionaries
                if self._action_buffer:
                    session.bulk_insert_mappings(ActionModel, self._action_buffer)
                    self._action_buffer.clear()

                if self._learning_exp_buffer:
                    session.bulk_insert_mappings(LearningExperienceModel, self._learning_exp_buffer)
                    self._learning_exp_buffer.clear()

                if self._health_incident_buffer:
                    session.bulk_insert_mappings(HealthIncident, self._health_incident_buffer)
                    self._health_incident_buffer.clear()

                if self.reproduction_buffer:
                    session.bulk_insert_mappings(ReproductionEventModel, self.reproduction_buffer)
                    self.reproduction_buffer.clear()

                if self._interaction_buffer:
                    session.bulk_insert_mappings(InteractionModel, self._interaction_buffer)
                    self._interaction_buffer.clear()

                if self._step_buffer:
                    session.bulk_insert_mappings(SimulationStepModel, self._step_buffer)
                    self._step_buffer.clear()

                # Commit once for all buffers
                session.commit()
            finally:
                # Restore original autoflush setting
                session.autoflush = original_autoflush

        self.db._execute_in_transaction(_flush)

    def log_agent(
        self,
        agent_id: str,
        birth_time: int,
        agent_type: str,
        position: Tuple[float, float],
        initial_resources: float,
        starting_health: float,
        starvation_counter: int,
        genome_id: Optional[str] = None,
        generation: int = 0,
        action_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a single agent's creation to the database.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent
        birth_time : int
            Time step when agent was created
        agent_type : str
            Type of agent (e.g., 'SystemAgent', 'IndependentAgent')
        position : Tuple[float, float]
            Initial (x, y) coordinates
        initial_resources : float
            Starting resource level
        starting_health : float
            Maximum health points
        starvation_counter : int
            Current count of consecutive steps with zero resources
        genome_id : Optional[str]
            Unique identifier for agent's genome
        generation : int
            Generation number in evolutionary lineage
        action_weights : Optional[Dict[str, float]]
            Dictionary mapping action names to their weights/probabilities

        Raises
        ------
        ValueError
            If input parameters are invalid
        SQLAlchemyError
            If database operation fails
        """
        # Ensure action_weights is properly serialized for JSON storage
        serialized_action_weights = None
        if action_weights is not None:
            try:
                # Convert action_weights to a serializable format
                serialized_action_weights = {str(k): float(v) for k, v in action_weights.items()}

                # Verify serialization by doing a round-trip through JSON
                json_test = json.dumps(serialized_action_weights)
                json.loads(json_test)  # This will raise an exception if not properly serializable

                logger.debug(f"Serialized action_weights for agent {agent_id}: {serialized_action_weights}")
            except Exception as e:
                logger.warning(f"Failed to serialize action_weights for agent {agent_id}: {e}")
                # Fall back to string representation if serialization fails
                serialized_action_weights = str(action_weights)

        agent_data = {
            "simulation_id": self.simulation_id,
            "agent_id": agent_id,
            "birth_time": birth_time,
            "agent_type": agent_type,
            "position": position,
            "initial_resources": initial_resources,
            "starting_health": starting_health,
            "starvation_counter": starvation_counter,
            "genome_id": genome_id,
            "generation": generation,
            "action_weights": serialized_action_weights,
        }
        self.log_agents_batch([agent_data])

    def log_agents_batch(self, agent_data_list: List[Dict]) -> None:
        """Batch insert multiple agents for better performance.

        Parameters
        ----------
        agent_data_list : List[Dict]
            List of dictionaries containing agent data with fields:
            - agent_id: str
            - birth_time: int
            - agent_type: str
            - position: Tuple[float, float]
            - initial_resources: float
            - starting_health: float
            - starvation_counter: int
            - genome_id: Optional[str]
            - generation: int
            - action_weights: Optional[Dict[str, float]]

        Raises
        ------
        ValueError
            If agent data is malformed
        SQLAlchemyError
            If database operation fails
        """
        try:

            def _batch_insert(session):
                mappings = []
                for data in agent_data_list:
                    # Validate that the required fields are present
                    required_fields = [
                        "agent_id",
                        "birth_time",
                        "agent_type",
                        "position",
                        "initial_resources",
                        "starting_health",
                        "starvation_counter",
                    ]
                    for field in required_fields:
                        if field not in data:
                            raise ValueError(f"Missing required field '{field}' in agent data")

                    # Process action_weights if present
                    action_weights = data.get("action_weights")
                    if action_weights is not None and not isinstance(action_weights, (dict, str)):
                        logger.warning(
                            f"Invalid action_weights format for agent {data['agent_id']}: {type(action_weights)}"
                        )
                        action_weights = str(action_weights)

                    # Create the mapping
                    mapping = {
                        "simulation_id": data["simulation_id"],
                        "agent_id": data["agent_id"],
                        "birth_time": data["birth_time"],
                        "agent_type": data["agent_type"],
                        "position_x": data["position"][0],
                        "position_y": data["position"][1],
                        "initial_resources": data["initial_resources"],
                        "starting_health": data["starting_health"],
                        "starvation_counter": data["starvation_counter"],
                        "genome_id": data.get("genome_id"),
                        "generation": data.get("generation", 0),
                        "action_weights": action_weights,
                    }
                    mappings.append(mapping)

                # Log the mappings for debugging
                logger.debug(f"Inserting {len(mappings)} agents into database")

                # Bulk insert the mappings
                session.bulk_insert_mappings(AgentModel, mappings)

            self.db._execute_in_transaction(_batch_insert)

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid agent data format: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during batch agent insert: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch agent insert: {e}")
            raise

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ) -> None:
        """Log a simulation step with all associated data.

        This method buffers step data and flushes when buffer is full
        or when the commit interval has elapsed.
        """
        try:

            def _insert(session):
                # Ensure resources_consumed has a default value if not provided
                if "resources_consumed" not in metrics:
                    metrics["resources_consumed"] = 0.0

                # Bulk insert agent states
                if agent_states:
                    agent_state_mappings = [
                        {
                            "simulation_id": self.simulation_id,
                            "step_number": step_number,
                            "agent_id": state[0],
                            "position_x": state[1],
                            "position_y": state[2],
                            "resource_level": state[3],
                            "current_health": state[4],
                            "starting_health": state[5],
                            "starvation_counter": state[6],
                            "is_defending": bool(state[7]),
                            "total_reward": state[8],
                            "age": state[9],
                        }
                        for state in agent_states
                    ]
                    # Create a set to track unique IDs and filter out duplicates
                    unique_states = {}
                    for mapping in agent_state_mappings:
                        # Include simulation_id in the ID to avoid collisions across simulations
                        sim_id = mapping.get("simulation_id")
                        if sim_id:
                            state_id = f"{sim_id}:{mapping['agent_id']}-{mapping['step_number']}"
                        else:
                            state_id = f"{mapping['agent_id']}-{mapping['step_number']}"
                        mapping["id"] = state_id
                        # Keep only the latest state if there are duplicates
                        unique_states[state_id] = mapping

                    # Use the filtered mappings for bulk insert
                    session.bulk_insert_mappings(AgentStateModel, unique_states.values())

                # Bulk insert resource states
                if resource_states:
                    resource_state_mappings = [
                        {
                            "simulation_id": self.simulation_id,
                            "step_number": step_number,
                            "resource_id": state[0],
                            "amount": state[1],
                            "position_x": state[2],
                            "position_y": state[3],
                        }
                        for state in resource_states
                    ]
                    session.bulk_insert_mappings(ResourceModel, resource_state_mappings)

                # Insert metrics
                simulation_step = SimulationStepModel(
                    simulation_id=self.simulation_id, step_number=step_number, **metrics
                )
                session.add(simulation_step)

            self.db._execute_in_transaction(_insert)

            # Check if we should flush based on buffer size
            if len(self._step_buffer) >= self._buffer_size:
                self.flush_all_buffers()
            else:
                # Check if we should flush based on time
                self._check_time_based_flush()

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data format in log_step: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in log_step: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in log_step: {e}")
            raise

    def log_resource(self, resource_id: int, initial_amount: float, position: Tuple[float, float]) -> None:
        """Log a new resource in the simulation."""
        try:
            resource_data = [
                {  # Changed to list for bulk_insert_mappings
                    "simulation_id": self.simulation_id,
                    "step_number": 0,  # Initial state
                    "resource_id": resource_id,
                    "amount": initial_amount,
                    "position_x": position[0],
                    "position_y": position[1],
                }
            ]

            def _insert(session):
                # Changed to bulk_insert_mappings for consistency
                session.bulk_insert_mappings(ResourceModel, resource_data)

            self.db._execute_in_transaction(_insert)

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid resource data format: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during resource insert: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during resource insert: {e}")
            raise


class ShardedDataLogger:
    """Data logger that routes data to appropriate database shards."""

    def __init__(self, sharded_db, simulation_id=None):
        """Initialize the sharded data logger.

        Parameters
        ----------
        sharded_db : ShardedSimulationDatabase
            The sharded database instance
        simulation_id : str, optional
            Unique identifier for this simulation, by default None
        """
        self.sharded_db = sharded_db
        self.simulation_id = simulation_id

    def log_agent_states(self, step_number, agent_states):
        """Log agent states to the appropriate shard."""
        shard_id = self.sharded_db._get_shard_for_step(step_number)
        self.sharded_db.shards[shard_id]["agents"].logger.log_agent_states(
            step_number, agent_states, simulation_id=self.simulation_id
        )

    def log_resources(self, step_number, resource_states):
        """Log resource states to the appropriate shard."""
        shard_id = self.sharded_db._get_shard_for_step(step_number)
        self.sharded_db.shards[shard_id]["resources"].logger.log_resources(
            step_number, resource_states, simulation_id=self.simulation_id
        )

    def log_metrics(self, step_number, metrics):
        """Log metrics to the appropriate shard."""
        shard_id = self.sharded_db._get_shard_for_step(step_number)
        self.sharded_db.shards[shard_id]["metrics"].logger.log_metrics(
            step_number, metrics, simulation_id=self.simulation_id
        )

    def log_action(self, step_number, action_data):
        """Log an action to the appropriate shard."""
        shard_id = self.sharded_db._get_shard_for_step(step_number)
        # Ensure simulation_id is included in action_data
        if self.simulation_id and "simulation_id" not in action_data:
            action_data["simulation_id"] = self.simulation_id
        self.sharded_db.shards[shard_id]["actions"].logger.log_agent_action(**action_data)

    def flush_all_buffers(self):
        """Flush all buffers across all shards."""
        for shard_id, databases in self.sharded_db.shards.items():
            for db_type, db in databases.items():
                db.logger.flush_all_buffers()
