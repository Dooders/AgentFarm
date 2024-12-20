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
import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError

from .models import (
    Agent,
    AgentAction,
    AgentState,
    HealthIncident,
    LearningExperience,
    ResourceState,
    SimulationStep,
)

logger = logging.getLogger(__name__)


class DataLogger:
    """Handles data logging operations for the simulation database."""

    def __init__(self, database, buffer_size: int = 1000):
        """Initialize the data logger.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for logging
        buffer_size : int, optional
            Maximum size of buffers before auto-flush, by default 1000
        """
        self.db = database
        self._buffer_size = buffer_size
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []
        self._resource_buffer = []

    def log_agent_action(
        self,
        step_number: int,
        agent_id: str,
        action_type: str,
        action_target_id: Optional[int] = None,
        position_before: Optional[Tuple[float, float]] = None,
        position_after: Optional[Tuple[float, float]] = None,
        resources_before: Optional[float] = None,
        resources_after: Optional[float] = None,
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

            action_data = {
                "step_number": step_number,
                "agent_id": agent_id,
                "action_type": action_type,
                "action_target_id": action_target_id,
                "position_before": str(position_before) if position_before else None,
                "position_after": str(position_after) if position_after else None,
                "resources_before": resources_before,
                "resources_after": resources_after,
                "reward": reward,
                "details": json.dumps(details) if details else None,
            }

            self._action_buffer.append(action_data)

            if len(self._action_buffer) >= self._buffer_size:
                self.flush_action_buffer()

        except ValueError as e:
            logger.error(f"Invalid input for agent action: {e}")
            raise
        except TypeError as e:
            logger.error(f"Type error in agent action data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error logging agent action: {e}")
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
            logger.error(f"Error logging learning experience: {e}")
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
            logger.error(f"Error logging health incident: {e}")
            raise

    def flush_action_buffer(self) -> None:
        """Flush the action buffer by batch inserting all buffered actions."""
        if not self._action_buffer:
            return

        buffer_copy = list(self._action_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(AgentAction, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._action_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush action buffer: {e}")
            raise

    def flush_learning_buffer(self) -> None:
        """Flush the learning experience buffer with transaction safety."""
        if not self._learning_exp_buffer:
            return

        buffer_copy = list(self._learning_exp_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(LearningExperience, buffer_copy)

            self.db._execute_in_transaction(_insert)
            self._learning_exp_buffer.clear()
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush learning buffer: {e}")
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

    def flush_all_buffers(self) -> None:
        """Flush all data buffers to the database.

        Safely writes all buffered data (actions, learning experiences, health incidents)
        to the database in transactions. Maintains data integrity by only clearing
        buffers after successful writes.

        Raises
        ------
        SQLAlchemyError
            If any buffer flush fails, with details about which buffer(s) failed
        """
        buffers = {
            "action": (self._action_buffer, self.flush_action_buffer),
            "learning": (self._learning_exp_buffer, self.flush_learning_buffer),
            "health": (self._health_incident_buffer, self.flush_health_buffer),
        }

        errors = []
        for buffer_name, (buffer, flush_func) in buffers.items():
            if buffer:  # Only attempt flush if buffer has data
                try:
                    flush_func()
                except SQLAlchemyError as e:
                    logger.error(f"Error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))
                except Exception as e:
                    logger.error(f"Unexpected error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))

        if errors:
            error_msg = "; ".join(f"{name}: {err}" for name, err in errors)
            raise SQLAlchemyError(f"Failed to flush buffers: {error_msg}")

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
            - starvation_threshold: int
            - genome_id: Optional[str]
            - generation: int

        Raises
        ------
        ValueError
            If agent data is malformed
        SQLAlchemyError
            If database operation fails
        """
        try:

            def _batch_insert(session):
                mappings = [
                    {
                        "agent_id": data["agent_id"],
                        "birth_time": data["birth_time"],
                        "agent_type": data["agent_type"],
                        "position_x": data["position"][0],
                        "position_y": data["position"][1],
                        "initial_resources": data["initial_resources"],
                        "starting_health": data["starting_health"],
                        "starvation_threshold": data["starvation_threshold"],
                        "genome_id": data.get("genome_id"),
                        "generation": data.get("generation", 0),
                    }
                    for data in agent_data_list
                ]
                session.bulk_insert_mappings(Agent, mappings)

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

    def log_agent(
        self,
        agent_id: str,
        birth_time: int,
        agent_type: str,
        position: Tuple[float, float],
        initial_resources: float,
        starting_health: float,
        starvation_threshold: int,
        genome_id: Optional[str] = None,
        generation: int = 0,
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
        starvation_threshold : int
            Steps agent can survive without resources
        genome_id : Optional[str]
            Unique identifier for agent's genome
        generation : int
            Generation number in evolutionary lineage

        Raises
        ------
        ValueError
            If input parameters are invalid
        SQLAlchemyError
            If database operation fails
        """
        agent_data = {
            "agent_id": agent_id,
            "birth_time": birth_time,
            "agent_type": agent_type,
            "position": position,
            "initial_resources": initial_resources,
            "starting_health": starting_health,
            "starvation_threshold": starvation_threshold,
            "genome_id": genome_id,
            "generation": generation,
        }
        self.log_agents_batch([agent_data])

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ) -> None:
        """Log comprehensive simulation state data for a single time step.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agent_states : List[Tuple]
            List of agent state tuples containing:
            (agent_id, position_x, position_y, resource_level, current_health,
             starting_health, starvation_threshold, is_defending, total_reward, age)
        resource_states : List[Tuple]
            List of resource state tuples containing:
            (resource_id, amount, position_x, position_y)
        metrics : Dict
            Dictionary of simulation metrics including:
            - total_agents: int
            - system_agents: int
            - independent_agents: int
            - control_agents: int
            - total_resources: float
            - average_agent_resources: float
            - resources_consumed: float
            - births: int
            - deaths: int
            And other relevant metrics

        Raises
        ------
        SQLAlchemyError
            If there's an error during database insertion
        ValueError
            If input data is malformed or invalid
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
                            "step_number": step_number,
                            "agent_id": state[0],
                            "position_x": state[1],
                            "position_y": state[2],
                            "resource_level": state[3],
                            "current_health": state[4],
                            "starting_health": state[5],
                            "starvation_threshold": state[6],
                            "is_defending": bool(state[7]),
                            "total_reward": state[8],
                            "age": state[9],
                        }
                        for state in agent_states
                    ]
                    # Create a set to track unique IDs and filter out duplicates
                    unique_states = {}
                    for mapping in agent_state_mappings:
                        state_id = f"{mapping['agent_id']}-{mapping['step_number']}"
                        mapping['id'] = state_id
                        # Keep only the latest state if there are duplicates
                        unique_states[state_id] = mapping

                    # Use the filtered mappings for bulk insert
                    session.bulk_insert_mappings(AgentState, unique_states.values())

                # Bulk insert resource states
                if resource_states:
                    resource_state_mappings = [
                        {
                            "step_number": step_number,
                            "resource_id": state[0],
                            "amount": state[1],
                            "position_x": state[2],
                            "position_y": state[3],
                        }
                        for state in resource_states
                    ]
                    session.bulk_insert_mappings(ResourceState, resource_state_mappings)

                # Insert metrics
                simulation_step = SimulationStep(step_number=step_number, **metrics)
                session.add(simulation_step)

            self.db._execute_in_transaction(_insert)

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data format in log_step: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in log_step: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in log_step: {e}")
            raise

    def log_resource(
        self, resource_id: int, initial_amount: float, position: Tuple[float, float]
    ) -> None:
        """Log a new resource in the simulation.

        Parameters
        ----------
        resource_id : int
            Unique identifier for the resource
        initial_amount : float
            Starting amount of the resource
        position : Tuple[float, float]
            (x, y) coordinates of the resource location

        Raises
        ------
        ValueError
            If input parameters are invalid
        SQLAlchemyError
            If database operation fails
        """
        try:
            resource_data = {
                "step_number": 0,  # Initial state
                "resource_id": resource_id,
                "amount": initial_amount,
                "position_x": position[0],
                "position_y": position[1],
            }

            def _insert(session):
                session.add(ResourceState(**resource_data))

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
