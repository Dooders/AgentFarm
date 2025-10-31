"""Experiment database module for managing multiple simulations.

This module provides classes for managing multiple simulations within a single
database file. It extends the SimulationDatabase to handle multiple simulations
by tagging all data with a simulation_id.

Key Components
-------------
- ExperimentDatabase : Main class for experiment database
- SimulationContext : Context for a specific simulation within an experiment
- ExperimentDataLogger : Logger that tags all data with simulation_id

Features
--------
- Store multiple simulations in a single database file
- Query and analyze data across simulations
- Filter data by simulation_id
- Consistent schema across all simulations
- Efficient data storage and retrieval
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import Index

from farm.database.data_logging import DataLogger, DataLoggingConfig
from farm.database.database import SimulationDatabase
from farm.database.models import (
    AgentStateModel,
    ExperimentModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentDataLogger(DataLogger):
    #! Just update data logger
    """Data logger that tags all data with simulation_id.

    This extends the standard DataLogger to ensure all data is tagged
    with the correct simulation_id, enabling multiple simulations to be
    stored in a single database file.
    """

    def __init__(
        self, database, simulation_id, config: DataLoggingConfig = DataLoggingConfig()
    ):
        """Initialize the experiment data logger.

        Parameters
        ----------
        database : ExperimentDatabase
            Database instance to use for logging
        simulation_id : str
            Unique identifier for this simulation
        config : DataLoggingConfig, optional
            Configuration for data logging, by default None
        """
        super().__init__(database, simulation_id, config)

        # Ensure simulation_id is always set
        if not simulation_id:
            raise ValueError("simulation_id is required for ExperimentDataLogger")

        self.simulation_id = simulation_id

    def log_agent(
        self,
        agent_id,
        birth_time,
        agent_type,
        position,
        initial_resources,
        starting_health,
        genome_id=None,
        generation=0,
        action_weights=None,
    ):
        """Log an agent with the correct simulation_id and make agent_id unique.

        This method prefixes the agent_id with the simulation_id to ensure
        uniqueness across simulations.
        """
        # Prefix agent_id with simulation_id to ensure uniqueness
        unique_agent_id = agent_id

        return super().log_agent(
            agent_id=unique_agent_id,
            birth_time=birth_time,
            agent_type=agent_type,
            position=position,
            initial_resources=initial_resources,
            starting_health=starting_health,
            genome_id=genome_id,
            generation=generation,
            action_weights=action_weights,
        )

    def log_agent_action(
        self,
        step_number,
        agent_id,
        action_type,
        action_target_id=None,
        resources_before=None,
        resources_after=None,
        reward=None,
        details=None,
    ):
        """Log an agent action with the correct simulation_id and prefixed agent_ids."""
        # Prefix agent_ids with simulation_id
        unique_agent_id = agent_id

        # Also prefix target_id if it exists
        unique_target_id = None
        if action_target_id:
            unique_target_id = action_target_id

        return super().log_agent_action(
            step_number=step_number,
            agent_id=unique_agent_id,
            action_type=action_type,
            action_target_id=unique_target_id,
            resources_before=resources_before,
            resources_after=resources_after,
            reward=reward,
            details=details,
        )

    def log_health_incident(
        self, step_number, agent_id, health_before, health_after, cause, details=None
    ):
        """Log a health incident with the correct simulation_id and prefixed agent_id."""
        # Prefix agent_id with simulation_id
        unique_agent_id = agent_id

        return super().log_health_incident(
            step_number=step_number,
            agent_id=unique_agent_id,
            health_before=health_before,
            health_after=health_after,
            cause=cause,
            details=details,
        )

    def log_reproduction_event(
        self,
        step_number,
        parent_id,
        offspring_id,
        success,
        parent_resources_before,
        parent_resources_after,
        offspring_initial_resources,
        failure_reason,
        parent_generation,
        offspring_generation,
        parent_position,
    ):
        """Log a reproduction event with the correct simulation_id and prefixed agent_ids."""
        # Prefix agent_ids with simulation_id
        unique_parent_id = parent_id

        # Also prefix offspring_id if it exists
        unique_offspring_id = None
        if offspring_id:
            unique_offspring_id = offspring_id

        return super().log_reproduction_event(
            step_number=step_number,
            parent_id=unique_parent_id,
            offspring_id=unique_offspring_id,
            success=success,
            parent_resources_before=parent_resources_before,
            parent_resources_after=parent_resources_after,
            offspring_initial_resources=offspring_initial_resources,
            failure_reason=failure_reason,
            parent_generation=parent_generation,
            offspring_generation=offspring_generation,
            parent_position=parent_position,
        )

    def log_learning_experience(
        self,
        step_number,
        agent_id,
        module_type,
        module_id,
        action_taken,
        action_taken_mapped,
        reward,
    ):
        """Log a learning experience with the correct simulation_id and prefixed agent_id."""

        return super().log_learning_experience(
            step_number=step_number,
            agent_id=agent_id,
            module_type=module_type,
            module_id=module_id,
            action_taken=action_taken,
            action_taken_mapped=action_taken_mapped,
            reward=reward,
        )

    def log_step(self, step_number, agent_states, resource_states, metrics):
        """Log a step with prefixed agent_ids.

        This method modifies the agent_states list to prefix all agent_ids with
        the simulation_id before passing it to the parent log_step method.
        """
        # Prefix agent_ids in agent_states
        modified_agent_states = []
        for state in agent_states:
            agent_id = state[0]
            unique_agent_id = agent_id
            modified_state = (unique_agent_id,) + state[1:]
            modified_agent_states.append(modified_state)

        # Override the default log_step to use a custom implementation
        # that can handle multiple simulations with the same step numbers
        try:

            def _insert(session):
                # Ensure resources_consumed has a default value if not provided
                if "resources_consumed" not in metrics:
                    metrics["resources_consumed"] = 0.0

                # Bulk insert agent states
                if modified_agent_states:
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
                        for state in modified_agent_states
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
                    session.bulk_insert_mappings(
                        AgentStateModel, unique_states.values()
                    )

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

                # Try to find an existing step to update
                existing_step = (
                    session.query(SimulationStepModel)
                    .filter(
                        SimulationStepModel.step_number == step_number,
                        SimulationStepModel.simulation_id == self.simulation_id,
                    )
                    .first()
                )

                if existing_step:
                    # Update the existing step
                    for key, value in metrics.items():
                        if hasattr(existing_step, key):
                            setattr(existing_step, key, value)
                else:
                    # Insert metrics as a new step
                    simulation_step = SimulationStepModel(
                        simulation_id=self.simulation_id,
                        step_number=step_number,
                        **metrics,
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
        except Exception as e:
            logger.error(f"Database error in log_step: {e}")
            raise


class SimulationContext:
    """Context for a specific simulation within an experiment database.

    This class provides a simulation-specific view of the experiment database,
    ensuring all data is tagged with the correct simulation_id.
    """

    def __init__(
        self, parent_db, simulation_id, config: DataLoggingConfig = DataLoggingConfig()
    ):
        """Initialize the simulation context.

        Parameters
        ----------
        parent_db : ExperimentDatabase
            The parent experiment database
        simulation_id : str
            Unique identifier for this simulation
        config : DataLoggingConfig, optional
            Configuration for data logging, by default None
        """
        self.parent_db = parent_db
        self.simulation_id = simulation_id
        self.logger = ExperimentDataLogger(parent_db, simulation_id, config)

    def log_step(self, step_number, agent_states, resource_states, metrics):
        """Log a simulation step with the correct simulation_id."""
        return self.logger.log_step(step_number, agent_states, resource_states, metrics)

    def log_agent(
        self,
        agent_id,
        birth_time,
        agent_type,
        position,
        initial_resources,
        starting_health,
        genome_id=None,
        generation=0,
        action_weights=None,
    ):
        """Log an agent with the correct simulation_id."""
        return self.logger.log_agent(
            agent_id=agent_id,
            birth_time=birth_time,
            agent_type=agent_type,
            position=position,
            initial_resources=initial_resources,
            starting_health=starting_health,
            genome_id=genome_id,
            generation=generation,
            action_weights=action_weights,
        )

    def log_reproduction_event(self, **kwargs):
        """Log a reproduction event with the correct simulation_id."""
        return self.logger.log_reproduction_event(**kwargs)

    def log_agent_action(self, **kwargs):
        """Log an agent action with the correct simulation_id."""
        return self.logger.log_agent_action(**kwargs)

    def log_health_incident(self, **kwargs):
        """Log a health incident with the correct simulation_id."""
        return self.logger.log_health_incident(**kwargs)

    def log_learning_experience(self, **kwargs):
        """Log a learning experience with the correct simulation_id."""
        return self.logger.log_learning_experience(**kwargs)

    def flush_all_buffers(self):
        """Flush all data buffers to the database."""
        return self.logger.flush_all_buffers()

    def close(self):
        """Close the database connection."""
        # Just flush any remaining data, don't close the parent
        self.logger.flush_all_buffers()


class ExperimentDatabase(SimulationDatabase):
    """Database for storing multiple simulations in a single file.

    This class extends SimulationDatabase to support multiple simulations
    in a single database file. Each simulation is tagged with a unique
    simulation_id, enabling data to be filtered by simulation.
    """

    def __init__(self, db_path: str, experiment_id: str, config=None):
        """Initialize a new ExperimentDatabase instance.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        experiment_id : str
            Unique identifier for this experiment
        config : SimulationConfig, optional
            Configuration object with database settings
        """
        # Initialize the parent class without a simulation_id
        super().__init__(db_path, config=config, simulation_id=None)

        self.experiment_id = experiment_id

        # Create the experiment record
        self._create_experiment_record()

        # Modify the SimulationStepModel.__table_args__ to remove primary key constraint
        # This is a hack, but it's the easiest way to work around the constraint
        # We'll update the model to use (step_number, simulation_id) as primary key
        if hasattr(SimulationStepModel, "__table_args__"):
            # Find the Index for step_number
            new_table_args = []
            for arg in SimulationStepModel.__table_args__:
                if (
                    isinstance(arg, Index)
                    and arg.name == "idx_simulation_steps_step_number"
                ):
                    continue
                new_table_args.append(arg)

            # Add new indexes and constraints
            new_table_args.append(
                Index(
                    "idx_sim_steps_step_sim",
                    "step_number",
                    "simulation_id",
                    unique=True,
                )
            )
            SimulationStepModel.__table_args__ = tuple(new_table_args)

    def _create_experiment_record(self):
        """Create the experiment record in the database."""

        def _create(session):
            # Check if the experiment already exists
            existing = (
                session.query(ExperimentModel)
                .filter(ExperimentModel.experiment_id == self.experiment_id)
                .first()
            )

            if not existing:
                # Create a new experiment record
                experiment = ExperimentModel(
                    experiment_id=self.experiment_id,
                    name=f"Experiment {self.experiment_id}",
                    description="Created with ExperimentDatabase",
                    status="running",
                )
                session.add(experiment)

        self._execute_in_transaction(_create)

    def create_simulation_context(
        self,
        simulation_id: str,
        parameters: Optional[Dict] = None,
        logging_config: DataLoggingConfig = DataLoggingConfig(),
    ):
        """Create a simulation-specific context.

        This method creates a simulation-specific context that tags all data
        with the simulation_id, enabling multiple simulations to be stored
        in a single database file.

        Parameters
        ----------
        simulation_id : str
            Unique identifier for this simulation
        parameters : Dict, optional
            Simulation parameters, by default None
        logging_config : DataLoggingConfig, optional
            Configuration for data logging, by default None

        Returns
        -------
        SimulationContext
            A context for the specific simulation
        """
        # Create the simulation record
        self._create_simulation_record(simulation_id, parameters or {})

        # Return a context for this simulation
        return SimulationContext(self, simulation_id, logging_config)

    def _create_simulation_record(self, simulation_id: str, parameters: Optional[Dict] = None):
        """Create a simulation record in the database.

        Parameters
        ----------
        simulation_id : str
            Unique identifier for this simulation
        parameters : Dict, optional
            Simulation parameters, by default None
        """

        def _create(session):
            # Check if the simulation already exists
            existing = (
                session.query(Simulation)
                .filter(Simulation.simulation_id == simulation_id)
                .first()
            )

            if not existing:
                # Create a new simulation record
                simulation = Simulation(
                    simulation_id=simulation_id,
                    experiment_id=self.experiment_id,
                    parameters=parameters or {},
                    status="running",
                    simulation_db_path=self.db_path,  # Use the experiment db path
                )
                session.add(simulation)

        self._execute_in_transaction(_create)

    def update_simulation_status(
        self, simulation_id: str, status: str, results_summary: Optional[Dict] = None
    ):
        """Update the status of a simulation.

        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        status : str
            New status (e.g., 'completed', 'failed')
        results_summary : Dict, optional
            Summary of simulation results, by default None
        """

        # Preserve prior semantics: normalize None to {}
        results_summary = results_summary or {}

        def _update(session):
            simulation = (
                session.query(Simulation)
                .filter(Simulation.simulation_id == simulation_id)
                .first()
            )

            if simulation:
                simulation.status = status
                # Use datetime object instead of timestamp
                if status == "completed":
                    simulation.end_time = datetime.now()
                if results_summary:
                    simulation.results_summary = results_summary

        self._execute_in_transaction(_update)

    def get_simulation_ids(self):
        """Get all simulation IDs in this experiment.

        Returns
        -------
        list
            List of simulation IDs
        """

        def _query(session):
            simulations = (
                session.query(Simulation)
                .filter(Simulation.experiment_id == self.experiment_id)
                .all()
            )
            return [sim.simulation_id for sim in simulations]

        return self._execute_in_transaction(_query)

    def update_experiment_status(self, status: str, results_summary: Optional[Dict] = None):
        """Update the status of the experiment.

        Parameters
        ----------
        status : str
            New status (e.g., 'completed', 'analyzing')
        results_summary : Dict, optional
            Summary of experiment results, by default None
        """

        # Preserve prior semantics: normalize None to {}
        results_summary = results_summary or {}

        def _update(session):
            experiment = (
                session.query(ExperimentModel)
                .filter(ExperimentModel.experiment_id == self.experiment_id)
                .first()
            )

            if experiment:
                experiment.status = status
                if results_summary:
                    experiment.results_summary = results_summary
                # Update last_updated timestamp
                experiment.last_updated = datetime.now()

        self._execute_in_transaction(_update)
