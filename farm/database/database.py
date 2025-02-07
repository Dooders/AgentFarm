"""Database module for simulation state persistence and analysis.

This module provides a SQLAlchemy-based database implementation for storing and 
analyzing simulation data. It handles all database operations including state 
logging, configuration management, and data analysis.

Key Components
-------------
- SimulationDatabase : Main database interface class
- DataLogger : Handles buffered data logging operations
- DataRetriever : Handles data querying operations
- SQLAlchemy Models : ORM models for different data types
    - Agent : Agent entity data
    - AgentState : Time-series agent state data
    - ResourceState : Resource position and amount tracking
    - SimulationStep : Per-step simulation metrics
    - AgentAction : Agent behavior logging
    - HealthIncident : Health-related events
    - SimulationConfig : Configuration management

Features
--------
- Efficient batch operations through DataLogger
- Transaction safety with rollback support
- Comprehensive error handling
- Multi-format data export (CSV, Excel, JSON, Parquet)
- Configuration management
- Performance optimized queries through DataRetriever

Usage Example
------------
>>> db = SimulationDatabase("simulation_results.db")
>>> db.save_configuration(config_dict)
>>> db.logger.log_step(step_number, agent_states, resource_states, metrics)
>>> db.export_data("results.csv")
>>> db.close()

Notes
-----
- Uses SQLite as the backend database
- Implements foreign key constraints
- Includes indexes for performance
- Supports concurrent access through session management
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker

from farm.database.data_retrieval import DataRetriever

from .data_logging import DataLogger
from .models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    Base,
    HealthIncident,
    ResourceModel,
    SimulationConfig,
    SimulationStepModel,
    ReproductionEventModel,
)
from .utilities import (
    create_database_schema,
    execute_with_retry,
    format_agent_state,
    format_position,
    safe_json_loads,
    validate_export_format,
)

logger = logging.getLogger(__name__)


class SimulationDatabase:
    """Database interface for simulation state persistence and analysis.

    This class provides a high-level interface for storing and retrieving simulation
    data using SQLAlchemy ORM. It handles all database operations including state
    logging, configuration management, and data analysis with transaction safety
    and efficient batch operations.

    Features
    --------
    - Buffered batch operations via DataLogger
    - Data querying via DataRetriever
    - Transaction safety with automatic rollback
    - Comprehensive error handling
    - Multi-format data export
    - Thread-safe session management

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file
    engine : sqlalchemy.engine.Engine
        SQLAlchemy database engine instance
    Session : sqlalchemy.orm.scoped_session
        Thread-local session factory
    logger : DataLogger
        Handles buffered data logging operations
    query : DataRetriever
        Handles data querying operations

    Methods
    -------
    update_agent_death(agent_id, death_time, cause)
        Update agent record with death information
    update_agent_state(agent_id, step_number, state_data)
        Update agent state in the database
    export_data(filepath, format, data_types, ...)
        Export simulation data to various file formats
    save_configuration(config)
        Save simulation configuration to database
    get_configuration()
        Retrieve current simulation configuration
    cleanup()
        Clean up database resources
    close()
        Close database connections

    Example
    -------
    >>> db = SimulationDatabase("simulation_results.db")
    >>> db.save_configuration({"agents": 100, "resources": 1000})
    >>> db.logger.log_step(1, agent_states, resource_states, metrics)
    >>> db.export_data("results.csv")
    >>> db.close()

    Notes
    -----
    - Uses SQLite as the backend database
    - Implements foreign key constraints
    - Creates required tables automatically
    - Handles concurrent access through thread-local sessions
    - Buffers operations through DataLogger for better performance
    """

    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance with SQLAlchemy.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file

        Notes
        -----
        - Enables foreign key support for SQLite
        - Creates session factory with thread-local scope
        - Initializes tables and indexes
        - Sets up batch operation buffers
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Enable foreign key support for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Create session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

        # Create tables and indexes
        self._create_tables()

        # Replace buffer initialization with DataLogger
        self.logger = DataLogger(self, buffer_size=1000)
        self.query = DataRetriever(self)

    def _execute_in_transaction(self, func: callable) -> Any:
        """Execute database operations within a transaction with error handling.

        Parameters
        ----------
        func : callable
            Function that takes a session argument and performs database operations

        Returns
        -------
        Any
            Result of the executed function

        Raises
        ------
        IntegrityError
            If there's a database constraint violation
        OperationalError
            If there's a database connection issue
        ProgrammingError
            If there's a SQL syntax error
        SQLAlchemyError
            For other database-related errors
        """
        session = self.Session()
        try:
            return execute_with_retry(session, lambda: func(session))
        finally:
            self.Session.remove()

    def close(self) -> None:
        """Close the database connection with enhanced error handling."""
        try:
            # Flush pending changes using DataLogger
            self.logger.flush_all_buffers()

            # Clean up sessions
            self.Session.remove()

            # Dispose engine connections
            if hasattr(self, "engine"):
                try:
                    self.engine.dispose()
                except SQLAlchemyError as e:
                    logger.error(f"Error disposing database engine: {e}")

        except SQLAlchemyError as e:
            logger.error(f"Database error during close: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database close: {e}")
        finally:
            # Ensure critical resources are released
            if hasattr(self, "Session"):
                try:
                    self.Session.remove()
                except Exception as e:
                    logger.error(f"Final cleanup error: {e}")

    def export_data(
        self,
        filepath: str,
        format: str = "csv",
        data_types: List[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        include_metadata: bool = True,
    ) -> None:
        """Export simulation data to various file formats with filtering options.

        Exports selected simulation data to a file in the specified format. Supports
        filtering by data type and time range, with optional metadata inclusion.

        Parameters
        ----------
        filepath : str
            Path where the export file will be saved
        format : str, optional
            Export format, one of: 'csv', 'excel', 'json', 'parquet'
            Default is 'csv'
        data_types : List[str], optional
            List of data types to export, can include:
            'metrics', 'agents', 'resources', 'actions'
            If None, exports all types
        start_step : int, optional
            Starting step number for data range
        end_step : int, optional
            Ending step number for data range
        include_metadata : bool, optional
            Whether to include simulation metadata, by default True

        Raises
        ------
        ValueError
            If format is unsupported or filepath is invalid
        SQLAlchemyError
            If there's an error retrieving data
        IOError
            If there's an error writing to the file
        """
        if not validate_export_format(format):
            raise ValueError(f"Unsupported export format: {format}")

        def _query(session):
            data = {}

            # Build step number filter
            step_filter = []
            if start_step is not None:
                step_filter.append(SimulationStepModel.step_number >= start_step)
            if end_step is not None:
                step_filter.append(SimulationStepModel.step_number <= end_step)

            # Default to all data types if none specified
            export_types = data_types or ["metrics", "agents", "resources", "actions"]

            # Collect requested data
            if "metrics" in export_types:
                metrics_query = session.query(SimulationStepModel)
                if step_filter:
                    metrics_query = metrics_query.filter(*step_filter)
                data["metrics"] = pd.read_sql(
                    metrics_query.statement, session.bind, index_col="step_number"
                )

            if "agents" in export_types:
                agents_query = (
                    session.query(AgentStateModel)
                    .join(AgentModel)
                    .options(joinedload(AgentStateModel.agent))
                )
                if step_filter:
                    agents_query = agents_query.filter(*step_filter)
                data["agents"] = pd.read_sql(agents_query.statement, session.bind)

            if "resources" in export_types:
                resources_query = session.query(ResourceModel)
                if step_filter:
                    resources_query = resources_query.filter(*step_filter)
                data["resources"] = pd.read_sql(resources_query.statement, session.bind)

            if "actions" in export_types:
                actions_query = session.query(ActionModel)
                if step_filter:
                    actions_query = actions_query.filter(*step_filter)
                data["actions"] = pd.read_sql(actions_query.statement, session.bind)

            # Add metadata if requested
            if include_metadata:
                config = self.get_configuration()
                data["metadata"] = {
                    "config": config,
                    "export_timestamp": datetime.now().isoformat(),
                    "data_range": {"start_step": start_step, "end_step": end_step},
                    "exported_types": export_types,
                }

            # Export based on format
            if format == "csv":
                # Create separate CSV files for each data type
                base_path = filepath.rsplit(".", 1)[0]
                for data_type, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(f"{base_path}_{data_type}.csv", index=False)
                    elif data_type == "metadata":
                        with open(f"{base_path}_metadata.json", "w") as f:
                            json.dump(df, f, indent=2)

            elif format == "excel":
                # Save all data types as separate sheets in one Excel file
                with pd.ExcelWriter(filepath) as writer:
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_excel(writer, sheet_name=data_type, index=False)
                        elif data_type == "metadata":
                            pd.DataFrame([df]).to_excel(
                                writer, sheet_name="metadata", index=False
                            )

            elif format == "json":
                # Convert all data to JSON format
                json_data = {
                    k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in data.items()
                }
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=2)

            elif format == "parquet":
                # Save as parquet format (good for large datasets)
                if len(data) == 1:
                    # Single dataframe case
                    next(iter(data.values())).to_parquet(filepath)
                else:
                    # Multiple dataframes case
                    base_path = filepath.rsplit(".", 1)[0]
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_parquet(f"{base_path}_{data_type}.parquet")

            else:
                raise ValueError(f"Unsupported export format: {format}")

        self._execute_in_transaction(_query)

    def update_agent_death(
        self, agent_id: str, death_time: int, cause: str = "starvation"
    ):
        """Update agent record with death information.

        Parameters
        ----------
        agent_id : str
            ID of the agent that died
        death_time : int
            Time step when death occurred
        cause : str, optional
            Cause of death, defaults to "starvation"
        """

        def _update(session):
            # Update agent record with death time
            agent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )

            if agent:
                agent.death_time = death_time

                # Log health incident for death
                health_incident = HealthIncident(
                    step_number=death_time,
                    agent_id=agent_id,
                    health_before=0,  # Death implies health reached 0
                    health_after=0,
                    cause=cause,
                    details=json.dumps({"final_state": "dead"}),
                )
                session.add(health_incident)

        self._execute_in_transaction(_update)

    def _load_agent_stats(self, conn, agent_id) -> Dict:
        """Load current agent statistics from database."""

        def _query(session):
            # Get latest state for the agent using lowercase table names
            latest_state = (
                session.query(
                    AgentStateModel.current_health,
                    AgentStateModel.resource_level,
                    AgentStateModel.total_reward,
                    AgentStateModel.age,
                    AgentStateModel.is_defending,
                    AgentStateModel.position_x,
                    AgentStateModel.position_y,
                    AgentStateModel.step_number,
                )
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number.desc())
                .first()
            )

            if latest_state:
                position = (latest_state[5], latest_state[6])
                return {
                    "current_health": latest_state[0],
                    "resource_level": latest_state[1],
                    "total_reward": latest_state[2],
                    "age": latest_state[3],
                    "is_defending": latest_state[4],
                    "current_position": format_position(position),
                }

            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": False,
                "current_position": format_position((0.0, 0.0)),
            }

        return self._execute_in_transaction(_query)

    def update_agent_state(self, agent_id: str, step_number: int, state_data: Dict):
        """Update agent state in the database.

        Parameters
        ----------
        agent_id : str
            ID of the agent to update
        step_number : int
            Current simulation step
        state_data : Dict
            Dictionary containing state data:
            - current_health: float
            - starting_health: float
            - resource_level: float
            - position: Tuple[float, float]
            - is_defending: bool
            - total_reward: float
            - starvation_threshold: int
        """

        def _update(session):
            # Get the agent to access its properties
            agent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return

            formatted_state = format_agent_state(agent_id, step_number, state_data)
            agent_state = AgentStateModel(**formatted_state)
            session.add(agent_state)

        self._execute_in_transaction(_update)

    def _create_tables(self):
        """Create the required database schema.

        Creates all tables defined in the SQLAlchemy models if they don't exist.
        Also creates necessary indexes for performance optimization.

        Raises
        ------
        SQLAlchemyError
            If there's an error creating the tables
        OperationalError
            If there's a database connection issue
        """

        def _create(session):
            create_database_schema(self.engine, Base)

        self._execute_in_transaction(_create)

    def update_notes(self, notes_data: Dict):
        """Update simulation notes in the database.

        Parameters
        ----------
        notes_data : Dict
            Dictionary of notes to update, where:
            - keys are note identifiers
            - values are note content objects

        Raises
        ------
        SQLAlchemyError
            If there's an error updating the notes
        """

        def _update(session):
            # Use merge instead of update to handle both inserts and updates
            for key, value in notes_data.items():
                session.merge(value)

        self._execute_in_transaction(_update)

    def cleanup(self):
        """Clean up database and GUI resources.

        Performs cleanup operations including:
        - Flushing all data buffers
        - Closing database connections
        - Disposing of the engine
        - Removing session

        This method should be called before application shutdown.

        Raises
        ------
        Exception
            If cleanup operations fail
        """
        try:
            # Flush any pending changes
            self.logger.flush_all_buffers()

            # Close database connections
            self.Session.remove()
            self.engine.dispose()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            # Ensure critical resources are released
            if hasattr(self, "Session"):
                self.Session.remove()

    def get_configuration(self) -> Dict:
        """Retrieve the simulation configuration from the database."""

        def _query(session):
            config = (
                session.query(SimulationConfig)
                .order_by(SimulationConfig.timestamp.desc())
                .first()
            )

            if config and config.config_data:
                return safe_json_loads(config.config_data) or {}
            return {}

        return self._execute_in_transaction(_query)

    def save_configuration(self, config: Dict) -> None:
        """Save simulation configuration to the database."""

        def _insert(session):
            import time

            config_obj = SimulationConfig(
                timestamp=int(time.time()), config_data=json.dumps(config)
            )
            session.add(config_obj)

        self._execute_in_transaction(_insert)

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_id: str = None,
        offspring_initial_resources: float = None,
        failure_reason: str = None,
        parent_position: tuple[float, float] = None,
        parent_generation: int = None,
        offspring_generation: int = None,
    ) -> None:
        """Log a reproduction event to the database.
        
        Parameters
        ----------
        step_number : int
            Current simulation step
        parent_id : str
            ID of parent agent attempting reproduction
        success : bool
            Whether reproduction was successful
        parent_resources_before : float
            Parent's resources before reproduction
        parent_resources_after : float
            Parent's resources after reproduction
        offspring_id : str, optional
            ID of created offspring (if successful)
        offspring_initial_resources : float, optional
            Resources given to offspring (if successful)
        failure_reason : str, optional
            Reason for failure (if unsuccessful)
        parent_position : tuple[float, float], optional
            Position where reproduction occurred
        parent_generation : int, optional
            Parent's generation number
        offspring_generation : int, optional
            Offspring's generation number (if successful)
        """
        def _log(session):
            event = ReproductionEventModel(
                step_number=step_number,
                parent_id=parent_id,
                offspring_id=offspring_id,
                success=success,
                parent_resources_before=parent_resources_before,
                parent_resources_after=parent_resources_after,
                offspring_initial_resources=offspring_initial_resources,
                failure_reason=failure_reason,
                parent_generation=parent_generation,
                offspring_generation=offspring_generation,
                parent_position_x=parent_position[0] if parent_position else None,
                parent_position_y=parent_position[1] if parent_position else None,
                timestamp=datetime.now()
            )
            session.add(event)

        self._execute_in_transaction(_log)
